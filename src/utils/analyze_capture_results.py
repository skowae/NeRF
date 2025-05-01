import os
import sys

# Add the parent directory (src/) to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import gc
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

from models.FastNGPNeRF import FastNGPNeRF
from Camera import Camera
from data.nerf_capture_dataset import NeRFCaptureDataset
from scipy.spatial.transform import Rotation as R, Slerp

def interpolate_pose(pose1, pose2, t):
   # Split into rotation + translation
   R1 = R.from_matrix(pose1[:3, :3].cpu().numpy())
   R2 = R.from_matrix(pose2[:3, :3].cpu().numpy())
   
   # SLERP setup
   key_times = [0, 1]
   key_rots = R.concatenate([R1, R2])
   slerp = Slerp(key_times, key_rots)
   R_interp = slerp(t).as_matrix()  # Interpolated rotation

   # Linearly interpolate translation
   T1 = pose1[:3, 3]
   T2 = pose2[:3, 3]
   T_interp = (1 - t) * T1 + t * T2

   # Construct full 3x4 interpolated pose
   pose_interp = np.eye(4, dtype=np.float32)
   pose_interp[:3, :3] = R_interp
   pose_interp[:3, 3] = T_interp.cpu().numpy()

   return pose_interp[:3]  # Return as 3x4

def renormalize_pose(pose, scene_center, scene_scale, already_unit=True):
   """Renormalized interpolated poses

   Args:
       pose (4,4): torch / np tensor in unit space (end-points already are)
       already_unit (bool, optional): if True we only snap back inside radius 1
       when ||t|| > 1. Defaults to True.
   Returns:
      pose_out (4, 4): tesnor with t in unit-sphere
   """
   if not torch.is_tensor(pose):
      pose = torch.tensor(pose)
      
   t = pose[:3, 3]
   if already_unit and t.norm() <= 1.0:
      print("Endpoint returning pose as is")
      return pose
   
   print("Renormalizing the pose")
   
   # Project back onto / inside the unit sphere
   t_world = t*scene_scale + scene_center
   t_unit = (t_world - scene_center)/scene_scale
   pose_out = pose.clone()
   pose_out[:3, 3] = t_unit
   return pose_out

def plot_learning_curve(log_path, save_path=None):
   """
   Plots the training and validation loss/SSIM/PSNR curves from the CSV log.
   
   Args:
      log_path (str): Path to CSV file.
      save_path (str): Where to save the plot (PNG). If None, shows the plot instead.
   """
   epochs = []
   train_losses, val_losses = [], []
   train_ssims, val_ssims = [], []
   train_psnrs, val_psnrs = [], []

   with open(log_path, 'r') as f:
      reader = csv.DictReader(f)
      for row in reader:
         epochs.append(int(row['epoch']))
         train_losses.append(float(row['train_loss']))
         val_losses.append(float(row['val_loss']))
         train_ssims.append(float(row['train_ssim']))
         val_ssims.append(float(row['val_ssim']))
         train_psnrs.append(float(row['train_psnr']))
         val_psnrs.append(float(row['val_psnr']))

   fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

   axs[0].plot(epochs, train_losses, label='Train Loss')
   axs[0].plot(epochs, val_losses, label='Val Loss')
   axs[0].set_ylabel("MSE Loss")
   axs[0].legend()
   axs[0].grid(True)

   axs[1].plot(epochs, train_ssims, label='Train SSIM')
   axs[1].plot(epochs, val_ssims, label='Val SSIM')
   axs[1].set_ylabel("SSIM")
   axs[1].legend()
   axs[1].grid(True)

   axs[2].plot(epochs, train_psnrs, label='Train PSNR')
   axs[2].plot(epochs, val_psnrs, label='Val PSNR')
   axs[2].set_xlabel("Epoch")
   axs[2].set_ylabel("PSNR")
   axs[2].legend()
   axs[2].grid(True)

   plt.tight_layout()

   if save_path:
      plt.savefig(save_path)
      print(f"Saved learning curve to {save_path}")
   else:
      plt.show()

   plt.close()

def render_from_model(model, rays_o, rays_d, device, near=0.1, far=4, N_samples=64, chunk_size=16384):
   """Render full image with chunking."""
   assert near is not None and far is not None, "Must Pass near and far explicitly"
   
   rays_o = rays_o.to(device)
   rays_d = rays_d.to(device)

   # Sample points along rays
   pts, z_vals = Camera.sample_points_along_rays(rays_o, rays_d, near=near, far=far, N_samples=N_samples, perturb=False)
   view_dirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
   view_dirs_expanded = view_dirs[..., None, :].expand_as(pts)

   # Flatten for chunked forward
   pts_flat = pts.reshape(-1, 3)
   view_dirs_flat = view_dirs_expanded.reshape(-1, 3)

   # Chunked rendering
   rgb_chunks = []
   sigma_chunks = []

   for i in range(0, pts_flat.shape[0], chunk_size):
      end = i + chunk_size
      rgb_chunk, sigma_chunk = model(pts_flat[i:end], view_dirs_flat[i:end])
      rgb_chunks.append(rgb_chunk)
      sigma_chunks.append(sigma_chunk)
      # print(f"Chunk {i}/{pts_flat.shape[0]}")

   rgb = torch.cat(rgb_chunks, dim=0).view(*pts.shape)
   sigma = torch.cat(sigma_chunks, dim=0).view(*pts.shape[:3])

   # print(f"render_image: shape of rgb {rgb.shape}, shape of sigma {sigma.shape}, z_vals {z_vals.shape}")
   rgb_map, weights = Camera.volume_render(rgb, sigma, z_vals.to(device))
   # rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)already_unit
   
   # print("EVAL")
   # print("RGB output stats:", rgb_map.min().item(), rgb_map.max().item())
   # print("Sigma output stats:", sigma.min().item(), sigma.max().item())

   return rgb_map


def render_rotating_gif_capture_poses(model, dataset, save_dir, device, near=0.1, far=4.0, every_n=1, N_samples=64):
   """Render a GIF using real camera poses from the LLFF dataset."""
   
   # Extract all of the poses
   c2ws = torch.stack([pose.squeeze(0) for pose in dataset.poses]).to(device)
   num_poses = len(c2ws)
   
   # Select indices to interpolate
   desired_num_poses = min(100, num_poses)
   selected_indices = torch.linspace(0, num_poses - 1, steps=desired_num_poses)
   frames = []
   os.makedirs(save_dir, exist_ok=True)

   print(f"Generating rotating view GIF from {len(dataset.poses)} poses...")

   frames = []
   steps = 99
   
   print(f"Steps {steps}, len(selected_c2ws)-1 {len(selected_indices) - 1}.  Quotient {steps // (len(selected_indices) - 1)}")
   for i in range(len(selected_indices) - 1):
      c2w_start = c2ws[int(selected_indices[i])]
      c2w_end = c2ws[int(selected_indices[i + 1])]
      
      img, pose, focal, pp, img_wh = dataset[int(selected_indices[i])]
      fl_x, fl_y = focal.squeeze(0).tolist()
      cx, cy = pp.squeeze(0).tolist()
      w, h = img_wh.squeeze(0).tolist()
      
      # print("Start eye:", eye, "-> points at origin?", np.linalg.norm(eye) < 1e-3, "Norm:", np.linalg.norm(eye))
      for j in range(steps // (len(selected_indices) - 1)):
         t = j / (steps // (len(selected_indices) - 1))
         
         c2w = interpolate_pose(c2w_start, c2w_end, t)
         c2w = torch.tensor(c2w)
         c2w = renormalize_pose(c2w, dataset.scene_center, dataset.scene_scale,
                                already_unit=(t in [0.0, 1.0]))
         
         forward = -c2w[:3, 2]
         eye = c2w[:3, 3]
         target = eye + forward
         
         cam = Camera(eye=eye, 
                      target=target, 
                      H=h, 
                      W=w, 
                      focal=(fl_x, fl_y), 
                      c2w=c2w)
         
         with torch.no_grad():
            rays_o, rays_d = cam.get_rays()
            rgb_map = render_from_model(model, rays_o, rays_d, device, near, far)
            
            img = (rgb_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)         
            frames.append(Image.fromarray(img))
         
         del rays_o, rays_d, rgb_map
         torch.cuda.empty_cache()

   gif_path = os.path.join(save_dir, "rotating_capture.gif")
   frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=300, loop=0)
   print(f"Saved LLFF trajectory GIF to {gif_path}")

def evaluate_real_scene(result_dir, data_dir, image_idx=0, N_samples=64, near=0.1, far=4.0):
   """
   Load model and LLFF dataset scene and evaluate on a validation image.

   Parameters:
   - result_dir: directory with saved model weights
   - data_dir: root LLFF data directory
   - scene: which scene to evaluate (default = 'testscene')
   - image_idx: which validation image index to evaluate
   """
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # Load model and weights
   model = FastNGPNeRF().to(device)
   weights_dir = os.path.join(result_dir, "weights")
   weights = sorted([f for f in os.listdir(weights_dir) if f.endswith(".pt")])
   if not weights:
      print("No weights found.")
      print("Cleaning up GPU memory...")
      torch.cuda.empty_cache()
      gc.collect()
      # Optionally delete large objects
      del model
      return
   best_weights = weights[-1]
   model.load_state_dict(torch.load(os.path.join(weights_dir, best_weights), map_location=device))
   model.eval()

   dataset = NeRFCaptureDataset(scene_dir=data_dir, split='train', subset=None)
   
   sample = dataset[0]
   rgb_gt, pose, focal, pp, img_wh = sample
            
   fl_x, fl_y = focal.squeeze(0).tolist()
   cx, cy = pp.squeeze(0).tolist()
   W, H = img_wh.squeeze(0).tolist()
         
   image_gt_np = rgb_gt.cpu().numpy()

   # Only call `.to(device)` on tensors
   rgb_gt = rgb_gt.to(device)
   pose = pose.to(device)
   
   cam = Camera(eye=pose[:, 3], target=None, focal=(fl_x, fl_y), H=H, W=W, c2w=pose, pp=(cx, cy))
   rays_o, rays_d = cam.get_rays()
   
   with torch.no_grad():
      rgb_map = render_from_model(model, rays_o, rays_d, device, near, far)
   
   # Save outputs
   out_dir = os.path.join(result_dir, "plots", "real_eval")
   os.makedirs(out_dir, exist_ok=True)

   # GT image
   gt_img_uint8 = (image_gt_np.transpose(1, 2, 0) * 255).astype(np.uint8)
   Image.fromarray(gt_img_uint8).save(os.path.join(out_dir, "gt_image.png"))

   # Prediction image
   pred_img_uint8 = (rgb_map.clamp(0.0, 1.0).detach().cpu().numpy() * 255).astype(np.uint8)
   Image.fromarray(pred_img_uint8).save(os.path.join(out_dir, "pred_image.png"))

   # Side-by-side
   side = np.concatenate([gt_img_uint8, pred_img_uint8], axis=1)
   Image.fromarray(side).save(os.path.join(out_dir, "comparison.png"))
   print(f"Saved evaluation images to {out_dir}")
   
   # Free GPU memory before gif rendering
   del rays_o, rays_d, rgb_map, pred_img_uint8
   torch.cuda.empty_cache()

   gc.collect()
   
   # Gif
   with torch.no_grad():
      render_rotating_gif_capture_poses(model, dataset, save_dir=out_dir, device=device, near=near, far=far)
   
   # Learning curve
   log_path = os.path.join(result_dir, "log", "log.csv")
   plot_path = os.path.join(result_dir, "plots", "real_eval", "learning_curve.png")
   plot_learning_curve(log_path=log_path, save_path=plot_path)


if __name__ == '__main__':
   if len(sys.argv) < 2:
      print("Usage: python analyze_results_real.py <result_dir>")
      sys.exit(1)

   data_dir = "./data/skow/robot"
   result_dir = sys.argv[1]
   evaluate_real_scene(result_dir, data_dir, image_idx=0, N_samples=64, near=0.05, far=1.2)