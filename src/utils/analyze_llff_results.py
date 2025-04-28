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

from models.NeRF import NeRF
from Camera import Camera
from data.LLFFDataset import LLFFDataset
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

def render_from_model(model, rays_o, rays_d, device, N_samples=64, chunk_size=2048):
   """Render full image with chunking."""
   rays_o = rays_o.to(device)
   rays_d = rays_d.to(device)

   # Sample points along rays
   pts, z_vals = Camera.sample_points_along_rays(rays_o, rays_d, near=2.0, far=6.0, N_samples=N_samples, perturb=False)
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
   # rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

   return rgb_map

# def render_rotating_gif_llff_poses(model, dataset, save_dir, device, every_n=1, N_samples=64):
#    """Render a GIF using real camera poses from the LLFF dataset."""
      
#    frames = []
#    os.makedirs(save_dir, exist_ok=True)

#    print(f"Generating rotating view GIF from {len(dataset.poses)} poses...")

#    for i in range(0, len(dataset.poses), every_n):
#       sample = dataset[i]
#       rgb_gt, pose, focal, H, W, bounds = sample

#       # Only call `.to(device)` on tensors
#       pose = pose
      
#       cam = Camera(eye=pose[:, 3], target=None, focal=focal, H=H, W=W, c2w=pose)
#       rays_o, rays_d = cam.get_rays()
      
#       rgb_map = render_from_model(model, rays_o.to(device), rays_d.to(device), device)
      
#       img = (rgb_map.clamp(0.0, 1.0).detach().cpu().numpy() * 255).astype(np.uint8)
#       img = np.transpose(img, (1, 0, 2))
#       frames.append(Image.fromarray(img))
      
#       torch.cuda.empty_cache()

#    gif_path = os.path.join(save_dir, "rotating_llff.gif")
#    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
#    print(f"Saved LLFF trajectory GIF to {gif_path}")

def render_rotating_gif_llff_poses(model, dataset, save_dir, device, every_n=1, N_samples=64):
   """Render a GIF using real camera poses from the LLFF dataset."""
   
   # Extract all of the poses
   c2ws = torch.stack([torch.from_numpy(pose) for pose in dataset.poses]).to(device)
      
   frames = []
   os.makedirs(save_dir, exist_ok=True)

   print(f"Generating rotating view GIF from {len(dataset.poses)} poses...")

   frames = []
   steps = 40
   
   print(f"Steps {steps}, len(c2ws)-1 {len(c2ws) - 1}.  Quotient {steps // (len(c2ws) - 1)}")
   for i in range(len(c2ws) - 1):
      for j in range(steps // (len(c2ws) - 1)):
         t = j / (steps // (len(c2ws) - 1))
         
         c2w = interpolate_pose(c2ws[i], c2ws[i+1], t)
         c2w = torch.tensor(c2w)
         cam = Camera(eye=c2w[:, 3], target=None, H=dataset.H, W=dataset.W, focal=dataset.focal, c2w=c2w)
         
         rays_o, rays_d = cam.get_rays()
         rgb_map = render_from_model(model, rays_o, rays_d, device)
         
         img = (rgb_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
         img = np.transpose(img, (1, 0, 2))
         
         frames.append(Image.fromarray(img))
         
         torch.cuda.empty_cache()

   gif_path = os.path.join(save_dir, "rotating_llff.gif")
   frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
   print(f"Saved LLFF trajectory GIF to {gif_path}")

def evaluate_real_scene(result_dir, data_dir, image_idx=0, N_samples=64, near=2.0, far=6.0):
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
   model = NeRF().to(device)
   weights_dir = os.path.join(result_dir, "weights")
   weights = sorted([f for f in os.listdir(weights_dir) if f.endswith(".pt")])
   if not weights:
      print("No weights found.")
      return
   best_weights = weights[-1]
   model.load_state_dict(torch.load(os.path.join(weights_dir, best_weights), map_location=device))
   model.eval()

   # Load validation image and camera
   # dataset = LLFFDataset(scene_dir=data_dir, split='val')
   dataset = LLFFDataset(scene_dir=data_dir, split='train')
   # image_gt, pose, focal, H, W, bounds = dataset[image_idx]
   # image_gt_np = image_gt.cpu().numpy()

   # # Build camera
   # eye = pose[:, 3]
   # target = torch.tensor([0, 0, 0], dtype=torch.float32).to(device)
   # cam = Camera(eye=eye.to(device), target=target, focal=focal, H=image_gt.shape[1], W=image_gt.shape[0])
   # rays_o, rays_d = cam.get_rays()
   # # Render image from model
   # # rgb_map = render_from_model(model, cam, near, far, N_samples, device)
   # rgb_map = render_from_model(model, rays_o, rays_d, device)
   # rgb_np = rgb_map.detach().cpu().numpy().clip(0, 1)
   
   sample = dataset[0]
   rgb_gt, pose, focal, H, W, bounds = sample
   image_gt_np = rgb_gt.cpu().numpy()

   # Only call `.to(device)` on tensors
   rgb_gt = rgb_gt.to(device)
   pose = pose.to(device)
   
   cam = Camera(eye=pose[:, 3], target=None, focal=focal, H=H, W=W, c2w=pose)
   rays_o, rays_d = cam.get_rays()
   
   with torch.no_grad():
      rgb_map = render_from_model(model, rays_o, rays_d, device)
   
   img = (rgb_map.clamp(0.0, 1.0).detach().cpu().numpy() * 255).astype(np.uint8)
   img = np.transpose(img, (1, 0, 2))

   # Save outputs
   out_dir = os.path.join(result_dir, "plots", "real_eval")
   os.makedirs(out_dir, exist_ok=True)

   # GT image
   gt_img_uint8 = (image_gt_np * 255).astype(np.uint8)
   gt_img_uint8 = np.transpose(gt_img_uint8, (1, 0, 2))
   Image.fromarray(gt_img_uint8).save(os.path.join(out_dir, "gt_image.png"))

   # Prediction image
   pred_img_uint8 = (img).astype(np.uint8)
   Image.fromarray(pred_img_uint8).save(os.path.join(out_dir, "pred_image.png"))

   # Side-by-side
   side = np.concatenate([gt_img_uint8, pred_img_uint8], axis=1)
   Image.fromarray(side).save(os.path.join(out_dir, "comparison.png"))
   print(f"Saved evaluation images to {out_dir}")
   
   # Free GPU memory before gif rendering
   del rays_o, rays_d, rgb_map, img
   torch.cuda.empty_cache()

   gc.collect()
   
   # Gif
   with torch.no_grad():
      render_rotating_gif_llff_poses(model, dataset, save_dir=out_dir, device=device)
   
   # Learning curve
   log_path = os.path.join(result_dir, "log", "log.csv")
   plot_path = os.path.join(result_dir, "plots", "real_eval", "learning_curve.png")
   plot_learning_curve(log_path=log_path, save_path=plot_path)


if __name__ == '__main__':
   if len(sys.argv) < 2:
      print("Usage: python analyze_results_real.py <result_dir>")
      sys.exit(1)

   data_dir = "./data/llff/testscene"
   result_dir = sys.argv[1]
   evaluate_real_scene(result_dir, data_dir, image_idx=0, N_samples=64, near=2.0, far=6.0)