import os
import sys

# Add the parent directory (src/) to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import gc
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (keeps IDEs happy)
import numpy as np
from PIL import Image

from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

from models.FastNGPNeRF import FastNGPNeRF
from Camera import Camera
from data.nerf_capture_dataset import NeRFCaptureDataset
from scipy.spatial.transform import Rotation as R, Slerp

import imageio.v3 as iio

def estimate_pivot(poses):
   """Estimates the point the camera poses are looking at

   Args:
       poses (N, 3, 4): unit space, returns (3,) pivot in unit-space
   """
   
   C = poses[:, :3, 3] # Camera centers (N, 3)
   D = -poses[:, :3, 2] # forward (unit) (N, 3)
   
   # Solve argmin_p sum(||(p - Ci) - [(p-Ci)*Di]Di||^2)
   A = torch.eye(3).to(D) - D.unsqueeze(-1)*D.unsqueeze(-2) # (N, 3, 3)
   b = (A@C.unsqueeze(-1)).squeeze(-1)
   pivot = torch.linalg.lstsq(A.sum(0), b.sum(0)).solution # (3,)
   
   return pivot

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

def turntable_pose(c2w_start, pivot, t):
   """Rotate the whole camera pose c2w_start around the pivot by 360deg*t while 
   keeping the camera's local orientation consistent.  All tensors are (3,) or 
   (3,3) and already in unit sphere coords. 
   
   Returns a (4, 4) pose
   """
   
   R_0 = c2w_start[:3, :3]
   eye_0 = c2w_start[:3, 3]
   
   # Rotation about global +Y
   rot_axis = torch.tensor([0.0, 1.0, 0.], device=eye_0.device, dtype=eye_0.dtype)
   theta = 2*torch.pi*t
   R_y = Camera.axis_angle_to_matrix(rot_axis, torch.tensor(theta)) # (3, 3)
   
   # new eye = pivot + R_y * (eye+0 - pivot)
   radius_vec = eye_0 - pivot
   eye_new = pivot + R_y@radius_vec
   
   # rotate the local frame the same way: R_new = R_Y*R_0
   R_new = R_y@R_0
   
   return torch.cat([R_new, eye_new.unsqueeze(-1)], dim=-1) # (3,4)

def plot_cameras_with_dirs(train_poses, gif_poses, out_dir, obj_center=None, stride=4):
   """
   train_poses, gif_poses : list[torch.Tensor]  each (3,4) in unit-space
   obj_center             : (3,) tensor or None - centre of object (defaults 0)
   stride                 : plot every *stride* dir arrow to reduce clutter
   """
   if obj_center is None:
      obj_center = torch.zeros(3)

   fig = plt.figure(figsize=(6,5))
   ax = fig.add_subplot(111, projection='3d')
   ax.set_box_aspect([1,1,1])

   # ---- centres -----
   C_train = torch.stack([p[:3,3] for p in train_poses]).cpu()
   C_gif   = torch.stack([p[:3,3] for p in gif_poses  ]).cpu()
   ax.scatter(C_train[:,0], C_train[:,1], C_train[:,2], s=4, c='blue', label='train')
   ax.scatter(C_gif[:,0],   C_gif[:,1],   C_gif[:,2],   s=8, c='red',  label='gif')

   # ---- direction arrows (every `stride`th) ----
   for i, p in enumerate(train_poses[::stride]):
      c = p[:3,3].cpu(); dir = (-p[:3,2]).cpu()*0.1   # -Z, shorten
      ax.quiver(c[0], c[1], c[2], dir[0], dir[1], dir[2],
                  length=5, color='blue', linewidth=0.6)
   for i, p in enumerate(gif_poses[::stride]):
      c = p[:3,3].cpu(); dir = (-p[:3,2]).cpu()*0.1
      ax.quiver(c[0], c[1], c[2], dir[0], dir[1], dir[2],
                  length=1, color='red',  linewidth=1.0)
   
   # ---- Up-vectors (optional diagnostic) ----
   for i, p in enumerate(train_poses[::stride]):
      c  = p[:3,3].cpu()
      up =  p[:3,1].cpu()*0.1                 # +Y axis of camera
      ax.quiver(c[0], c[1], c[2], up[0], up[1], up[2],
               length=1, color='cyan', linewidth=0.4)
   for i, p in enumerate(gif_poses[::stride]):
      c  = p[:3,3].cpu()
      up =  p[:3,1].cpu()*0.1
      ax.quiver(c[0], c[1], c[2], up[0], up[1], up[2],
               length=1, color='magenta', linewidth=0.7)


   # optional: plot object centre
   ax.scatter([obj_center[0]],[obj_center[1]],[obj_center[2]],
               c='k', s=30, marker='x', label="object")
   ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
   
   ax.set_xlim3d([-1, 1])
   ax.set_ylim3d([-1, 1])
   ax.set_zlim3d([-1, 1])
    
   ax.legend()
   fig.tight_layout()
   fig.subplots_adjust(top=0.92)
   fig.suptitle("Camera Views")
   plt.savefig(os.path.join(out_dir, "cameras.png"), dpi=300, bbox_inches='tight')
   plt.show()

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

def xyz_to_idx(x, y, z, res=32):
   return x + y*res + z*res*res

def get_density_grid(model, device, grid_res):
   coords = torch.stack(torch.meshgrid(
      torch.arange(grid_res, device=device),
      torch.arange(grid_res, device=device),
      torch.arange(grid_res, device=device), indexing='ij'), dim=-1).view(-1, 3)
   density_grid = torch.zeros(grid_res**3, dtype=torch.uint8, device=device)
   with torch.no_grad():
      for i in range(0, coords.shape[0], 65536):
         pts = ((coords[i:i + 65536].float()/(grid_res - 1))*2 - 1)
         sigma = model.query_density(pts)
         mask = sigma > 1.0
         density_grid[xyz_to_idx(coords[i:i +65536, 0],
                                 coords[i:i +65536, 1],
                                 coords[i:i +65536, 2])] |= mask
   
   return density_grid

def render_from_model(model, rays_o, rays_d, grid_res, density_grid, device, 
                      near=None, far=None, N_samples=64, chunk_size=16384):
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
   
   cell_size = 1.0/grid_res
   x, y, z = ((pts_flat + 1)/2 / cell_size).long().T.unbind(0)
   idx = xyz_to_idx(x, y, z)
   mask = (density_grid[idx] == 1).to(device) # only sample when true
   sigma.view(-1)[~mask] = 0.0

   # print(f"render_image: shape of rgb {rgb.shape}, shape of sigma {sigma.shape}, z_vals {z_vals.shape}")
   rgb_map, weights = Camera.volume_render(rgb, sigma, z_vals.to(device))
   # rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
   
   # print("RGB output stats:", rgb_map.min().item(), rgb_map.max().item())
   # print("Sigma output stats:", sigma.min().item(), sigma.max().item())

   return rgb_map

def render_one_turntable(model, dataset, device, out_dir, grid_res, density_grid, num_frames=120):
   model.eval()
   with torch.no_grad():
      
      # Obtain the pivot
      poses_unit = torch.stack(dataset.poses).squeeze(1).to(device)
      pivot = estimate_pivot(poses_unit)
      
      # Anchor the pose and intrinisics
      dists = (poses_unit[:, :3, 3] - pivot).norm(dim=1)
      idx_anchor = torch.argmax(dists).item()
      
      img0, pose0, focal, pp, img_wh = dataset[idx_anchor]
      c2w0 = pose0.squeeze(0).to(device)  # (3, 4) already unit sphere
      fl_x, fl_y, = focal.squeeze(0).tolist()
      cx, cy = pp.squeeze(0).tolist()
      W, H = [int(v) for v in img_wh.squeeze(0).tolist()]
      
      # Single frame test
      test_t = 0.004
      c2w_test = turntable_pose(c2w0, pivot, test_t)
      cam_test = Camera(eye=c2w_test[:,3], target=None, c2w=c2w_test,
                        H=H, W=W, focal=(fl_x, fl_y), pp=(cx, cy))
      rays_o, rays_d = cam_test.get_rays()
      rgb_map = render_from_model(model, rays_o, rays_d, grid_res, density_grid, 
                                  device, near=0.05, far=1.20)
      Image.fromarray((rgb_map.clamp(0,1).cpu().numpy()*255).astype(np.uint8)
                     ).save(os.path.join(out_dir, "novel_test.png"))
      
      frames = []
      train_poses = [p.squeeze(0).to(device) for p in dataset.poses] # already unit
      gif_poses = []
      for k in range(num_frames):
         t = k/num_frames
         theta = 2*torch.pi*t
         R_y = Camera.axis_angle_to_matrix(torch.tensor([0., 1., 0., ], device=device),
                                           theta=torch.tensor(theta, device=device))
         eye_new = pivot + R_y@(c2w0[:3, 3] - pivot)
         R_new = R_y@c2w0[:3, :3]
         c2w_new = torch.cat([R_new, eye_new.unsqueeze(-1)], dim=-1)
         gif_poses.append(c2w_new.cpu())
         
         cam = Camera(eye=c2w_new[:, 3], target=None, c2w=c2w_new, H=H, W=W, 
                      focal=(fl_x, fl_y), pp=(cx, cy))
         
         rays_o, rays_d = cam.get_rays() # (H, W, 3) each
         
         # print("mean |rays_d| :", rays_d.norm(dim=-1).mean().item())
         
         #cam.visualize_corner_rays(rays_o.cpu(), rays_d.cpu(), pivot.cpu())

         rgb_map = render_from_model(model, rays_o, rays_d, grid_res, density_grid, device, near=0.05, far=1.20)
         
         frame_np = (rgb_map.clamp(0, 1).cpu().numpy()*255).astype(np.uint8)
         frames.append(Image.fromarray(frame_np))
      
      gif_path = os.path.join(out_dir, "turntable.gif")
      frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
      print("saved", gif_path)
      
      plot_cameras_with_dirs(train_poses, gif_poses, out_dir, 
                             obj_center=pivot.cpu(), stride=6)

def evaluate_real_scene(result_dir, data_dir, grid_res=32, image_idx=0, N_samples=64, near=0.05, far=1.2):
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
   model.load_state_dict(torch.load(os.path.join(weights_dir, "best_model.pt"), map_location=device))
   model.eval()
   
   density_grid = get_density_grid(model, device, grid_res)

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
      rgb_map = render_from_model(model, rays_o, rays_d, grid_res, density_grid, device, near, far)
   
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
      # render_rotating_gif_capture_poses(model, dataset, save_dir=out_dir, device=device, near=near, far=far)
      render_one_turntable(model, dataset, device, out_dir, grid_res, 
                           density_grid, num_frames=500)
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
   evaluate_real_scene(result_dir, data_dir, grid_res=32, image_idx=0, N_samples=64, near=0.05, far=1.2)