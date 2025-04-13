import os
import sys

# Add the parent directory (src/) to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.image import imread
from PIL import Image
from Camera import Camera
from models.NeRF import NeRF
from utils.get_rays import *

def plot_loss_curve(log_path, save_dir):
   """Plot the training loss from CSV log."""
   epochs, losses = [], []
   with open(log_path, 'r') as f:
      reader = csv.reader(f)
      next(reader)  # skip header
      for row in reader:
         epochs.append(int(row[0]))
         losses.append(float(row[1]))

   plt.figure(figsize=(8, 4))
   plt.plot(epochs, losses, label='Loss', color='blue')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('Training Loss Curve')
   plt.grid(True)
   plt.legend()
   plt.tight_layout()
   out_path = os.path.join(save_dir, "loss_curve.png")
   plt.savefig(out_path)
   print(f"Saved loss curve to {out_path}")
   plt.close()

def render_rotating_gif(model, target, focal, device, save_dir, H=100, W=100, N=64, radius=2.0, steps=60):
   """Render multiple views by rotating the camera and save a GIF."""
   frames = []
   target = target.to(device)
   for theta in np.linspace(0, 2 * np.pi, steps):
      eye = torch.tensor([
         radius * np.cos(theta),  # X position
         radius * np.sin(theta),  # Y position
         2.0                      # Fixed Z height
      ], dtype=torch.float32, device=device)
      cam = Camera(eye=eye, target=target, focal=focal, H=H, W=W)
      rays_o, rays_d = cam.get_rays()
      
      pts, z_vals = Camera.sample_points_along_rays(rays_o.to(device), rays_d.to(device), 2.0, 6.0, N)
      pts_flat = pts.view(-1, 3)
      
      # Get the view directions
      view_dirs = rays_d / rays_d.norm(dim=-1, keepdim=True)  # shape: [H, W, 3]
      view_dirs_flat = view_dirs.view(-1, 3)

      # Expand view directions to match N_samples along rays
      view_dirs_expanded = view_dirs[:, :, None, :].expand(H, W, N, 3)
      view_dirs_flat = view_dirs_expanded.reshape(-1, 3)

      with torch.no_grad():
         rgb_flat, sigma_flat = model(pts_flat, view_dirs_flat)
      rgb = rgb_flat.view(H, W, N, 3)
      sigma = sigma_flat.view(H, W, N)
      rgb_map, weights = Camera.volume_render(rgb, sigma, z_vals.to(device))
      # rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
      img = (rgb_map.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
      frames.append(Image.fromarray(img))

   gif_path = os.path.join(save_dir, "rotating_view.gif")
   frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
   print(f"Saved rotating view GIF to {gif_path}")

@torch.no_grad()
def render_model_image(model, cam, device, N_samples):
   rays_o, rays_d = cam.get_rays()
   H, W = rays_o.shape[:2]

   pts, z_vals = Camera.sample_points_along_rays(rays_o, rays_d, near=2.0, far=6.0, N_samples=N_samples)
   pts_flat = pts.view(-1, 3)

   view_dirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
   view_dirs_expanded = view_dirs[:, :, None, :].expand(H, W, N_samples, 3)
   view_dirs_flat = view_dirs_expanded.reshape(-1, 3)

   rgb_flat, sigma_flat = model(pts_flat.to(device), view_dirs_flat.to(device))
   rgb = rgb_flat.view(H, W, N_samples, 3)
   sigma = sigma_flat.view(H, W, N_samples)
   rgb_map, _ = Camera.volume_render(rgb, sigma, z_vals.to(device))
   return (rgb_map.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)

def render_gt_image(cam_preview, device, N_render):
   # Use the final eye and target used for preview image generation
   rays_o, rays_d = cam_preview.get_rays()
   pts, z_vals = Camera.sample_points_along_rays(rays_o.to(device), rays_d.to(device), 2.0, 6.0, N_render)
   rgb_gt_preview, sigma_gt_preview = evaluate_fake_radiance_field_complex(pts)
   rgb_map_gt, weights = Camera.volume_render(rgb_gt_preview, sigma_gt_preview, z_vals)

   # Save the ground truth image
   gt_img = (rgb_map_gt.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
   return gt_img

def evaluate_trained_model(result_dir, model, cam, device, N_samples=64):
   model = model.to(device)
   model.eval()

   output_dir = os.path.join(result_dir, "plots", "evaluation")
   os.makedirs(output_dir, exist_ok=True)

   # Render GT and model output
   print("Rendering ground truth image...")
   gt_img = render_gt_image(cam, device, N_samples)
   Image.fromarray(gt_img).save(os.path.join(output_dir, "ground_truth.png"))

   print("Rendering model image...")
   pred_img = render_model_image(model, cam, device, N_samples)
   Image.fromarray(pred_img).save(os.path.join(output_dir, "model_render.png"))

   # Plot loss
   print("Plotting loss curve...")
   log_path = os.path.join(result_dir, "log", "log.csv")
   plot_loss_curve(log_path, output_dir)

   # Render rotating gif
   print("Rendering rotating gif...")
   render_rotating_gif(model, target=cam.target, focal=cam.focal, device=device, save_dir=output_dir,
                     H=cam.H, W=cam.W, N=N_samples, steps=60)

   print(f"Evaluation complete. Outputs saved in {output_dir}")

if __name__ == '__main__':
   if len(sys.argv) != 2:
      print("Usage: python analyze_results.py <path_to_result_dir>")
      sys.exit(1)

   H=100
   W=100
   focal=100
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   eye = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32, device=device)
   target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

   cam = Camera(eye=eye, target=target, focal=focal, H=H, W=W)
   
   model = NeRF().to(device)
   model.load_state_dict(torch.load('/home/skowae1/JHU_EP/dlcv/NeRF/src/results/20250412-115020/weights/best.pt', map_location=device))
   
   evaluate_trained_model(
      sys.argv[1],
      model,
      cam,
      device,
      N_samples=64
   )
