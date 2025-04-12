import os
import sys
import csv
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.image import imread
from PIL import Image
from Camera import Camera
import models.NeRF as nerf_model

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

def save_latest_preview_image(plots_dir, save_dir):
   """Save a copy of the most recent rendered preview image."""
   files = sorted([f for f in os.listdir(plots_dir) if f.endswith('.png') and 'epoch' in f])
   if not files:
      print("No preview images found.")
      return None, None

   latest = files[-1]
   src = os.path.join(plots_dir, latest)
   dst = os.path.join(save_dir, "latest_preview.png")
   img = imread(src)

   plt.figure(figsize=(5, 5))
   plt.imshow(img)
   plt.title(f"Latest Render: {latest}")
   plt.axis('off')
   plt.tight_layout()
   plt.savefig(dst)
   plt.close()
   print(f"Saved preview image to {dst}")

   return dst, img

def plot_3d_rays_overlay(save_dir, image, eye, directions, num_rays=10):
   """Overlay some sampled rays in 3D and save the result."""
   fig = plt.figure(figsize=(8, 6))
   ax = fig.add_subplot(111, projection='3d')

   # Plot some sampled rays
   for i in range(0, directions.shape[0], directions.shape[0] // num_rays):
      for j in range(0, directions.shape[1], directions.shape[1] // num_rays):
         d = directions[i, j]
         x, y, z = eye.cpu().numpy()
         dx, dy, dz = d.cpu().numpy()
         ax.quiver(x, y, z, dx, dy, dz, length=2.0, color='r')

   ax.set_xlim([-3, 3])
   ax.set_ylim([-3, 3])
   ax.set_zlim([-3, 3])
   ax.set_title("Example Ray Directions in 3D")
   plt.tight_layout()
   out_path = os.path.join(save_dir, "ray_overlay.png")
   plt.savefig(out_path)
   print(f"Saved 3D rays overlay to {out_path}")
   plt.close()

def render_rotating_gif(model, target, focal, device, save_dir, H=100, W=100, radius=2.5, steps=60):
   """Render multiple views by rotating the camera and save a GIF."""
   frames = []
   for theta in np.linspace(0, 2 * np.pi, steps):
      eye = torch.tensor([
         radius * np.cos(theta),
         0.0,
         radius * np.sin(theta)
      ], dtype=torch.float32)
      cam = Camera(eye=eye, target=target, focal=focal, H=H, W=W)
      rays_o, rays_d = cam.get_rays()
      pts, z_vals = Camera.sample_points_along_rays(rays_o.to(device), rays_d.to(device), 2.0, 6.0, 64)
      pts_flat = pts.view(-1, 3)
      with torch.no_grad():
         rgb_flat, sigma_flat = model(pts_flat)
      rgb = rgb_flat.view(H, W, 64, 3)
      sigma = sigma_flat.view(H, W, 64)
      weights = Camera.compute_weights(sigma, z_vals.to(device))
      rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
      img = (rgb_map.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
      frames.append(Image.fromarray(img))

   gif_path = os.path.join(save_dir, "rotating_view.gif")
   frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
   print(f"Saved rotating view GIF to {gif_path}")

def analyze_result_dir(result_dir, eye=None, directions=None, model=None, focal=None, target=None, device=None):
   """Run all default post-training analysis routines."""
   log_path = os.path.join(result_dir, "log", "log.csv")
   plots_dir = os.path.join(result_dir, "plots")
   analysis_dir = os.path.join(plots_dir, "analysis")
   os.makedirs(analysis_dir, exist_ok=True)

   print("\n--- Post-training Analysis ---")
   plot_loss_curve(log_path, analysis_dir)
   preview_path, img = save_latest_preview_image(plots_dir, analysis_dir)

   if preview_path and eye is not None and directions is not None:
      plot_3d_rays_overlay(analysis_dir, img, eye, directions)

   if all(x is not None for x in [model, focal, target, device]):
      render_rotating_gif(model, target, focal, device, analysis_dir)

if __name__ == '__main__':
   if len(sys.argv) != 2:
      print("Usage: python analyze_results.py <path_to_result_dir>")
      sys.exit(1)

   analyze_result_dir(sys.argv[1])
