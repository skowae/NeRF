import os
import gc
import csv
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.NeRF import NeRF
from Camera import Camera
from get_rays import evaluate_fake_radiance_field_torus
from utils.analyze_results import *

def render_in_chunks_safe(model, pts_flat, chunk_size=2048):
   """
   Renders large sets of 3D points by processing them in smaller chunks
   to avoid out-of-memory errors on the GPU.
   """
   rgb_chunks = []
   sigma_chunks = []
   i = 0
   total = pts_flat.shape[0]

   while i < total:
      end = min(i + chunk_size, total)
      try:
         # Forward pass through NeRF for a chunk of points
         rgb_chunk, sigma_chunk = model(pts_flat[i:end])
         rgb_chunks.append(rgb_chunk)
         sigma_chunks.append(sigma_chunk)
         # print(f"rendered chunk {i}/{total}")
         i = end
      except torch.cuda.OutOfMemoryError:
         # On OOM, try splitting into smaller chunks recursively
         print(f"[OOM] on chunk {i}:{end}, retrying smaller")
         torch.cuda.empty_cache()
         gc.collect()
         if chunk_size <= 128:
               raise RuntimeError("Chunk size reduced too far. Still OOM.")
         sub_rgb, sub_sigma = render_in_chunks_safe(model, pts_flat[i:end], chunk_size // 2)
         rgb_chunks.append(sub_rgb)
         sigma_chunks.append(sub_sigma)
         i = end

   return torch.cat(rgb_chunks, dim=0), torch.cat(sigma_chunks, dim=0)

def generate_training_data(device, H, W, focal, near, far, N_samples, target):
   """
   Generate synthetic training data by rendering a torus from different viewpoints
   """
   images, rays_o_all, rays_d_all = [], [], []
   for angle in torch.linspace(0, 2 * torch.pi, steps=40):
      eye = torch.tensor([2 * torch.cos(angle), 2 * torch.sin(angle), 2.0], device=device)
      cam = Camera(eye=eye, target=target, focal=focal, H=H, W=W)
      rays_o, rays_d = cam.get_rays()
      pts, z_vals = cam.sample_points_along_rays(rays_o, rays_d, near, far, N_samples)
      rgb, sigma = evaluate_fake_radiance_field_torus(pts)
      rgb_map, _ = Camera.volume_render(rgb, sigma, z_vals)
      images.append(rgb_map.detach())
      rays_o_all.append(rays_o)
      rays_d_all.append(rays_d)

   return images, rays_o_all, rays_d_all

def render_preview_image(model, result_dir, target, focal, device, epoch):
   """
   Render a lower-resolution preview image from a fixed camera
   """
   H_render, W_render, N_render = 100, 100, 64
   eye = torch.tensor([2.0, 2.0, 2.0], device=device)
   cam = Camera(eye=eye, target=target, focal=focal, H=H_render, W=W_render)
   rays_o, rays_d = cam.get_rays()
   pts, z_vals = cam.sample_points_along_rays(rays_o, rays_d, 2.0, 6.0, N_render)
   pts_flat = pts.view(-1, 3)
   rgb_flat, sigma_flat = render_in_chunks_safe(model, pts_flat.to(device))

   rgb_out = rgb_flat.view(H_render, W_render, N_render, 3)
   sigma_out = sigma_flat.view(H_render, W_render, N_render)

   rgb_map, _ = Camera.volume_render(rgb_out, sigma_out, z_vals)
   rgb_map = rgb_map.clamp(0.0, 1.0).cpu().numpy()

   plt.imsave(f"{result_dir}/plots/epoch_{epoch:03d}.png", rgb_map)
   
   return cam.eye.detach().cpu(), rays_d.detach().cpu()


def train():
   try: 
      # Config
      epochs = 11
      
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      # Set up directories for saving logs, weights, and images
      timestamp = time.strftime("%Y%m%d-%H%M%S")
      result_dir = f"results/{timestamp}"
      os.makedirs(f"{result_dir}/log", exist_ok=True)
      os.makedirs(f"{result_dir}/weights", exist_ok=True)
      os.makedirs(f"{result_dir}/plots", exist_ok=True)
      log_path = f"{result_dir}/log/log.csv"

      # Initialize NeRF model and optimizer
      model = NeRF().to(device)
      optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

      # Define camera and ray sampling parameters
      H, W, focal = 100, 100, 100
      near, far, N_samples = 2.0, 6.0, 64
      target = torch.tensor([0.0, 0.0, 0.0], device=device)

      # Generate training data from synthetic torus
      images_gt, rays_o_all, rays_d_all = generate_training_data(device, H, W, focal, near, far, N_samples, target)

      # Initialize training log
      with open(log_path, 'w', newline='') as csvfile:
         writer = csv.writer(csvfile)
         writer.writerow(["epoch", "loss"])

      # Init best loss
      best_loss = float('inf')
      for epoch in range(epochs):
         model.train()
         total_loss = 0

         # Iterate over training examples (views)
         for rays_o, rays_d, rgb_gt in zip(rays_o_all, rays_d_all, images_gt):
            pts, z_vals = Camera.sample_points_along_rays(rays_o, rays_d, near, far, N_samples)
            pts_flat = pts.view(-1, 3)

            rgb_flat, sigma_flat = render_in_chunks_safe(model, pts_flat.to(device))
            rgb_out = rgb_flat.view(H, W, N_samples, 3)
            sigma_out = sigma_flat.view(H, W, N_samples)

            # dists = z_vals[..., 1:] - z_vals[..., :-1]
            # dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)
            # alpha = 1. - torch.exp(-sigma_out * dists)
            # weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
            # rgb_map = torch.sum(weights[..., None] * rgb_out, dim=-2)
            
            rgb_map, weights = Camera.volume_render(rgb_out, sigma_out, z_vals)
            # print("RGB mean:", rgb_map.mean().item(), "std:", rgb_map.std().item())


            loss = F.mse_loss(rgb_map, rgb_gt.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # total_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
            # print("Total grad:", total_grad)


         print(f"Epoch {epoch}: loss = {total_loss:.4f}")
         with open(log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, total_loss])
         
         if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), f"{result_dir}/weights/best.pt")

         if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                  eye_render, rays_d_render = render_preview_image(model, result_dir, target, focal, device, epoch)
      
      best_path = os.path.join(result_dir, 'weights', 'best.pt')
      model.load_state_dict(torch.load(best_path, map_location=device))
      print(f"Loaded best weights from {best_path} for analysis.")
      # Call post-training analysis
      analyze_result_dir(
         result_dir=result_dir,
         eye=eye_render,
         directions=rays_d_render,
         model=model,
         focal=focal,
         target=target,
         device=device
      )
      
      return result_dir

   except KeyboardInterrupt:
      print("Training interrupted by user.")
      
   finally:
      print("Cleaning up GPU memory...")
      torch.cuda.empty_cache()
      gc.collect()
      # Optionally delete large objects
      del model, optimizer, rays_o_all, rays_d_all, images_gt

if __name__ == '__main__':
   
   result_dir = train()