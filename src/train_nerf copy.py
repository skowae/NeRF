import os
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from models.NeRF import NeRF
from Camera import Camera
from get_rays import evaluate_fake_radiance_field_torus
from pathlib import Path
from datetime import datetime
import csv
import gc

# -------------------- CONFIG --------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

H, W = 100, 100
focal = 60.0
N_samples = 64
near, far = 0.0, 6.0
lr = 5e-4
epochs = 100
views = 10


# Device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- OUTPUT DIR --------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path("results") / timestamp
log_dir = run_dir / "log"
weights_dir = run_dir / "weights"
plots_dir = run_dir / "plots"
for d in [log_dir, weights_dir, plots_dir]:
   d.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "metrics.csv"

# -------------------- GROUND TRUTH --------------------
def generate_training_data():
   """Generates multi-view images and rays for the torus scene."""
   poses = []
   images = []
   rays_origins = []
   rays_directions = []

   for i in range(views):
      angle = 2 * np.pi * i / views
      eye = torch.tensor([2.0 * np.sin(angle), 1.5, 2.0 * np.cos(angle)], dtype=torch.float32, device=device)
      target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
      cam = Camera(eye=eye, target=target, focal=focal, H=H, W=W)
      rays_o, rays_d = cam.get_rays()
      pts, z_vals = cam.sample_points_along_rays(rays_o, rays_d, near, far, N_samples)
      rgb, _ = evaluate_fake_radiance_field_torus(pts)
      images.append(rgb)
      poses.append(cam.c2w)
      rays_origins.append(rays_o)
      rays_directions.append(rays_d)

   return torch.stack(images), torch.stack(rays_origins), torch.stack(rays_directions), torch.stack(poses)

# --------------- CHUNKS -----------------------
def render_in_chunks(model, pts_flat, chunk_size=2048):
   rgb_chunks = []
   sigma_chunks = []
   for i in range(0, pts_flat.shape[0], chunk_size):
      print(f"rendering chunks {i}/{pts_flat.shape[0]}")
      rgb_chunk, sigma_chunk = model(pts_flat[i:i+chunk_size])
      rgb_chunks.append(rgb_chunk)
      sigma_chunks.append(sigma_chunk)
   return torch.cat(rgb_chunks, dim=0), torch.cat(sigma_chunks, dim=0)

def render_in_chunks_safe(model, pts_flat, chunk_size=2048):
   rgb_chunks = []
   sigma_chunks = []
   i = 0
   total = pts_flat.shape[0]

   while i < total:
      end = min(i + chunk_size, total)
      try:
         rgb_chunk, sigma_chunk = model(pts_flat[i:end])
         rgb_chunks.append(rgb_chunk)
         sigma_chunks.append(sigma_chunk)
         print(f"rendered chunk {i}/{total}")
         i = end
      except torch.cuda.OutOfMemoryError:
         print(f"[OOM] on chunk {i}:{end}, retrying smaller")
         torch.cuda.empty_cache()
         gc.collect()
         if chunk_size <= 128:
               raise RuntimeError("Chunk size reduced too far. Still OOM.")
         # Retry same chunk with smaller size
         sub_rgb, sub_sigma = render_in_chunks_safe(model, pts_flat[i:end], chunk_size // 2)
         rgb_chunks.append(sub_rgb)
         sigma_chunks.append(sub_sigma)
         i = end  # advance

   return torch.cat(rgb_chunks, dim=0), torch.cat(sigma_chunks, dim=0)



# -------------------- TRAINING --------------------
def train():
   model = NeRF().to(device)
   optimizer = torch.optim.Adam(model.parameters(), lr=lr)

   images_gt, rays_o_all, rays_d_all, _ = generate_training_data()
   images_gt = images_gt.to(device)
   rays_o_all = rays_o_all.to(device)
   rays_d_all = rays_d_all.to(device)
   print("I just ran generate training data")

   best_loss = float('inf')
   with open(log_path, 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(["epoch", "loss"])

   for epoch in range(1, epochs + 1):
      model.train()
      total_loss = 0

      for view in range(views):
         print("inside view loop")
         rays_o = rays_o_all[view]
         rays_d = rays_d_all[view]
         target_rgb = images_gt[view]

         pts, z_vals = Camera.sample_points_along_rays(rays_o, rays_d, near, far, N_samples)
         print("I just ran sample points")
         pts_flat = pts.view(-1, 3)
         rgb_flat, sigma_flat = render_in_chunks_safe(model, pts_flat.to(device))
         print("I just ran render in chunks")
         rgb = rgb_flat.view(H, W, N_samples, 3)
         sigma = sigma_flat.view(H, W, N_samples)
         
         rendered_img = Camera.volume_render(rgb, sigma, z_vals.to(device))
         print("I just ran volume render")
         
         loss = F.mse_loss(rendered_img, target_rgb)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

         total_loss += loss.item()
         
         # Clean up
         gc.collect()
         torch.cuda.empty_cache()


      avg_loss = total_loss / views
      print(f"Epoch {epoch}: loss = {avg_loss:.6f}")

      with open(log_path, 'a', newline='') as f:
         writer = csv.writer(f)
         writer.writerow([epoch, avg_loss])

      if avg_loss < best_loss:
         best_loss = avg_loss
         torch.save(model.state_dict(), weights_dir / "best.pt")

      # Plot output image
      with torch.no_grad():
         eye = torch.tensor([2.0, 1.5, 2.0], dtype=torch.float32, device=device)
         target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
         cam = Camera(eye, target, focal, H, W)
         rays_o, rays_d = cam.get_rays()
         pts, z_vals = cam.sample_points_along_rays(rays_o, rays_d, near, far, N_samples)
         pts_flat = pts.view(-1, 3)
         rgb_flat, sigma_flat = render_in_chunks_safe(model, pts_flat.to(device))
         print("I just ran render chunks for vis")
         rgb = rgb_flat.view(H, W, N_samples, 3)
         sigma = sigma_flat.view(H, W, N_samples)
         img = Camera.volume_render(rgb, sigma, z_vals.to(device)).cpu().numpy()
         print("I just ran render chunks for vis")
         plt.imsave(plots_dir / f"epoch_{epoch:04}.png", img)
      
      torch.cuda.empty_cache()

if __name__ == "__main__":
   train()
