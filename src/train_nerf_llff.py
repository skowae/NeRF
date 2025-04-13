
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.NeRF import NeRF
from data.LLFFDataset import LLFFDataset
from utils.positional_encoding import PositionalEncoding
from utils.analyze_llff_results import *
from tqdm import tqdm
from datetime import datetime
import numpy as np
import csv
from skimage.metrics import structural_similarity as ssim
from PIL import Image


def psnr(x, y):
   return -10.0 * torch.log10(torch.mean((x - y) ** 2))

def compute_ssim(img1, img2):
   return ssim(
      img1.detach().cpu().numpy(),
      img2.detach().cpu().numpy(),
      win_size=7,                  # set smaller if needed (e.g. 5 or 3)
      channel_axis=-1,            # replaces `multichannel=True`
      data_range=1.0
   )

# === Render Utility for Evaluation and Preview ===
def render_image(model, rays_o, rays_d, device, N_samples=64, chunk_size=2048):
   """Render full image with chunking."""
   rays_o = rays_o.to(device)
   rays_d = rays_d.to(device)

   # Sample points along rays
   pts, z_vals = Camera.sample_points_along_rays(rays_o, rays_d, near=2.0, far=6.0, N_samples=N_samples)
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


# === Main Training Function ===
def train():
   epochs = 2001
   data_dir = "./data/llff/testscene"
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   # Load LLFF training and validation datasets
   dataset = LLFFDataset(scene_dir=data_dir, split="train")
   # Load LLFF training and validation datasets
   val_dataset = LLFFDataset(scene_dir=data_dir, split="train")
   train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

   H, W, focal = dataset.H, dataset.W, dataset.focal
   # Initialize NeRF model and optimizer
   model = NeRF().to(device)
   optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   result_dir = f"./results/{timestamp}"
   os.makedirs(result_dir, exist_ok=True)
   os.makedirs(f"{result_dir}/log", exist_ok=True)
   os.makedirs(f"{result_dir}/weights", exist_ok=True)
   os.makedirs(f"{result_dir}/plots", exist_ok=True)

   log_path = os.path.join(result_dir, "log", "log.csv")
   with open(log_path, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(["epoch", "train_loss", "val_loss", "train_psnr", "val_psnr", "train_ssim", "val_ssim"])

   best_val_loss = float("inf")
   best_val_ssim = float(-1.0)
   best_val_psnr = float(0.0)
   # === Training Loop ===
   for epoch in range(1, epochs):
      # Set model to training mode and zero out metrics
      model.train()
      train_losses, train_psnrs, train_ssims = [], [], []

      for sample in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
         rgb_gt, pose, focal, H, W, bounds = [s.squeeze(0).to(device) for s in sample]
         
         
         cam = Camera(eye=pose[:, 3], target=None, focal=focal, H=H, W=W, c2w=pose)
         rays_o, rays_d = cam.get_rays()
         
         rgb_map = render_image(model, rays_o, rays_d, device)
         # print(f"Size of gt = {rgb_gt.shape}, size of model output map = {rgb_map.shape}")

         loss = F.mse_loss(rgb_map, rgb_gt)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

         train_losses.append(loss.item())
         train_psnrs.append(psnr(rgb_map, rgb_gt).item())
         train_ssims.append(compute_ssim(rgb_map, rgb_gt))

      model.eval()
      val_losses, val_psnrs, val_ssims = [], [], []
      for sample in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
         rgb_gt, pose, focal, H, W, bounds = [s.squeeze(0).to(device) for s in sample]
         
         cam = Camera(eye=pose[:, 3], target=None, focal=focal, H=H, W=W, c2w=pose)
         rays_o, rays_d = cam.get_rays()
         
         rgb_map = render_image(model, rays_o, rays_d, device)

         val_loss = F.mse_loss(rgb_map, rgb_gt)
         val_losses.append(val_loss.item())
         val_psnrs.append(psnr(rgb_map, rgb_gt).item())
         val_ssims.append(compute_ssim(rgb_map, rgb_gt))

      train_loss = np.mean(train_losses)
      val_loss = np.mean(val_losses)
      train_psnr = np.mean(train_psnrs)
      val_psnr = np.mean(val_psnrs)
      train_ssim = np.mean(train_ssims)
      val_ssim = np.mean(val_ssims)

      print(f"[Epoch {epoch}/{epochs}]: train_loss = {train_loss:.4f} | val_loss = {val_loss:.4f} | train_ssim = {train_ssim:.4f} | val_loss = {val_loss:.4f} | val_ssim = {val_ssim:.4f} | train_psnr = {train_psnr:.4f} | val_psnr = {val_psnr:.4f}")

      with open(log_path, 'a') as f:
         writer = csv.writer(f)
         writer.writerow([epoch, train_loss, val_loss, train_psnr, val_psnr, train_ssim, val_ssim])

      if train_psnr > best_val_psnr or train_ssim > best_val_ssim:
         print(f"Saving new best weights epoch {epoch}")
         torch.save(model.state_dict(), os.path.join(result_dir, "weights", "best_model.pt"))
         
      if val_ssim > best_val_ssim:
         best_val_psnr = val_ssim
         
      if val_psnr > best_val_psnr:
         best_val_psnr = val_psnr

      # Render preview every 5 epochs for visualization
      if epoch % 10 == 0:
         # Render and save preview image
         # sample = val_dataset[0]
         sample = dataset[0]
         rgb_gt, pose, focal, H, W, bounds = sample

         # Only call `.to(device)` on tensors
         rgb_gt = rgb_gt.to(device)
         pose = pose.to(device)
         
         cam = Camera(eye=pose[:, 3], target=None, focal=focal, H=H, W=W, c2w=pose)
         rays_o, rays_d = cam.get_rays()
         
         rgb_map = render_image(model, rays_o, rays_d, device)
         
         img = (rgb_map.clamp(0.0, 1.0).detach().cpu().numpy() * 255).astype(np.uint8)
         img = np.transpose(img, (1, 0, 2))
         Image.fromarray(img).save(os.path.join(result_dir, "plots", f"epoch_{epoch:03d}.png"))

      scheduler.step()
      
      torch.cuda.empty_cache()


   # === Run Post-training Analysis ===
   evaluate_real_scene(result_dir, data_dir, image_idx=0, N_samples=64, near=2.0, far=6.0)
   return result_dir

if __name__ == "__main__":
   train()