import os
import torch
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
from datetime import datetime 
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt 
import csv
from skimage.metrics import structural_similarity as ssim
from PIL import Image

from Camera import Camera 
from data.nerf_capture_dataset import NeRFCaptureDataset 
from models.FastMLPNeRF import FastMLPNeRF
from utils.analyze_capture_results import *

def main():
   # Configuration 
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   scene_dir = './data/skow/fern'
   img_wh = (100, 75)
   batch_size = 1
   lr = 5e-4
   weight_decay = 1e-6
   gamma = 0.5
   epochs = 500
   N_samples = 64
   near = 0.1
   far = 4.0
   
   # Create the outpur dirs
   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   out_dir = f'results/SkowCapture/{timestamp}'
   os.makedirs(out_dir, exist_ok=True)
   os.makedirs(os.path.join(out_dir, 'log'), exist_ok=True)
   os.makedirs(os.path.join(out_dir, 'weights'), exist_ok=True)
   os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
   
   # Load the dataset
   dataset = NeRFCaptureDataset(scene_dir, img_wh=img_wh)
   dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
   
   # Initialize the model 
   model = FastMLPNeRF().to(device)
   optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=gamma)
   
   train(model, device, optimizer, scheduler, dataloader, epochs, N_samples, near, far, out_dir)
   
   evaluate_real_scene(out_dir, scene_dir, image_idx=0, N_samples=64, near=near, far=far)

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

def render_image(model, rays_o, rays_d, device, perturb=True, N_samples=64, chunk_size=2048):
   """Render full image with chunking."""
   
   rays_o = rays_o.to(device)
   rays_d = rays_d.to(device)

   # Sample points along rays
   pts, z_vals = Camera.sample_points_along_rays(rays_o, rays_d, near=0.1, far=4.0, N_samples=N_samples, perturb=perturb)
            
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

def train(model, device, optimizer, scheduler, dataloader, epochs, N_samples, near, far, results_dir):
   """Training loop

   Args:
       model (_type_): _description_
       optimizer (_type_): _description_
       scheduler (_type_): _description_
       epochs (_type_): _description_
       N_samples (_type_): _description_
       near (_type_): _description_
       far (_type_): _description_
   """
   
   save_period = 10
   
   log_path = os.path.join(results_dir, "log", "log.csv")
   with open(log_path, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(["epoch", "train_loss", "val_loss", "train_psnr", "val_psnr", "train_ssim", "val_ssim"])

   best_val_loss = float("inf")
   best_val_ssim = float(-1.0)
   best_val_psnr = float(0.0)
   
   # === Training Loop ===
   for epoch in range(1, epochs):
      # Set model to training mode and zero out metrics
      train_losses, train_psnrs, train_ssims = [], [], []
      first_val_batch = None
      model.train()
      for img, pose, focal, pp, img_wh in tqdm(dataloader, desc=f"Epoch {epoch} [Train]"):
         if first_val_batch is None:
            # Save a copy
            first_val_batch = (img.clone(), pose.clone(), focal.clone(), pp.clone(), img_wh.clone())

         img = img.to(device)  # (1, 3, H, W)
         pose = pose.squeeze(0).to(device) # (4, 4)
         fl_x, fl_y = focal.squeeze(0).tolist()
         cx, cy = pp.squeeze(0).tolist()
         w, h = img_wh.squeeze(0).tolist()
         
         # Create rays
         H, W = img.shape[-2], img.shape[-1]
         cam = Camera(eye=pose[:, 3], target=None, focal=(fl_x, fl_y), H=H, W=W, c2w=pose, pp=(cx, cy))
         rays_o, rays_d = cam.get_rays()
         
         # Render image
         rgb_map = render_image(model, rays_o, rays_d, device, perturb=True)

         # Compute the loss
         rgb_gt = img.squeeze(0).permute(1, 2, 0) # (H, W, 3)

         loss = F.mse_loss(rgb_map, rgb_gt)
         
         # Gradient descent 
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         
         train_losses.append(loss.item())
         train_psnrs.append(psnr(rgb_map, rgb_gt).item())
         train_ssims.append(compute_ssim(rgb_map, rgb_gt))
      
      val_losses, val_psnrs, val_ssims = [], [], []
      # first_val_batch = None
      # model.eval()
      # for img, pose, focal, pp, img_wh in tqdm(dataloader, desc=f"Epoch {epoch} [Val]"):
      #    # Check if this is the first batch
      #    if first_val_batch is None:
      #       # Save a copy
      #       first_val_batch = (img.clone(), pose.clone(), focal.clone(), pp.clone(), img_wh.clone())
      #    img = img.to(device)  # (1, 3, H, W)
      #    pose = pose.squeeze(0).to(device) # (4, 4)
      #    fl_x, fl_y = focal.squeeze(0).tolist()
      #    cx, cy = pp.squeeze(0).tolist()
      #    w, h = img_wh.squeeze(0).tolist()
         
      #    # Create rays
      #    H, W = img.shape[-2], img.shape[-1]
      #    cam = Camera(eye=pose[:, 3], target=None, focal=(fl_x, fl_y), H=H, W=W, c2w=pose, pp=(cx, cy))
      #    rays_o, rays_d = cam.get_rays()
         
      #    # Render image
      #    rgb_map = render_image(model, rays_o, rays_d, device, perturb=False)

      #    # Compute the loss
      #    rgb_gt = img.squeeze(0).permute(1, 2, 0) # (H, W, 3)
      #    loss = F.mse_loss(rgb_map, rgb_gt)
         
      #    val_losses.append(loss.item())
      #    val_psnrs.append(psnr(rgb_map, rgb_gt).item())
      #    val_ssims.append(compute_ssim(rgb_map, rgb_gt))
      
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
         
      # TODO actually compare to eval
      if train_psnr > best_val_psnr or train_ssim > best_val_ssim:
         print(f"Saving new best weights epoch {epoch}")
         torch.save(model.state_dict(), os.path.join(results_dir, "weights", "best_model.pt"))
         
      if train_ssim > best_val_ssim:
         best_val_psnr = train_ssim
         
      if train_psnr > best_val_psnr:
         best_val_psnr = train_psnr

      # Render preview every 5 epochs for visualization
      if epoch % save_period == 0 and first_val_batch is not None:
         # Render and save preview image
         img, pose, focal, pp, img_wh = first_val_batch
         
         img = img.to(device)  # (1, 3, H, W)
         pose = pose.squeeze(0).to(device) # (4, 4)
         fl_x, fl_y = focal.squeeze(0).tolist()
         cx, cy = pp.squeeze(0).tolist()
         w, h = img_wh.squeeze(0).tolist()
         
         # Create rays
         H, W = img.shape[-2], img.shape[-1]
         cam = Camera(eye=pose[:, 3], target=None, focal=(fl_x, fl_y), H=H, W=W, c2w=pose, pp=(cx, cy))
         rays_o, rays_d = cam.get_rays()
         
         model.eval()
         rgb_map = render_image(model, rays_o, rays_d, device, perturb=False)
         
         img = (rgb_map.clamp(0.0, 1.0).detach().cpu().numpy() * 255).astype(np.uint8)
         Image.fromarray(img).save(os.path.join(results_dir, "plots", f"epoch_{epoch:03d}.png"))

      scheduler.step()
      
      torch.cuda.empty_cache()

if __name__ == "__main__":
   main()