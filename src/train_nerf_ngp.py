import os
import torch
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
from torch.amp import autocast, GradScaler
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from datetime import datetime 
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt 
import csv
from PIL import Image

from Camera import Camera 
from data.nerf_capture_dataset import NeRFCaptureDataset 
from models.FastNGPNeRF import FastNGPNeRF
from utils.analyze_capture_results import *

def main():
   # Configuration 
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"DEVICE {device}")
   scene_dir = './data/skow/robot'
   img_wh = (100, 75)
   batch_size = 1
   lr = 1e-2
   eps = 1e-15
   weight_decay = 1e-6
   epochs = 500
   gamma = 0.1**(1/epochs)
   N_samples = 64
   near = 0.05
   far = 1.20
   perturb = True
   jitter = False
   run_eval = False
   
   # Create the outpur dirs
   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   out_dir = f'results/SkowCapture/{timestamp}'
   os.makedirs(out_dir, exist_ok=True)
   os.makedirs(os.path.join(out_dir, 'log'), exist_ok=True)
   os.makedirs(os.path.join(out_dir, 'weights'), exist_ok=True)
   os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
   
   # Load the dataset
   dataset = NeRFCaptureDataset(scene_dir, img_wh=img_wh, subset=None)
   
   # Initialize the model 
   model = FastNGPNeRF().to(device)
   # model.apply(init_weights) # Commented out to prevent overwriting biases
   optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps)
   scaler = GradScaler()
   scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
   
   train(model, device, optimizer, scaler, scheduler, dataset, batch_size, epochs, N_samples, near, far, out_dir, perturb, jitter, run_eval)
   
   evaluate_real_scene(out_dir, scene_dir, image_idx=0, N_samples=64, near=near, far=far)

def jitter_pose(c2w, std=0.002):
   # c2w : (4,4) pose in unit-sphere frame
   jitter = torch.randn(3, device=c2w.device)*std
   c2w_j = c2w.clone()
   c2w_j[:3, 3] += jitter
   return c2w_j

def psnr(x, y):
   return -10.0 * torch.log10(torch.mean((x - y) ** 2))

def render_image(model, rays_o, rays_d, device, perturb=True, near=None, far=None, N_samples=64, chunk_size=16384):
   """Render full image with chunking."""
   
   assert near is not None and far is not None, "Must Pass near and far explicitly"
   
   rays_o = rays_o.to(device)
   rays_d = rays_d.to(device)

   # Sample points along rays
   pts, z_vals = Camera.sample_points_along_rays(rays_o, rays_d, near=near, far=far, N_samples=N_samples, perturb=perturb)
            
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
   
   # print("RGB output stats:", rgb_map.min().item(), rgb_map.max().item())
   # print("Sigma output stats:", sigma.min().item(), sigma.max().item())

   return rgb_map

def init_weights(m):
      if isinstance(m, torch.nn.Linear):
         torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
         if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def train(model, device, optimizer, scaler, scheduler, dataset, batch_size, epochs, N_samples, near, far, results_dir, perturb, jitter, run_eval):
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
   
   try: 
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
      ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
      
      save_period = 10
      ssim_period = 10
      
      log_path = os.path.join(results_dir, "log", "log.csv")
      with open(log_path, 'w') as f:
         writer = csv.writer(f)
         writer.writerow(["epoch", "train_loss", "val_loss", "train_psnr", "val_psnr", "train_ssim", "val_ssim"])

      best_val_loss = float("inf")
      best_val_ssim = float(-1.0)
      best_val_psnr = float(0.0)
      
      # === Training Loop ===
      for epoch in range(1, epochs):
         step = 0
         ssim_metric.reset()
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
            
            # Jitter the pose
            if jitter:
               pose = jitter_pose(pose)
               
            # Create rays
            H, W = img.shape[-2], img.shape[-1]
            cam = Camera(eye=pose[:, 3], target=None, focal=(fl_x, fl_y), H=H, W=W, c2w=pose, pp=(cx, cy))
            rays_o, rays_d = cam.get_rays()
            
            with autocast(device_type='cuda', dtype=torch.float16):
               # Render image
               rgb_map = render_image(model, rays_o, rays_d, device, perturb=perturb, near=near, far=far)

               # Compute the loss
               rgb_gt = img.squeeze(0).permute(1, 2, 0) # (H, W, 3)

               loss = F.mse_loss(rgb_map, rgb_gt)
            
            # Gradient descent 
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            
            train_losses.append(loss.item())
            train_psnrs.append(psnr(rgb_map, rgb_gt).item())
            if step%ssim_period == 0:
               # Torchmetrics expects (N, 3, H, W)
               pred = rgb_map.permute(2, 0, 1).unsqueeze(0) # -> (1, 3, H, W)
               gt = rgb_gt.permute(2, 0, 1).unsqueeze(0)
               ssim_val = ssim_metric(pred, gt)
               # Detatch for logging
               train_ssims.append(ssim_val.item())
            
            step += 1
         
         train_loss = np.mean(train_losses)
         train_ssim = np.mean(train_ssims)
         train_psnr = np.mean(train_psnrs)
         
         # Eval loop
         val_losses, val_psnrs, val_ssims = [], [], []
         if run_eval:
            val_step = 0
            first_val_batch = None
            model.eval()
            for img, pose, focal, pp, img_wh in tqdm(dataloader, desc=f"Epoch {epoch} [Val]"):
               # Check if this is the first batch
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
               rgb_map = render_image(model, rays_o, rays_d, device, perturb=False, near=near, far=far)

               # Compute the loss
               rgb_gt = img.squeeze(0).permute(1, 2, 0) # (H, W, 3)
               loss = F.mse_loss(rgb_map, rgb_gt)
               
               val_losses.append(loss.item())
               val_psnrs.append(psnr(rgb_map, rgb_gt).item())
               if val_step%ssim_period:
                  # Torchmetrics expects (N, 3, H, W)
                  ssim_val = ssim_metric(rgb_map.permute(0, 3, 1, 2),
                                         rgb_gt.permute(0, 3, 1, 2))
                  # Detatch for logging
                  val_ssims.append(ssim_val.item())
            
               val_step += 1

            val_loss = np.mean(val_losses)
            val_psnr = np.mean(val_psnrs)
            val_ssim = np.mean(val_ssims)
         else:
            val_loss = train_loss
            val_psnr = train_psnr
            val_ssim = train_ssim
         
         print(f"[Epoch {epoch}/{epochs}]: train_loss = {train_loss:.4f} | val_loss = {val_loss:.4f} | train_ssim = {train_ssim:.4f} | val_ssim = {val_ssim:.4f} | train_psnr = {train_psnr:.4f} | val_psnr = {val_psnr:.4f}")

         with open(log_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, train_psnr, val_psnr, train_ssim, val_ssim])
            
         # TODO actually compare to eval
         if val_psnr > best_val_psnr or val_ssim > best_val_ssim:
            print(f"Saving weights")
            torch.save(model.state_dict(), os.path.join(results_dir, "weights", "best_model.pt"))
            
         if val_ssim > best_val_ssim:
            print(f"New best SSIM = {val_ssim}")
            best_val_ssim = val_ssim
            
         if val_psnr > best_val_psnr:
            print(f"New best PSNR {val_psnr}")
            best_val_psnr = val_psnr

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
            cam = Camera(eye=pose[:, 3], target=None, focal=(fl_x, fl_y), H=H, 
                         W=W, c2w=pose, pp=(cx, cy))
            rays_o, rays_d = cam.get_rays()
            
            model.eval()
            rgb_map = render_image(model, rays_o, rays_d, device, perturb=False, 
                                   near=near, far=far)
            
            pred_img_uint8 = (rgb_map.clamp(0.0, 1.0).detach().cpu().numpy() * 255).astype(np.uint8)
            image_gt_np = img.squeeze(0).cpu().numpy()
            gt_img_uint8 = (image_gt_np.transpose(1, 2, 0) * 255).astype(np.uint8)
            side = np.concatenate([gt_img_uint8, pred_img_uint8], axis=1)
            Image.fromarray(side).save(os.path.join(results_dir, "plots", f"epoch_{epoch:03d}.png"))

         scheduler.step()
         
         torch.cuda.empty_cache()
   
   except KeyboardInterrupt:
      print("Training interrupted by user.")
      
   finally:
      print("Cleaning up GPU memory...")
      torch.cuda.empty_cache()
      gc.collect()
      # Optionally delete large objects
      del model, optimizer, rays_o, rays_d, img, rgb_map, pose

if __name__ == "__main__":
   main()
   