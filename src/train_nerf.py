import os
import gc
import csv
import time
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
import matplotlib.pyplot as plt
from models.NeRF import NeRF
from Camera import Camera
from utils.get_rays import *
from utils.analyze_results import *

def render_in_chunks_safe(model, pts_flat, view_dirs_flat, chunk_size=2048):
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
         rgb_chunk, sigma_chunk = model(pts_flat[i:end], view_dirs_flat[i:end])
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
         sub_rgb, sub_sigma = render_in_chunks_safe(model, pts_flat[i:end], view_dirs_flat[i:end], chunk_size // 2)
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
      rgb, sigma = evaluate_fake_radiance_field_complex(pts)
      # rgb, sigma = evaluate_fake_radiance_field_torus(pts)
      # rgb, sigma = evaluate_fake_radiance_field_sphere(pts)
      rgb_map, _ = Camera.volume_render(rgb, sigma, z_vals)
      images.append(rgb_map.detach())
      rays_o_all.append(rays_o)
      rays_d_all.append(rays_d)

   return images, rays_o_all, rays_d_all


def train():
   try: 
      # Config
      epochs = 21
      
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
      scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)  # Or your decay rate


      # Define camera and ray sampling parameters
      H, W, focal = 100, 100, 100
      near, far, N_samples = 2.0, 6.0, 64
      target = torch.tensor([0.0, 0.0, 0.0], device=device)
      eye = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32, device=device)

      cam = Camera(eye=eye, target=target, focal=focal, H=H, W=W)
      
      # Generate training data from synthetic torus
      images_gt, rays_o_all, rays_d_all = generate_training_data(device, H, W, focal, near, far, N_samples, target)
      num_views = len(images_gt)      

      gt_img = render_gt_image(cam, device=device, N_render=N_samples)
      Image.fromarray(gt_img).save(os.path.join(result_dir, "plots", "ground_truth.png")) 

      # Initialize training log
      with open(log_path, 'w', newline='') as csvfile:
         writer = csv.writer(csvfile)
         writer.writerow(["epoch", "loss", "ssim", "psnr"])

      # Init best loss and metrics
      best_loss = float('inf')      
      for epoch in range(epochs):
         model.train()
         total_loss = 0
         ssim_total = 0.0
         psnr_total = 0.0

         # Iterate over training examples (views)
         for rays_o, rays_d, rgb_gt in zip(rays_o_all, rays_d_all, images_gt):
            # Get the points
            pts, z_vals = Camera.sample_points_along_rays(rays_o, rays_d, near, far, N_samples)
            pts_flat = pts.view(-1, 3)
            
            # Add noise to z_vals (stratified sampling)
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])  # midpoint between samples
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand
            z_vals, _ = torch.sort(z_vals, dim=-1)
            
            # Get the view directions
            view_dirs = rays_d / rays_d.norm(dim=-1, keepdim=True)  # shape: [H, W, 3]
            view_dirs_flat = view_dirs.view(-1, 3)

            # Expand view directions to match N_samples along rays
            view_dirs_expanded = view_dirs[:, :, None, :].expand(H, W, N_samples, 3)
            view_dirs_flat = view_dirs_expanded.reshape(-1, 3)
            
            rgb_flat, sigma_flat = render_in_chunks_safe(model, pts_flat.to(device), view_dirs_flat.to(device))
            rgb_out = rgb_flat.view(H, W, N_samples, 3)
            sigma_out = sigma_flat.view(H, W, N_samples)
            
            rgb_map, weights = Camera.volume_render(rgb_out, sigma_out, z_vals)
            # print("RGB mean:", rgb_map.mean().item(), "std:", rgb_map.std().item())
            
            # logging
            # if torch.isnan(rgb_map).any():
            #    print("NaN in RGB map!")
            #    print("Sigma stats:", sigma_out.min().item(), sigma_out.max().item())
            #    print("RGB stats:", rgb_out.min().item(), rgb_out.max().item())
            #    continue  # Skip this view to prevent poisoning training

            # print("rgb_map stats:", rgb_map.min().item(), rgb_map.max().item())
            # print("rgb_gt stats:", rgb_gt.min().item(), rgb_gt.max().item())
            # print("Loss value (pre):", F.mse_loss(rgb_map, rgb_gt))

            # print("Has NaN in rgb_map?", torch.isnan(rgb_map).any().item())
            # print("Has NaN in rgb_gt?", torch.isnan(rgb_gt).any().item())


            loss = F.mse_loss(rgb_map, rgb_gt.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
            # Clamp and convert to numpy for logging
            rgb_map_np = rgb_map.clamp(0, 1).cpu().detach().numpy()
            rgb_gt_np = rgb_gt.clamp(0, 1).cpu().detach().numpy()

            # Compute SSIM and PSNR per-channel image (convert to H, W, 3)
            ssim = ssim_fn(rgb_gt_np, rgb_map_np, channel_axis=2, data_range=1.0)
            psnr = psnr_fn(rgb_gt_np, rgb_map_np, data_range=1.0)
            
            ssim_total += ssim
            psnr_total += psnr
            
            # total_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
            # print("Total grad:", total_grad)

         loss_avg = total_loss / num_views
         ssim_avg = ssim_total / num_views
         psnr_avg = psnr_total / num_views

         print(f"[Epoch {epoch}/{epochs}]: loss avg = {loss_avg:.4f} | ssim avg = {ssim_avg:.4f} | psnr_avg = {psnr_avg:.4f}")
         with open(log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, loss_avg, ssim_avg, psnr_avg])
         
         if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), f"{result_dir}/weights/best.pt")

         if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
               model.eval()
               pred_img = render_model_image(model, cam, device, N_samples)
               Image.fromarray(pred_img).save(os.path.join(result_dir, "plots", f"epoch_{epoch}.png"))
               model.train()
         
         scheduler.step()
                  
      
      best_path = os.path.join(result_dir, 'weights', 'best.pt')
      model.load_state_dict(torch.load(best_path, map_location=device))
      print(f"Loaded best weights from {best_path} for analysis.")
      
      model.eval()
      # Call post-training analysis
      evaluate_trained_model(
         result_dir,
         model,
         cam,
         device,
         N_samples=64
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