import torch
import matplotlib.pyplot as plt
from Camera import Camera
from models.NeRF import NeRF
from get_rays import evaluate_fake_radiance_field_torus

def main():
   # Image and camera parameters
   H, W = 100, 100
   focal = 60.0
   N_samples = 64
   near = 0.0
   far = 6.0

   eye = torch.tensor([1.0, 2.0, 0.0])
   target = torch.tensor([0.0, 0.0, 0.0])

   # Initialize the camera
   cam = Camera(eye=eye, target=target, focal=focal, H=H, W=W)

   # Initialize NeRF model
   model = NeRF()
   model.eval()  # No gradients needed for inference


   # Generate rays
   rays_o, rays_d = cam.get_rays()
   print("Ray origins shape:", rays_o.shape)
   print("Ray directions shape:", rays_d.shape)
   
   # Visualize corner rays in 3D
   cam.visualize_corner_rays(rays_o, rays_d)

   # Sample points
   pts, z_vals = cam.sample_points_along_rays(rays_o, rays_d, near=near, far=far, N_samples=N_samples)
   print("Sampled points shape:", pts.shape)
   print("Sampled z-values shape:", z_vals.shape)
   
   # Flatten all sampled points: [H, W, N_samples, 3] -> [H * W * N_samples, 3]
   pts_flat = pts.reshape(-1, 3)

   # Forward through NeRF
   with torch.no_grad():
      rgb_flat, sigma_flat = model(pts_flat)

   # Reshape back to [H, W, N_samples, 3] and [H, W, N_samples]
   nerf_rgb = rgb_flat.view(H, W, N_samples, 3)
   nerf_sigma = sigma_flat.view(H, W, N_samples)
   
   print("Max sigma (NeRF):", nerf_sigma.max().item())
   
   final_nerf_img, _ = cam.volume_render(nerf_rgb, nerf_sigma, z_vals)


   # Evaluate torus radiance field
   rgb, sigma = evaluate_fake_radiance_field_torus(pts)
   print("RGB shape:", rgb.shape)
   print("Sigma shape:", sigma.shape)

   # Visualize middle density row
   plt.imshow(sigma[H//2].cpu(), cmap='gray')
   plt.title("Density Slice (middle row)")
   plt.show()

   # Render
   final_img, _ = cam.volume_render(rgb, sigma, z_vals)
   print("Final image shape:", final_img.shape)

   # Visualize density and rendered image
   slice_idx = 32
   plt.imshow(sigma[:, :, slice_idx].cpu().numpy(), cmap='gray')
   plt.title("Density Field Slice (Torus)")
   plt.colorbar()
   plt.show()
   
   plt.imshow(final_nerf_img.cpu().numpy())
   plt.title("Rendered NeRF Torus")
   plt.axis("off")
   plt.show()

   plt.imshow(final_img.cpu().numpy())
   plt.title("Rendered Fake Torus")
   plt.axis("off")
   plt.show()

if __name__ == "__main__":
    main()