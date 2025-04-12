import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_stable_up(eye, target):
   forward = target - eye
   forward = forward / torch.norm(forward)

   up_guess = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

   # If forward is close to vertical, change guess
   if torch.allclose(forward, up_guess, atol=1e-2) or torch.allclose(forward, -up_guess, atol=1e-2):
      up_guess = torch.tensor([0.0, 0.0, 1.0])

   # Project out component of up_guess along forward
   up_proj = up_guess - torch.dot(up_guess, forward) * forward
   up = up_proj / torch.norm(up_proj)
   return up

def evaluate_fake_radiance_field_sphere(pts):
   """
   Inputs:
   - pts: [H, W, N_samples, 3] sampled 3D points

   Returns:
   - rgb: [H, W, N_samples, 3] color at each point
   - sigma: [H, W, N_samples] density at each point
   """
   # Distance from the origin
   dist = torch.norm(pts, dim=-1)  # [H, W, N_samples]

   # Density: solid inside radius 1.0
   sigma = torch.where(dist < 1.0, torch.ones_like(dist) * 50.0, torch.zeros_like(dist))

   # Color: constant red
   rgb = torch.zeros_like(pts)
   rgb[..., 0] = 1.0  # red channel

   return rgb, sigma

def evaluate_fake_radiance_field_torus(pts):
   """
   Inputs:
   - pts: [H, W, N_samples, 3] sampled 3D points

   Returns:
   - rgb: [H, W, N_samples, 3]
   - sigma: [H, W, N_samples]
   """

   x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]

   # --- Torus centered at origin in XZ plane ---
   R = 1.2  # Major radius (from center to tube)
   r = 0.3  # Minor radius (tube thickness)

   dist_to_ring = torch.sqrt(x**2 + z**2)
   torus_eq = (dist_to_ring - R)**2 + y**2  # Equation for torus

   sigma_torus = (torus_eq < r**2).float() * 5.0

   rgb_torus = torch.zeros_like(pts)
   rgb_torus[..., 0] = 1.0  # Red torus
   
   return rgb_torus, sigma_torus


def get_rays(H, W, focal, c2w):
   """
   Generate ray origins and directions for each pixel in an image.

   Inputs:
   - H, W: image height and width
   - focal: focal length of pinhole camera
   - c2w: [3, 4] camera-to-world transformation matrix

   Returns:
   - rays_o: [H, W, 3] ray origins
   - rays_d: [H, W, 3] ray directions
   """
   i, j = torch.meshgrid(
      torch.arange(W, dtype=torch.float32),
      torch.arange(H, dtype=torch.float32),
      indexing='ij'
   )

   dirs = torch.stack([
      (i - W * 0.5) / focal,
      -(j - H * 0.5) / focal,
      torch.ones_like(i)  # +Z is forward
   ], dim=-1)

   # Apply camera rotation
   rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
   rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

   # All rays originate from the same camera position
   rays_o = c2w[:3, 3].expand_as(rays_d)

   return rays_o, rays_d



def look_at_camera(eye, target, up):
   """
   Computes a right-handed camera-to-world matrix [3x4].
   - `eye`: [3] Camera position
   - `target`: [3] What the camera looks at
   - `up`: [3] Up direction (not necessarily orthogonal to forward)
   """
   forward = target - eye
   forward = forward / torch.norm(forward)

   right = torch.cross(forward, up)
   right = right / torch.norm(right)

   true_up = torch.cross(right, forward)

   # Rotation matrix: camera's x, y, z (right, up, forward)
   rot = torch.stack([right, true_up, forward], dim=1)  # [3, 3]
   trans = eye.view(3, 1)                               # [3, 1]

   return torch.cat([rot, trans], dim=1)                # [3, 4]





def sample_points_along_rays(rays_o, rays_d, near, far, N_samples):
   """
   Inputs:
   - rays_o: [H, W, 3] ray origins
   - rays_d: [H, W, 3] ray directions
   - near, far: bounds for sampling along the ray
   - N_samples: number of points to sample along each ray

   Returns:
   - pts: [H, W, N_samples, 3] sampled 3D points
   - z_vals: [H, W, N_samples] depth values
   """
   H, W = rays_o.shape[:2]

   # [N_samples] depths between near and far
   z_vals = torch.linspace(near, far, N_samples).view(1, 1, N_samples).expand(H, W, N_samples)

   # pts = ray_origin + ray_direction * depth
   pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [H, W, N_samples, 3]
   
   return pts, z_vals

def volume_render(rgb, sigma, z_vals):
   """
   Alpha Compositing
   Inputs:
   - rgb: [H, W, N_samples, 3]
   - sigma: [H, W, N_samples]
   - z_vals: [H, W, N_samples]

   Returns:
   - final_rgb: [H, W, 3] the rendered image
   """
   device = rgb.device
   H, W, N_samples = sigma.shape

   # Distance between adjacent samples
   dists = z_vals[..., 1:] - z_vals[..., :-1]  # [H, W, N_samples-1]
   dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)  # pad last

   # Compute alpha = 1 - exp(-sigma * delta)
   alpha = 1.0 - torch.exp(-sigma * dists)  # [H, W, N_samples]

   # Compute accumulated transmittance (cumprod of (1 - alpha))
   T = torch.cumprod(torch.cat([
      torch.ones((H, W, 1), device=device),
      1.0 - alpha + 1e-10
   ], dim=-1), dim=-1)[..., :-1]  # [H, W, N_samples]

   # Weights for each sample
   weights = alpha * T  # [H, W, N_samples]

   # Final rendered color
   final_rgb = torch.sum(weights[..., None] * rgb, dim=-2)  # [H, W, 3]
   print("Image min:", final_rgb.min().item(), "max:", final_rgb.max().item())
   
   print("Max alpha:", alpha.max().item())
   print("Max weight:", weights.max().item())
   print("Final image max pixel:", final_rgb.max().item())

   return final_rgb


def main():
   ## Test ray generation

   H, W = 100, 100
   focal = 60.0

   # Camera-to-world matrix: identity rotation + translation to (0, 0, 4)
   eye = torch.tensor([1.0, 2.0, 0.0])      # above and back
   target = torch.tensor([0.0, 0.0, 0.0])   # looking at center
   # Instead of a fixed up, compute it:
   up = compute_stable_up(eye, target)

   # up = torch.tensor([0.0, 1.0, 0.0])


   c2w = look_at_camera(eye, target, up)

   print(c2w)

   rays_o, rays_d = get_rays(H, W, focal, c2w)
   print("Ray origins shape:", rays_o.shape)
   print("Ray directions shape:", rays_d.shape)

   pts, z_vals = sample_points_along_rays(rays_o, rays_d, near=0.0, far=6.0, N_samples=128)
   print("Ray origin:", rays_o[0, 0])
   print("Ray direction:", rays_d[0, 0])
   print("Sampled points [0,0]:", pts[0, 0])

   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')

   # Plot camera origin
   ax.scatter(*rays_o[0, 0].cpu().numpy(), color='red', label='Camera origin')

   # Plot 4 corner rays (0,0), (0,W-1), (H-1,0), (H-1,W-1)
   corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]

   for (u, v) in corners:
      ray_pts = pts[u, v].cpu().numpy()  # [N_samples, 3]
      ax.plot(ray_pts[:, 0], ray_pts[:, 1], ray_pts[:, 2], label=f'Ray ({u}, {v})')

   # Torus center marker
   ax.scatter(0, 0, 0, color='black', label='Torus center', s=50)

   # Set limits
   ax.set_xlim([-2, 2])
   ax.set_ylim([-2, 2])
   ax.set_zlim([-2, 2])
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   ax.set_title('Corner Rays from Camera')
   ax.legend()
   plt.tight_layout()
   plt.show()



   ## Test sample points along rays

   near = 0.0
   far = 10.0
   N_samples = 256

   pts, z_vals = sample_points_along_rays(rays_o, rays_d, near, far, N_samples)
   print("Sampled points shape:", pts.shape)       # [100, 100, 64, 3]
   print("Sampled z-values shape:", z_vals.shape)  # [100, 100, 64]
   # print("Example ray samples [0,0,:]:", pts[0, 0])


   ## Test evaluate fake radiance field

   rgb, sigma = evaluate_fake_radiance_field_torus(pts)
   print("RGB shape:", rgb.shape)     # [100, 100, 64, 3]
   print("Sigma shape:", sigma.shape) # [100, 100, 64]

   ## Test volume rendering

   plt.imshow(sigma[H//2].cpu(), cmap='gray')
   plt.title("Density Slice (middle row)")
   plt.show()

   final_img = volume_render(rgb, sigma, z_vals)
   print("Final image shape:", final_img.shape)  # [100, 100, 3]

   slice_idx = 32  # halfway along the ray
   plt.imshow(sigma[:, :, slice_idx].cpu().numpy(), cmap='gray')
   plt.title("Density Field Slice (Torus)")
   plt.colorbar()
   plt.show()

   plt.imshow(final_img.cpu().numpy())
   plt.title("Rendered Fake Torus")
   plt.axis("off")
   plt.show()

if __name__ == "__main__":
    main()