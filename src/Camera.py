import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Helper function for the camera class
def look_at_camera(eye, target, up=None):
   # Make sure everything is a float
   eye = eye.float()
   target = target.float()

   forward = target - eye
   forward = forward / torch.norm(forward)

   if up is None:
      up = torch.tensor([0.0, 1.0, 0.0], dtype=eye.dtype, device=eye.device)
   
   up = up.float()

   if torch.abs(torch.dot(forward, up)) > 0.999:
      up = torch.tensor([0.0, 0.0, 1.0], dtype=eye.dtype, device=eye.device)

   right = torch.cross(up, forward, dim=0)
   right = right / torch.norm(right)

   true_up = torch.cross(forward, right, dim=0)

   rot = torch.stack([right, true_up, forward], dim=1)
   trans = eye.view(3, 1)

   return torch.cat([rot, trans], dim=1)  # [3, 4]


# Camera abstraction used for visualization and modeling 
class Camera:
   def __init__(self, eye, target, focal, H, W, c2w=None, pp=None, up=None):
      self.eye = eye
      self.target = target
      self.focal = focal # (fx, fy)
      self.H = H
      self.W = W
      self.principal_point = pp
      self.up = up
      # Check if c2w is provided by the user 
      if c2w is not None:
         self.c2w = c2w
      else:
         self.c2w = look_at_camera(self.eye, self.target, self.up)

   def get_rays(self):
      device = self.eye.device
      j, i = torch.meshgrid(
         torch.arange(self.H, dtype=torch.float32, device=device),
         torch.arange(self.W, dtype=torch.float32, device=device),
         indexing='ij'
      )
      
      if self.principal_point is not None:
         cx, cy = self.principal_point
      else:
         cx, cy = self.W*0.5, self.H*0.5

      dirs = torch.stack([
         (i - cx) / self.focal[0],
         -(j - cy) / self.focal[1],
         -torch.ones_like(i)
      ], dim=-1)
      
      rays_d = torch.sum(dirs[..., None, :] * self.c2w[:3, :3], dim=-1)
      rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
      rays_o = self.c2w[:3, 3].expand_as(rays_d)

      return rays_o, rays_d

   @staticmethod
   def axis_angle_to_matrix(axis, theta):
      """Rodrigues rotation formula (vectorized).

      Args:
          axis (..., 3): unit vectors
          theta (...,): rotation angle in radians

      Returns:
          R (..., 3, 3): rotation matrices
      """
      axis = axis/axis.norm(dim=-1, keepdim=True).clamp(min=1e-9)
      ax, ay, az = axis.unbind(-1)
      c = torch.cos(theta)
      s = torch.sin(theta)
      C = 1.0 - c
      
      R = torch.stack([
         c + ax*ax*C,    ax*ay*C - az*s, ax*az*C + ay*s,
         ay*ax*C + az*s,    c + ay*ay*C, ay*az*C - ax*s,
         az*ax*C - ay*s, az*ay*C + ax*s,    c + az*az*C
      ], dim=-1).reshape(axis.shape[:-1] + (3, 3))
      return R
   
   @staticmethod
   def look_at_to_matrix(eye, target, up=None):
      """Build a camera-to-world matrix for meye/target/up. Returns a 3x4 
      tensor: [R | t] in your existing convention.  All tensors must be on the 
      same device & dtype

      Args:
          eye (_type_): camera pose
          target (_type_): target pose
          up (_type_, optional): _description_. Defaults to None.

      Returns:
          _type_: _description_
      """
      if up is None:
         up = torch.tensor([0., 1., 0.], dtype=eye.dtype, device=eye.device)
         
      forward = (target - eye)
      forward = forward/forward.norm(dim=-1, keepdim=True).clamp(min=1e-9)
      
      right = torch.cross(up.expand_as(forward), forward, dim=-1)
      right = right/right.norm(dim=-1, keepdim=True).clamp(min=1e-9)
      
      true_up = torch.cross(forward, right, dim=-1)
      
      R = torch.stack([right, true_up, forward], dim=-1) # (.., 3, 3)
      t = eye.unsqueeze(-1)   # (..., 3, 1)
      return torch.cat([R, t], dim=-1)

   @staticmethod
   def sample_points_along_rays(rays_o, rays_d, near, far, N_samples, perturb=True):
      H, W = rays_o.shape[:2]
      z_vals = torch.linspace(near, far, N_samples, device=rays_o.device).view(1, 1, N_samples).expand(H, W, N_samples)
      if perturb:
         # Add noise to z_vals (stratified sampling)
         mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])  # midpoint between samples
         upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
         lower = torch.cat([z_vals[..., :1], mids], dim=-1)
         t_rand = torch.rand_like(z_vals)
         z_vals = lower + (upper - lower) * t_rand
         z_vals, _ = torch.sort(z_vals, dim=-1)
   
      pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
      return pts, z_vals

   @staticmethod
   def volume_render(rgb, sigma, z_vals, white_bkgd=False):
      device = rgb.device
      
      # Move everything to same device
      sigma = sigma.to(device)
      z_vals = z_vals.to(device)
      
      H, W, N_samples = sigma.shape

      dists = z_vals[..., 1:] - z_vals[..., :-1]
      dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e5)], dim=-1)

      alpha = 1.0 - torch.exp(-sigma * dists)
      
      # ones = torch.ones((H, W, 1), device=device)
      # eps = 1e-10
      # one = torch.tensor(1.0, device=device)
      
      trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 
                                       1.0 - alpha + 1e-10], dim=-1),
                            dim=-1)[..., :-1]
         
      # T = torch.cumprod(torch.cat([
      #    ones,
      #    one - alpha + eps
    
      # ], dim=-1), dim=-1)[..., :-1]
      
      weights = alpha * trans
      final_rgb = torch.sum(weights[..., None] * rgb, dim=-2) # (H, W, 3)
      
      if white_bkgd:
         final_rgb = final_rgb + (1.0 - weights.sum(dim=-1, keepdim=True))

      # print("Image min:", final_rgb.min().item(), "max:", final_rgb.max().item())
      # print("Max alpha:", alpha.max().item())
      # print("Max weight:", weights.max().item())
      # print("Final image max pixel:", final_rgb.max().item())

      return final_rgb, weights

   def visualize_corner_rays(self, rays_o, rays_d, obj_center, xlim=1, ylim=1, zlim=1):
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.scatter(obj_center[0], obj_center[1], obj_center[2], color='black', label='Origin', s=50)
      ax.quiver(*rays_o[0, 0], *rays_d[0, 0], color='r', length=1.0)
      ax.quiver(*rays_o[0, -1], *rays_d[0, -1], color='g', length=1.0)
      ax.quiver(*rays_o[-1, 0], *rays_d[-1, 0], color='b', length=1.0)
      ax.quiver(*rays_o[-1, -1], *rays_d[-1, -1], color='y', length=1.0)
      ax.set_title("Ray directions from camera corners")
      ax.set_xlim([-xlim, xlim])
      ax.set_ylim([-ylim, ylim])
      ax.set_zlim([-zlim, zlim])
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      plt.show()
