import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.positional_encoding import *

# From NeRF paper layout
# Input (x) --[PE]-->  Input Encoding --→ Fully Connected Layers --→ Feature (256d)
#                                                                  │
#                                           [PE(view_dir)]         ↓
#                                        → PosEnc View Dir    +--> Linear (128d)
#                                                            |
#                                                            ↓
#                                                         Output RGB


class NeRF(nn.Module):
   def __init__(self, 
               input_dim=3,          # Number of input dimensions (usually 3 for x, y, z)
               xyz_freqs=10,    # Number of frequency bands in positional encoding
               dir_freqs=4,
               hidden_dim=256,        # Number of hidden units per layer
               num_layers=8,         # Total number of layers in the MLP
               skips=[4]):            # Layers at which to add skip connections
      super(NeRF, self).__init__()

      self.xyz_freqs = xyz_freqs
      self.dir_freqs = dir_freqs
      self.input_dim = input_dim
      self.hidden_dim = hidden_dim
      self.num_layers = num_layers
      self.skip_connections = skips

      # Positional encoding increases input dimension
      # Assume x is (x, y, z) → input_dim = 3
      self.pe_xyz = PositionalEncoding(num_freqs=xyz_freqs, input_dims=3)
      self.pe_dir = PositionalEncoding(num_freqs=dir_freqs, input_dims=3)
      
      self.input_xyz_dim = self.pe_xyz.output_dims
      self.input_dir_dim = self.pe_dir.output_dims
      
      # Trunk: MLP for position encoding
      self.layers_xyz = nn.ModuleList()
      for i in range(self.num_layers):
         if i == 0:
               in_ch = self.input_xyz_dim
         elif i in skips:
               in_ch = self.input_xyz_dim + self.hidden_dim
         else:
               in_ch = self.hidden_dim
         self.layers_xyz.append(nn.Linear(in_ch, self.hidden_dim))
      
      # Output for sigma (density)
      self.sigma_out = nn.Linear(self.hidden_dim, 1)
      
      # Bottleneck layer before RGB head
      self.feature_layer = nn.Linear(self.hidden_dim, self.hidden_dim)

      # RGB branch takes feature + view_dir encoding
      self.rgb_layers = nn.Sequential(
         nn.Linear(self.hidden_dim + self.input_dir_dim, self.hidden_dim // 2),
         # nn.Linear(self.hidden_dim, self.hidden_dim // 2),
         nn.ReLU(),
         nn.Linear(self.hidden_dim // 2, 3),
         nn.Sigmoid()
      )

   def forward(self, x, view_dirs):
      """
      Forward pass through the NeRF model.

      Args:
         x: Input coordinates of shape (N, 3)
         view_dirs: Input view direction encoding

      Returns:
         rgb: Tensor of shape (N, 3), values in [0, 1]
         sigma: Tensor of shape (N,), non-negative density
      """
      x_encoded = self.pe_xyz(x)
      
      h = x_encoded
      
      # Base network
      for i, layer in enumerate(self.layers_xyz):
         if i in self.skip_connections:
               h = torch.cat([x_encoded, h], dim=-1)
         h = F.relu(layer(h))

      # Sigma (view-independent)
      sigma = F.softplus(self.sigma_out(h))  # ensures sigma ≥ 0

      # View-dependent color prediction
      if view_dirs is not None:
         d_encoded = self.pe_dir(view_dirs)
         feat = self.feature_layer(h)
         h_rgb = torch.cat([feat, d_encoded], dim=-1)
         rgb = self.rgb_layers(h_rgb)
      else:
         rgb = torch.zeros(x.shape[0], 3, device=x.device)
      
      # rgb = torch.sigmoid(self.rgb_layers(h_rgb))  # Clamp RGB values to [0, 1]
      # rgb = self.rgb_layers(h)
      
      if torch.isnan(sigma).any():
         print("⚠️ NaN detected in sigma")
      if torch.isnan(rgb).any():
         print("⚠️ NaN detected in rgb")

      return rgb, sigma
