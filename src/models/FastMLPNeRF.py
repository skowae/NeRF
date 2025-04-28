import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.positional_encoding import PositionalEncoding

# Create the FastNeRF class
class FastMLPNeRF(nn.Module):
   def __init__(self, num_layers=8, hidden_dim=256, skips=[4], num_encoding_fn_xyz=10, num_encoding_fn_dir=4):
      super(FastMLPNeRF, self).__init__()
      
      
      input_dim_xyz = 3*(2*num_encoding_fn_xyz + 1)
      input_dim_dir = 3*(2*num_encoding_fn_dir + 1)
      
      # Positional encoders
      self.pe_xyz = PositionalEncoding(num_freqs=num_encoding_fn_xyz, input_dims=3)
      
      self.pe_dir = PositionalEncoding(num_freqs=num_encoding_fn_dir, input_dims=3)
      
      
      # MLP for (x, y, z)
      self.mlp_xyz = nn.ModuleList()
      for i in range(num_layers):
         if i == 0:
            in_dim = input_dim_xyz
         elif i in skips:
            in_dim = hidden_dim + input_dim_xyz
         else:
            in_dim = hidden_dim

         self.mlp_xyz.append(nn.Linear(in_dim, hidden_dim))
         
      # Output heads
      # Density
      self.fc_sigma = nn.Linear(hidden_dim, 1)
      # Feature
      self.fc_feature = nn.Linear(hidden_dim, hidden_dim)
      
      # MLP for RGB after viewdir injection
      self.mlp_rgb = nn.Sequential(
         nn.Linear(hidden_dim + input_dim_dir, hidden_dim//2),
         nn.ReLU(),
         nn.Linear(hidden_dim//2, 3),
         nn.Sigmoid()
      )
      
   def forward(self, x, view_dirs):
      """Forward call of the model

      Args:
          x (N, 3): points
          view_dirs (N, 3): view directions
      """
      # Positional encode
      x_encoded = self.pe_xyz(x)
      d_encoded = self.pe_dir(view_dirs)
      
      h = x_encoded
      for i, layer in enumerate(self.mlp_xyz):
         if i == 4:
            h = torch.cat([h, x_encoded], dim=-1)
         h = F.relu(layer(h))
      
      sigma = F.softplus(self.fc_sigma(h))
      feature = self.fc_feature(h)
      
      h_rgb = torch.cat([feature, d_encoded], dim=-1)
      rgb = self.mlp_rgb(h_rgb)
      
      return rgb, sigma

