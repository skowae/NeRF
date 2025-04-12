import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.positional_encoding import *

def positional_encoding(x, num_frequencies=10):
   """
   Applies positional encoding to input coordinates.

   Args:
      x: Tensor of shape (..., D), where D is typically 3 for 3D coords
      num_frequencies: Number of frequency bands used for encoding

   Returns:
      Encoded tensor of shape (..., D * (2 * num_frequencies + 1))
   """
   frequencies = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32, device=x.device)
   encoded = [x]  # Include the original input
   for freq in frequencies:
      encoded.append(torch.sin(freq * x))
      encoded.append(torch.cos(freq * x))
   return torch.cat(encoded, dim=-1)


class NeRF(nn.Module):
   def __init__(self, 
               input_dim=3,          # Number of input dimensions (usually 3 for x, y, z)
               num_frequencies=10,    # Number of frequency bands in positional encoding
               hidden_dim=256,        # Number of hidden units per layer
               num_layers=8,         # Total number of layers in the MLP
               skips=(4,)):            # Layers at which to add skip connections
      super().__init__()

      self.num_frequencies = num_frequencies
      self.input_dim = input_dim
      self.skip_connections = skips

      # Positional encoding increases input dimension
      pe_dim = input_dim * (2 * num_frequencies + 1)
      
      self.encoder = PositionalEncoding(num_frequencies, include_input=True, log_scale=True)

      self.layers = nn.ModuleList()

      # Construct MLP layers
      for i in range(num_layers):
         in_dim = pe_dim if i == 0 else hidden_dim
         if i in skips:
               in_dim += pe_dim  # Add input again at skip layers
         self.layers.append(nn.Linear(in_dim, hidden_dim))

      # Output layers
      self.sigma_out = nn.Linear(hidden_dim, 1)   # Outputs density (sigma)
      self.rgb_out = nn.Linear(hidden_dim, 3)     # Outputs color (R, G, B)

   def forward(self, x):
      """
      Forward pass through the NeRF model.

      Args:
         x: Input coordinates of shape (N, 3)

      Returns:
         rgb: Tensor of shape (N, 3), values in [0, 1]
         sigma: Tensor of shape (N,), non-negative density
      """
      # x_pe = positional_encoding(x, self.num_frequencies)  # Apply positional encoding
      x_pe = self.encoder(x)
      h = x_pe
      
      # Pass through MLP with optional skip connections
      for i, layer in enumerate(self.layers):
         if i in self.skip_connections:
               h = torch.cat([h, x_pe], dim=-1)
         h = F.relu(layer(h))

      # Predict sigma (density) and rgb (color)
      sigma = F.relu(self.sigma_out(h))           # Ensure sigma is non-negative
      rgb = torch.sigmoid(self.rgb_out(h))        # Keep RGB in [0, 1] range

      return rgb, sigma.squeeze(-1)
