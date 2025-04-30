import torch
import torch.nn as nn 
from utils.hash_grid_encoder import HashGridEncoder 
from utils.positional_encoding import PositionalEncoding

# Input 3D Position (x, y, z) ---> [ HashGrid Encoder ]
#                                        ↓
#                          Encoded Features (all levels)
#                                        ↓
#                            [ Small MLP for Density ]
#                                        ↓
#                ┌─────────────────────────────────────┐
#                ↓                                     ↓
#          Density σ                             Feature Vector
#                ↓                                     ↓
#               (--- If view_dir available ---)
#                                        ↓
#                       View Direction (dx, dy, dz)
#                                 ↓
#                  [ Positional Encoding of View Dir ]
#                                 ↓
#               Concatenated Feature + Encoded View Dir
#                                        ↓
#                            [ Small MLP for Color ]
#                                        ↓
#                            Output RGB (r, g, b)


# New NeRF NGP model
class FastNGPNeRF(nn.Module):
   def __init__(self,
                n_levels=16,
                n_features_per_level=2,
                log2_hashmap_size=19,
                base_resolution=16,
                finest_resolution=512, 
                hidden_dim=64):
      super().__init__()
      
      # HashGrid for 3D position
      self.hash_encoder = HashGridEncoder(
         n_levels=n_levels,
         n_features_per_level=n_features_per_level,
         log2_hashmap_size=log2_hashmap_size, 
         base_resolution=base_resolution, 
         finest_resolution=finest_resolution
      )
      
      # Positional encoding for view direction
      self.dir_encoder = PositionalEncoding(num_freqs=4, include_input=True)
      
      # Small MLP to predict density and features
      input_dim_density = n_levels*n_features_per_level
      self.fc_density = nn.Sequential(
         nn.Linear(input_dim_density, hidden_dim), 
         nn.ReLU(),
         nn.Linear(hidden_dim, 1 + 15)
      )
      for layer in self.fc_density:
         if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-4)
            torch.nn.init.constant_(layer.bias, 0)
      
      # Small MLP to predict RGB
      input_dim_color = 15 + self.dir_encoder.output_dims      
      self.fc_color = nn.Sequential(
         nn.Linear(input_dim_color, hidden_dim),
         nn.ReLU(), 
         nn.Linear(hidden_dim, 3)
      )
      for layer in self.fc_color:
         if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=1e-4)
            torch.nn.init.constant_(layer.bias, 0)
      
   def forward(self, x, view_dirs=None):
      """Forward call.  Inference

      Args:
          x (N, 3): coordinates in [0, 1]
          view_dirs (N, 3): Normalized viewing directions. Defaults to None.
      """
      
      x_encoded = self.hash_encoder(x)
      
      if view_dirs is not None:
         view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True)
         d_encoded = self.dir_encoder(view_dirs)
      else:
         with torch.no_grad():
            d_encoded = torch.zeros(x.shape[0], self.dir_encoder.output_dims, device=x.device)
      
      density_features = self.fc_density(x_encoded)
      sigma = torch.relu(density_features[..., 0])
      features = density_features[..., 1:]
      
      color_input = torch.cat([features, d_encoded], dim=-1)
      rgb = torch.sigmoid(self.fc_color(color_input))
      
      return rgb, sigma