import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.positional_encoding import PositionalEncoding

# Create the FastNeRF class
class FastNeRF(nn.Module):
   def __init__(self, num_encoding_fn_xyz=10, num_encoding_fn_dir=4, 
                include_input_xyz=True, include_input_dir=True):
      super(FastNeRF, self).__init__()
      
      self.include_input_xyz = include_input_xyz
      self.include_input_dir = include_input_dir
      
      input_dim_xyz = 3*(2*num_encoding_fn_xyz + int(include_input_xyz))
      input_dim_dir = 3*(2*num_encoding_fn_dir + int(include_input_dir))
      
      # Positional encoders
      self.pe_xyz = PositionalEncoding(num_freqs=num_encoding_fn_xyz, 
                                       input_dims=3,
                                       include_input=include_input_xyz)
      
      self.pe_dir = PositionalEncoding(num_freqs=num_encoding_fn_dir,
                                       input_dims=3,
                                       include_input=include_input_dir)
      
      hidden_dim = 128
      
      # backbone
      self.fc1 = nn.Linear(input_dim_xyz, hidden_dim)
      self.fc2 = nn.Linear(hidden_dim, hidden_dim)
      self.fc3 = nn.Linear(hidden_dim, hidden_dim)
      
      # Density output
      self.fc_density = nn.Linear(hidden_dim, 1)
      
      # Color branch
      self.fc_feature = nn.Linear(hidden_dim, hidden_dim)
      self.fc_color = nn.Linear(hidden_dim + input_dim_dir, 3)
      
   def forward(self, x, view_dirs=None):
      """Forward call of the model

      Args:
          x (N, 3): points
          view_dirs (N, 3): view directions
      """
      x = self.pe_xyz(x)
      
      h = F.relu(self.fc1(x))
      h = F.relu(self.fc2(h))
      h = F.relu(self.fc3(h))
      
      # Density output
      sigma = F.relu(self.fc_density(h))
      
      # Process the view dirs
      if view_dirs is not None:
         view_dirs_encoded = self.pe_dir(view_dirs)
         h_feat = F.relu(self.fc_feature(h))
         h_combined = torch.cat([h_feat, view_dirs_encoded], dim=-1)
         # Color input
         rgb = torch.sigmoid(self.fc_color(h_combined))
      else:
         rgb = torch.sigmoid(self.fc_feature(h))
      
      return rgb, sigma.squeeze(-1)
      