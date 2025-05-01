import torch
import torch.nn as nn
import numpy as np

# HashGridEncoder abstraction used by the instant NGP mode
class HashGridEncoder(nn.Module):
   def __init__(self, n_levels=16, n_features_per_level=2, log2_hashmap_size=19, 
                base_resolution=16, finest_resolution=512):
      """_summary_

      Args:
          n_levels (int, optional): Number of different resolution grids. Like octaves. Defaults to 16.
          n_features_per_level (int, optional): How many features each vertex in the grid stores. Defaults to 2.
          log2_hashmap_size (int, optional): Size of hashtable. Defaults to 19.
          base_resolution (int, optional): Smalles grid res. Defaults to 16.
          finest_resolution (int, optional): Largest grid res. Defaults to 512.
      """
      super().__init__()
      
      self.n_levels = n_levels
      self.n_features_per_level = n_features_per_level
      
      # Exponentially increasing resolutions per level
      b = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / (n_levels - 1))
      self.resolutions = [int(base_resolution*(b**i)) for i in range(n_levels)]
      
      self.hash_size = 2**log2_hashmap_size
      
      # Create the small embeddings
      self.embeddings = nn.ModuleList()
      for i in range(n_levels):
         table_size = min(self.hash_size, self.resolutions[i]**3)
         emb = nn.Embedding(table_size, n_features_per_level)
         nn.init.uniform_(emb.weight, a=1e-4, b=1e-3)
         self.embeddings.append(emb)
      
      self.register_buffer('primes', torch.tensor([1, 2654435761, 805459861], dtype=torch.long))
   
   def hash_fn(self, coords, size):
      return ((coords*self.primes).sum(dim=-1) & 0xFFFFFFFF)%size
   
   def forward(self, x):
      """Forward function

      Args:
          x (N, 3): Tensor of 3D points in [0, 1] coordinates
      Returns: (N, n_levels*n_features_per_level)
      """
      
      x = x.unsqueeze(1) # (N, 1, 3)
      
      feats = []
      
      # Loop over the levels 
      for emb, res in zip(self.embeddings, self.resolutions):
         scale = res
         pos = x*scale # (N, 1, 3)
      
         # Compute the lower and upper grid indeces
         pos_0 = pos.floor().long() # (N, 1, 3) - lower corner
         pos_1 = pos_0 + 1 # (N, 1, 3) - upper corner
         
         # Interpolate features
         w = pos - pos_0.float() # (N, 1, 3)
         
         # Hash the 8 corners
         # Make sure the poses are in bounds
         pos_0 = pos_0%res
         pos_1 = pos_1%res
         
         # Determine the table size
         table_size = min(self.hash_size, res**3)
         
         # Unpack the points efficiently
         c_000 = self.hash_fn(torch.cat([pos_0[..., 0:1], pos_0[..., 1:2], pos_0[..., 2:3]], dim=-1), table_size)
         c_001 = self.hash_fn(torch.cat([pos_0[..., 0:1], pos_0[..., 1:2], pos_1[..., 2:3]], dim=-1), table_size)
         c_010 = self.hash_fn(torch.cat([pos_0[..., 0:1], pos_1[..., 1:2], pos_0[..., 2:3]], dim=-1), table_size)
         c_011 = self.hash_fn(torch.cat([pos_0[..., 0:1], pos_1[..., 1:2], pos_1[..., 2:3]], dim=-1), table_size)
         c_100 = self.hash_fn(torch.cat([pos_1[..., 0:1], pos_0[..., 1:2], pos_0[..., 2:3]], dim=-1), table_size)
         c_101 = self.hash_fn(torch.cat([pos_1[..., 0:1], pos_0[..., 1:2], pos_1[..., 2:3]], dim=-1), table_size)
         c_110 = self.hash_fn(torch.cat([pos_1[..., 0:1], pos_1[..., 1:2], pos_0[..., 2:3]], dim=-1), table_size)
         c_111 = self.hash_fn(torch.cat([pos_1[..., 0:1], pos_1[..., 1:2], pos_1[..., 2:3]], dim=-1), table_size)
         
         # Now retrieve the embeddings
         f_000 = emb(c_000)
         f_001 = emb(c_001)
         f_010 = emb(c_010)
         f_011 = emb(c_011)
         f_100 = emb(c_100)
         f_101 = emb(c_101)
         f_110 = emb(c_110)
         f_111 = emb(c_111)
         
         # Trilinear interpolate
         fx00 = f_000*(1 - w[..., 0:1]) + f_100*w[..., 0:1]
         fx01 = f_001*(1 - w[..., 0:1]) + f_101*w[..., 0:1]
         fx10 = f_010*(1 - w[..., 0:1]) + f_110*w[..., 0:1]
         fx11 = f_011*(1 - w[..., 0:1]) + f_111*w[..., 0:1]
         
         fxy0 = fx00*(1 - w[..., 1:2]) + fx10*w[..., 1:2]
         fxy1 = fx01*(1 - w[..., 1:2]) + fx11*w[..., 1:2]
         
         fxyz = fxy0*(1 - w[..., 2:3]) + fxy1*w[..., 2:3]
         
         feats.append(fxyz.squeeze(1))
         
      return torch.cat(feats, dim=-1)
      
