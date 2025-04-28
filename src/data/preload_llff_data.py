import os
import sys

# Add the parent directory (src/) to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import imageio
from tqdm import tqdm
from typing import Tuple
from utils.get_rays import *


def load_llff_data(data_dir: str):
   poses_bounds = np.load(os.path.join(data_dir, "poses_bounds.npy"))  # [N, 17]
   poses = poses_bounds[:, :15].reshape(-1, 3, 5)                       # [N, 3, 5]
   bounds = poses_bounds[:, 15:]                                       # [N, 2] -> [near, far]

   img_dir = os.path.join(data_dir, 'images')
   img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('png', 'jpg'))])
   images = [imageio.v2.imread(os.path.join(img_dir, f)) for f in img_files]
   images = [torch.from_numpy(img.astype(np.float32) / 255.0) for img in images]

   return poses, bounds, images


def split_dataset_indices(N: int, train_ratio=0.8, val_ratio=0.10) -> Tuple[list, list, list]:
   indices = list(range(N))
   np.random.shuffle(indices)
   train_end = int(train_ratio * N)
   val_end = int((train_ratio + val_ratio) * N)
   return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def precompute_and_split_llff(data_dir: str, out_dir: str, H=378, W=504):
   os.makedirs(out_dir, exist_ok=True)
   poses, bounds, images = load_llff_data(data_dir)
   N = len(images)

   train_idx, val_idx, test_idx = split_dataset_indices(N)
   splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}

   for split, indices in splits.items():
      rays_o_all, rays_d_all, rgbs_all, nears_all, fars_all = [], [], [], [], []

      for i in tqdm(indices, desc=f"Processing {split} set"):
         c2w = torch.eye(4)
         c2w[:3, :4] = torch.from_numpy(poses[i])
         focal = poses[i, -1, -1]

         rays_o, rays_d = get_rays(H, W, focal, c2w)
         rays_o_all.append(rays_o)
         rays_d_all.append(rays_d)
         rgbs_all.append(images[i])

         near = bounds[i, 0]
         far = bounds[i, 1]
         nears_all.append(torch.full((H, W), near))
         fars_all.append(torch.full((H, W), far))

      torch.save({
         "rays_o": torch.stack(rays_o_all),
         "rays_d": torch.stack(rays_d_all),
         "rgb":    torch.stack(rgbs_all),
         "near":   torch.stack(nears_all),
         "far":    torch.stack(fars_all),
         "focal":  poses[0, -1, -1]
      }, os.path.join(out_dir, f"{split}.pt"))


def main():
   # Example usage:
   precompute_and_split_llff(
      data_dir="/home/skowae1/JHU_EP/dlcv/NeRF/src/data/llff/testscene",    # <-- change to your dataset
      out_dir="/home/skowae1/JHU_EP/dlcv/NeRF/src/data/llff/testscene/partitions"    # <-- where to store .pt files
   )
   
if __name__ == '__main__':
   main()
