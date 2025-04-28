from pathlib import Path
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class LLFFDataset(Dataset):
   def __init__(self, scene_dir, split='train', factor=8):
      """
      Args:
         scene_dir (str or Path): Path to the LLFF scene (e.g., `.../data/nerf_llff_data/fern`)
         split (str): One of 'train' or 'val'
         factor (int): Downsampling factor for images
      """
      self.scene_dir = Path(scene_dir)
      self.split = split
      self.factor = factor
      self.near = 2.0
      self.far = 6.0

      # Load poses and bounds
      poses_bounds = np.load(self.scene_dir / 'poses_bounds.npy')
      self.poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
      self.bounds = poses_bounds[:, -2:]
      self.focal = self.poses[0, -1, -1]

      # Extract images
      image_dir = self.scene_dir / f'images_{factor}'
      if not image_dir.exists():
         raise FileNotFoundError(f"Image directory {image_dir} not found.")

      image_paths = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg')) + sorted(image_dir.glob('*.PNG')) + sorted(image_dir.glob('*.JPG'))
      if not image_paths:
         raise FileNotFoundError(f"No images found in {image_dir}")

      self.images = [imageio.v2.imread(p) / 255.0 for p in image_paths]
      self.images = np.stack(self.images, axis=0)  # [N, H, W, 3]
      self.images = torch.from_numpy(self.images).float()

      # Split train and val
      total = len(self.images)
      indices = np.arange(total)
      # val_indices = indices[::8]
      val_indices = []
      train_indices = np.array([i for i in indices if i not in val_indices])

      if split == 'train':
         self.indices = train_indices
      elif split == 'val':
         self.indices = val_indices
      else:
         raise ValueError(f"Unknown split: {split}")

      # Filter poses and images
      self.poses = self.poses[self.indices]
      self.images = self.images[self.indices]

      # Compute height/width
      self.H, self.W = self.images.shape[1:3]
      self.resize_factor = 100 / self.W
      
      # Resize the images
      if self.resize_factor < 1.0:
         new_H = int(self.H * self.resize_factor)
         new_W = int(self.W * self.resize_factor)
         resized = []
         
         for img in self.images:
            img = img.permute(2, 0, 1)  # [3, H, W] for TF.resize
            img_resized = TF.resize(img, [new_H, new_W], interpolation=Image.BILINEAR)
            resized.append(img_resized.permute(2, 1, 0))  # back to [W, H, 3]
            
         self.images = torch.stack(resized)  # [N, H, W, 3]self.focal *= self.resize_factor
         self.H = int(self.H * self.resize_factor)
         self.W = int(self.W * self.resize_factor)

   def __len__(self):
      return len(self.images)

   def __getitem__(self, idx):
      image = self.images[idx]  # [H, W, 3]
      
      pose = self.poses[idx, :3, :4]  # [3, 4]
      return image, torch.from_numpy(pose).float(), self.focal, self.H, self.W, self.bounds[idx]
