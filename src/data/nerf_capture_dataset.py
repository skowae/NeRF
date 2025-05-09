import os
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T 

# Define the NeRFCapture dataset class
class NeRFCaptureDataset(Dataset):
   def __init__(self, scene_dir, split="train", img_wh=(100, 75), transform=None, subset=None):
      super().__init__()
      
      self.scene_dir = scene_dir
      self.img_wh = img_wh
      self.transform = transform
      
      # Load the transforms
      with open(os.path.join(scene_dir, 'transforms.json'), 'r') as f:
         self.meta = json.load(f)
      
      self.image_paths = []
      self.poses = []
      self.fl_x = []
      self.fl_y = []
      self.cx = []
      self.cy = []
      self.w = []
      self.h = []
      self.split = split
      
      frames = self.meta["frames"]
      
      if subset is not None:
         frames = random.sample(frames, subset)
      
      self.frames = frames
      
      # Loop through the frames
      for frame in self.frames:
         fname = os.path.join(scene_dir, frame['file_path'] + ".png")
         self.image_paths.append(fname)
         self.cx.append(frame["cx"])
         self.cy.append(frame["cy"])
         self.fl_x.append(frame["fl_x"])
         self.fl_y.append(frame["fl_y"])
         self.w.append(frame["w"])
         self.h.append(frame["h"])
      
      self.poses = [torch.from_numpy(np.array(frame["transform_matrix"])).float() for frame in self.frames]
      # Normalize poses to a unit sphere
      centers = torch.stack([pose[:3, 3] for pose in self.poses]) # (N, 3)
      scene_center = centers.mean(0)
      dists = (centers - scene_center).norm(dim=-1)
      scene_scale = 1.2*dists.max() # Scaling by factor of 1.2
      
      for pose in self.poses:
         pose[:3, 3] = (pose[:3, 3] - scene_center)/scene_scale
         
      self.scene_center = scene_center
      self.scene_scale = scene_scale
      
      self.N_images = len(self.frames)
      
      # Set up a simple trnasform if none is provided
      if self.transform is None:
         self.transform = T.Compose([
            T.Resize((img_wh[1], img_wh[0])),
            T.ToTensor(),
         ])

      # Divide the data into train and val
      perm = torch.randperm(len(self.poses))
      val_fraction = 0.10
      val_count = int(len(perm)*val_fraction)
      
      self.val_idx = perm[:val_count].tolist()
      self.train_idx = perm[val_count:].tolist()
      
   def __len__(self):
      return len(self.train_idx) if self.split=='train' else len(self.val_idx)
   
   def __getitem__(self, idx):
      # Load the image
      real_idx = self.train_idx[idx] if self.split=='train' else self.val_idx[idx]
      img_path = self.image_paths[real_idx]
      img = Image.open(img_path).convert('RGB')
      orig_w = self.w[real_idx]
      orig_h = self.h[real_idx]
      
      img = self.transform(img) # (3, H, W)

      # Rescale the intrinsics
      new_w, new_h = self.img_wh
      
      scale_w = new_w/orig_w
      scale_h = new_h/orig_h
      
      scaled_fl_x = self.fl_x[real_idx]*scale_w
      scaled_fl_y = self.fl_y[real_idx]*scale_h
      scaled_cx = self.cx[real_idx]*scale_w
      scaled_cy = self.cy[real_idx]*scale_h
      
      # Pack initrinsics
      focal = torch.tensor([scaled_fl_x, scaled_fl_y], dtype=torch.float32)
      principal_point = torch.tensor([scaled_cx, scaled_cy], dtype=torch.float32)
      img_wh = torch.tensor([new_w, new_h], dtype=torch.float32)
      pose = self.poses[real_idx] # (4,4)
      
      return img, pose, focal, principal_point, img_wh
