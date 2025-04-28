import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T 

# Define the NeRFCapture dataset class
class NeRFCaptureDataset(Dataset):
   def __init__(self, scene_dir, split='train', img_wh=(100, 75), transform=None):
      super().__init__()
      
      self.scene_dir = scene_dir
      self.split = split   # FOr now use all images for training
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
      
      # Loop through the frames
      for frame in self.meta['frames']:
         fname = os.path.join(scene_dir, frame['file_path'] + ".png")
         self.image_paths.append(fname)
         self.poses.append(np.array(frame['transform_matrix'], dtype=np.float32))
         self.cx.append(frame["cx"])
         self.cy.append(frame["cy"])
         self.fl_x.append(frame["fl_x"])
         self.fl_y.append(frame["fl_y"])
         self.w.append(frame["w"])
         self.h.append(frame["h"])
         
      # (N, 4, 4)
      self.poses = torch.tensor(np.stack(self.poses))
      self.N_images = len(self.image_paths)
      
      # Set up a simple trnasform if none is provided
      if self.transform is None:
         self.transform = T.Compose([
            T.Resize((img_wh[1], img_wh[0])),
            T.ToTensor(),
         ])
      
   def __len__(self):
      return self.N_images
   
   def __getitem__(self, idx):
      # Load the image
      img_path = self.image_paths[idx]
      img = Image.open(img_path).convert('RGB')
      orig_w = self.w[idx]
      orig_h = self.h[idx]
      
      img = self.transform(img) # (3, H, W)

      # Rescale the intrinsics
      new_w, new_h = self.img_wh
      
      scale_w = new_w/orig_w
      scale_h = new_h/orig_h
      
      scaled_fl_x = self.fl_x[idx]*scale_w
      scaled_fl_y = self.fl_y[idx]*scale_h
      scaled_cx = self.cx[idx]*scale_w
      scaled_cy = self.cy[idx]*scale_h
      
      # Pack initrinsics
      focal = torch.tensor([scaled_fl_x, scaled_fl_y], dtype=torch.float32)
      principal_point = torch.tensor([scaled_cx, scaled_cy], dtype=torch.float32)
      img_wh = torch.tensor([new_w, new_h], dtype=torch.float32)
      pose = self.poses[idx] # (4,4)
      
      return img, pose, focal, principal_point, img_wh
