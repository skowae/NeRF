import torch
import os
import sys

# Add the parent directory (src/) to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.FastNGPNeRF import FastNGPNeRF

# --------------------------------------------------
# 1. Instantiate the *same* model class and optimizer
# --------------------------------------------------
model = FastNGPNeRF()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # same optim class & args

# --------------------------------------------------
# 2. Decide where you want the checkpoint to land
#    – map_location lets you load a GPU checkpoint on CPU or vice‑versa
# --------------------------------------------------
results_dir = '/home/skowae1/JHU_EP/dlcv/NeRF/src/results/SkowCapture/20250501_190308'
ckpt_path = os.path.join(results_dir, 
                         "weights", 
                         "best_0.405.pth")  # example
checkpoint = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# 3. Restore the state
# --------------------------------------------------
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
start_epoch = checkpoint["epoch"] + 1        # so you can resume training

torch.save(model.state_dict(), os.path.join(results_dir, "weights", "new_best_model.pt"))

# (Optional) if you use a learning‑rate scheduler:
# scheduler.load_state_dict(checkpoint["scheduler"])
# --------------------------------------------------
# 4. Put the model on the desired device and (if evaluating) switch to eval()
# --------------------------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# If you’re going to *continue* training:
# model.train()
# If you’re only going to evaluate/infer:
# model.eval()
