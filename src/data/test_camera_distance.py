import json
import numpy as np

with open('skow/robot/transforms.json', 'r') as f:
    meta = json.load(f)

positions = []
for frame in meta['frames']:
    c2w = np.array(frame['transform_matrix'])
    position = c2w[:3, 3]
    positions.append(position)

positions = np.array(positions)
dists = np.linalg.norm(positions, axis=1)
print(f"Min distance: {np.min(dists):.2f} meters, Max distance: {np.max(dists):.2f} meters")
