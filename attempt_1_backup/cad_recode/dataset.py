# cad_recode/dataset.py
import os, random
import numpy as np
import torch
from torch.utils.data import Dataset
import cadquery as cq  # CadQuery for executing CAD scripts
from cad_recode.utils import sample_points_on_shape, farthest_point_sample

def farthest_point_sample(points, k):
    """Furthest Point Sampling: select k points from the input point cloud (Nx3) that maximize coverage."""
    points = np.asarray(points)
    N = points.shape[0]
    assert N >= k
    selected_idx = []
    # Start with a random point (or we can pick the first point)
    idx = 0  
    selected_idx.append(idx)
    distances = np.full(N, np.inf)
    # Greedily select the next point farthest from current set
    for _ in range(1, k):
        # update distances to nearest selected point
        dist_to_last = np.linalg.norm(points[idx] - points, axis=1)
        distances = np.minimum(dist_to_last, distances)
        idx = np.argmax(distances)
        selected_idx.append(idx)
    return points[selected_idx]

class CadRecodeDataset(Dataset):
    def __init__(self, root_dir, split='train', n_points=256, noise_std=0.01, noise_prob=0.5):
        self.split = split
        self.n_points = n_points
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        # Gather all .py files in the split directory (including batched subfolders)
        split_dir = os.path.join(root_dir, split)
        self.files = []
        for batch_dir in os.listdir(split_dir):
            batch_path = os.path.join(split_dir, batch_dir)
            if os.path.isdir(batch_path):
                for fname in os.listdir(batch_path):
                    if fname.endswith('.py'):
                        self.files.append(os.path.join(batch_path, fname))
        self.files.sort()  # sort by name for consistency
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        # Read CadQuery code
        with open(file_path, 'r') as f:
            code = f.read()
        # Execute the CadQuery script to get the CAD shape
        try:
            # Create a namespace with cadquery imported as cq
            local_vars = {}
            exec(code, {"cq": cq}, local_vars)
        except Exception as e:
            raise RuntimeError(f"Failed to execute CAD script {file_path}: {e}")
        # The script typically assigns the final shape to a variable (e.g. r)
        # We'll try common variable names or use the last Workplane in locals
        shape = None
        if "result" in local_vars:
            shape = local_vars["result"]
        elif "r" in local_vars:
            shape = local_vars["r"]
        elif "shape" in local_vars:
            shape = local_vars["shape"]
        # If shape is a CadQuery Workplane, get the underlying solid object
        if isinstance(shape, cq.Workplane):
            try:
                shape = shape.val()  # get the CAD solid from the workplane
            except:
                shape = shape.objects[0]  # fallback: first object
        if shape is None:
            raise RuntimeError(f"No shape found in script {file_path}")
        # Sample points on the surface of the shape
        points = sample_points_on_shape(shape, n_samples=1024)  # implement with mesh tessellation or point sampling
        # Downsample to n_points (256) using FPS
        if points.shape[0] > self.n_points:
            points = farthest_point_sample(points, self.n_points)
        # Normalize the point cloud (center and scale)
        centroid = points.mean(axis=0)
        points = points - centroid
        scale = np.linalg.norm(points, axis=1).max()
        if scale > 1e-6:
            points = points / scale
        # Optionally add Gaussian noise (only in training mode)
        if self.split == 'train' and random.random() < self.noise_prob:
            noise = np.random.normal(loc=0.0, scale=self.noise_std, size=points.shape)
            points = points + noise
        # Convert to torch tensors
        points = torch.from_numpy(points.astype(np.float32))  # shape (256,3)
        code_str = code  # ground truth CadQuery code as string
        return points, code_str
