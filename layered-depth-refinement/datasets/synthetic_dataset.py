"""
Synthetic Thin Structure Dataset

Loads Blender-rendered images with perfect ground truth layered depth.
Used for Phase 2 validation.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class SyntheticThinStructureDataset(Dataset):
    """
    Dataset for Blender-rendered synthetic thin structure scenes.
    
    Expected directory structure:
        data_dir/
        ├── scene_00000_rgb0001.png
        ├── scene_00000_depth0001.exr
        ├── scene_00000_meta.json
        ├── scene_00001_rgb0001.png
        ...
    """
    def __init__(self, data_dir, split='train', transform=None, 
                 image_size=(480, 848), max_depth=1.0):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.max_depth = max_depth
        
        # Default transforms
        if transform is None:
            self.rgb_transform = T.Compose([
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.rgb_transform = transform
        
        # Find all scenes
        self.scenes = self._find_scenes()
        
        # Split
        n_total = len(self.scenes)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        if split == 'train':
            self.scenes = self.scenes[:n_train]
        elif split == 'val':
            self.scenes = self.scenes[n_train:n_train + n_val]
        elif split == 'test':
            self.scenes = self.scenes[n_train + n_val:]
    
    def _find_scenes(self):
        """Find all scene IDs in the data directory."""
        scenes = []
        if not os.path.exists(self.data_dir):
            return scenes
            
        for f in sorted(os.listdir(self.data_dir)):
            if f.endswith('_meta.json'):
                scene_id = f.replace('_meta.json', '')
                scenes.append(scene_id)
        return scenes
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        scene_id = self.scenes[idx]
        
        # Load RGB
        rgb_path = os.path.join(self.data_dir, f"{scene_id}_rgb0001.png")
        if os.path.exists(rgb_path):
            rgb = Image.open(rgb_path).convert('RGB')
            rgb = self.rgb_transform(rgb)
        else:
            rgb = torch.zeros(3, *self.image_size)
        
        # Load depth (EXR or NPY)
        depth_path_exr = os.path.join(self.data_dir, f"{scene_id}_depth0001.exr")
        depth_path_npy = os.path.join(self.data_dir, f"{scene_id}_depth.npy")
        
        if os.path.exists(depth_path_npy):
            depth = np.load(depth_path_npy).astype(np.float32)
            depth = torch.from_numpy(depth).unsqueeze(0)
        elif os.path.exists(depth_path_exr):
            try:
                import imageio
                depth = imageio.imread(depth_path_exr).astype(np.float32)
                if depth.ndim == 3:
                    depth = depth[:, :, 0]
                depth = torch.from_numpy(depth).unsqueeze(0)
            except Exception:
                depth = torch.zeros(1, *self.image_size)
        else:
            depth = torch.zeros(1, *self.image_size)
        
        # Resize depth
        if depth.shape[1:] != tuple(self.image_size):
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0), size=self.image_size, mode='nearest'
            ).squeeze(0)
        
        # Valid mask
        valid_mask = (depth > 0) & (depth < self.max_depth)
        
        # Load metadata
        meta_path = os.path.join(self.data_dir, f"{scene_id}_meta.json")
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)
        
        return {
            'rgb': rgb,
            'gt_depth': depth,
            'valid_mask': valid_mask.float(),
            'scene_id': scene_id,
            'metadata': metadata
        }

