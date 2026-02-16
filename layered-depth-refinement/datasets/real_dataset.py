"""
Real Dual-Camera Dataset

Loads real data captured from dual RealSense D405 cameras.
Supports both single-view and multi-view modes.
"""

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class RealDualCameraDataset(Dataset):
    """
    Dataset for real dual-camera captures.
    
    Expected directory structure:
        data_dir/
        ├── rgb_top/
        │   ├── 000000.jpg
        │   ├── 000001.jpg
        │   ...
        ├── rgb_angled/
        │   ├── 000000.jpg
        │   ...
        ├── depth_top/
        │   ├── 000000.npy
        │   ...
        ├── depth_angled/
        │   ├── 000000.npy
        │   ...
        ├── gt_depth/           (optional, from triangulation)
        │   ├── 000000.npy
        │   ...
        ├── gt_thin_mask/       (optional, from annotation)
        │   ├── 000000.npy
        │   ...
        └── calibration.json
    """
    def __init__(self, data_dir, split='train', mode='dual',
                 image_size=(480, 848), max_depth=1.0, transform=None):
        """
        Args:
            data_dir: path to dataset root
            split: 'train', 'val', or 'test'
            mode: 'single' (top camera only) or 'dual' (both cameras)
            image_size: target (H, W)
            max_depth: maximum valid depth in meters
            transform: optional custom transforms
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.mode = mode
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
        
        # Load calibration
        self.calibration = self._load_calibration()
        
        # Find frame IDs
        self.frame_ids = self._find_frames()
        
        # Split
        n_total = len(self.frame_ids)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        if split == 'train':
            self.frame_ids = self.frame_ids[:n_train]
        elif split == 'val':
            self.frame_ids = self.frame_ids[n_train:n_train + n_val]
        elif split == 'test':
            self.frame_ids = self.frame_ids[n_train + n_val:]
    
    def _load_calibration(self):
        """Load camera calibration parameters."""
        calib_path = os.path.join(self.data_dir, 'calibration.json')
        if os.path.exists(calib_path):
            with open(calib_path) as f:
                return json.load(f)
        return None
    
    def _find_frames(self):
        """Find all frame IDs."""
        rgb_dir = os.path.join(self.data_dir, 'rgb_top')
        if not os.path.exists(rgb_dir):
            return []
        
        frame_ids = []
        for f in sorted(os.listdir(rgb_dir)):
            if f.endswith(('.jpg', '.png')):
                frame_id = os.path.splitext(f)[0]
                frame_ids.append(frame_id)
        return frame_ids
    
    def _load_image(self, path):
        """Load and transform an image."""
        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            return self.rgb_transform(img)
        return torch.zeros(3, *self.image_size)
    
    def _load_depth(self, path):
        """Load a depth map."""
        if os.path.exists(path):
            depth = np.load(path).astype(np.float32)
            depth = torch.from_numpy(depth)
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)
            if depth.shape[1:] != tuple(self.image_size):
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(0), size=self.image_size, mode='nearest'
                ).squeeze(0)
            return depth
        return torch.zeros(1, *self.image_size)
    
    def _get_camera_tensors(self):
        """Get camera intrinsics and extrinsics as tensors."""
        if self.calibration is None:
            return None
        
        try:
            K1 = torch.tensor(self.calibration['intrinsics_top']['K'], dtype=torch.float32)
            K2 = torch.tensor(self.calibration['intrinsics_angled']['K'], dtype=torch.float32)
            R = torch.tensor(self.calibration['extrinsics']['R'], dtype=torch.float32)
            T = torch.tensor(self.calibration['extrinsics']['T'], dtype=torch.float32)
            
            if T.dim() == 1:
                T = T.unsqueeze(-1)
            
            return {'K1': K1, 'K2': K2, 'R': R, 'T': T}
        except (KeyError, TypeError):
            return None
    
    def __len__(self):
        return len(self.frame_ids)
    
    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        
        # Load top camera (primary view)
        rgb_top = self._load_image(os.path.join(self.data_dir, 'rgb_top', f'{frame_id}.jpg'))
        depth_top = self._load_depth(os.path.join(self.data_dir, 'depth_top', f'{frame_id}.npy'))
        
        sample = {
            'rgb': rgb_top,
            'raw_depth': depth_top,
            'frame_id': frame_id,
        }
        
        # Load ground truth if available
        gt_path = os.path.join(self.data_dir, 'gt_depth', f'{frame_id}.npy')
        if os.path.exists(gt_path):
            sample['gt_depth'] = self._load_depth(gt_path)
            sample['valid_mask'] = ((sample['gt_depth'] > 0) & 
                                    (sample['gt_depth'] < self.max_depth)).float()
        else:
            sample['valid_mask'] = ((depth_top > 0) & 
                                    (depth_top < self.max_depth)).float()
        
        # Load thin mask if available
        thin_path = os.path.join(self.data_dir, 'gt_thin_mask', f'{frame_id}.npy')
        if os.path.exists(thin_path):
            sample['gt_thin_mask'] = self._load_depth(thin_path)
        
        # Load dual camera data if requested
        if self.mode == 'dual':
            rgb_angled = self._load_image(
                os.path.join(self.data_dir, 'rgb_angled', f'{frame_id}.jpg')
            )
            depth_angled = self._load_depth(
                os.path.join(self.data_dir, 'depth_angled', f'{frame_id}.npy')
            )
            sample['rgb2'] = rgb_angled
            sample['raw_depth2'] = depth_angled
            sample['camera_params'] = self._get_camera_tensors()
        
        return sample

