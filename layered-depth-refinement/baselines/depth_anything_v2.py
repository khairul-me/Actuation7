"""
Depth Anything V2 Baseline

Monocular foundation depth model.
Used both as standalone baseline and as backbone for our method.
"""

import os
import sys
import torch
import torch.nn.functional as F


class DepthAnythingV2Baseline:
    """
    Wrapper for Depth Anything V2 as a baseline method.
    """
    def __init__(self, model_size='vitl', weights_path=None, device='cuda'):
        self.device = device
        
        # Setup paths
        if weights_path is None:
            weights_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'Depth-Anything-V2', 'checkpoints', 'depth_anything_v2_vitl.pth'
            )
        
        da_repo = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'Depth-Anything-V2'
        )
        if da_repo not in sys.path:
            sys.path.insert(0, da_repo)
        
        from depth_anything_v2.dpt import DepthAnythingV2
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        self.model = DepthAnythingV2(**model_configs[model_size])
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        self.model = self.model.to(device).eval()
    
    @torch.no_grad()
    def predict(self, rgb):
        """
        Args:
            rgb: (B, 3, H, W) tensor, normalized [0, 1]
        Returns:
            depth: (B, 1, H, W)
        """
        depth = self.model.infer_image(rgb)
        if isinstance(depth, torch.Tensor):
            if depth.dim() == 2:
                depth = depth.unsqueeze(0).unsqueeze(0)
            elif depth.dim() == 3:
                depth = depth.unsqueeze(1)
        else:
            import numpy as np
            depth = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0).to(self.device)
        return depth
    
    def __call__(self, batch):
        """Process a batch."""
        rgb = batch['rgb'].to(self.device)
        depth = self.predict(rgb)
        return {'depth': depth, 'composite_depth': depth}

