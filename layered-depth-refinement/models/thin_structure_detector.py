"""
Thin Structure Detection Module

Detects where thin structures exist using multi-cue analysis:
- High depth gradients
- Low texture (uniform color)
- Depth discontinuities
- High base depth uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThinStructureDetector(nn.Module):
    """
    Detects where thin structures exist using multi-cue analysis.
    
    Thin structures have:
    - High depth gradients
    - Low texture (uniform color)
    - Depth discontinuities
    - High base depth uncertainty
    """
    def __init__(self, feature_dim=256):
        super().__init__()
        
        # Gradient analysis
        self.gradient_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU()
        )
        
        # Texture analysis (from RGB)
        self.texture_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU()
        )
        
        # Combined thin detection
        self.detector = nn.Sequential(
            nn.Conv2d(32 + 32 + feature_dim, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()  # Probability [0, 1]
        )
        
    def compute_depth_gradients(self, depth):
        """Compute depth gradient magnitude using Sobel filters."""
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
            dtype=depth.dtype, device=depth.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
            dtype=depth.dtype, device=depth.device
        ).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(depth, sobel_x, padding=1)
        grad_y = F.conv2d(depth, sobel_y, padding=1)
        
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return gradient_magnitude
    
    def forward(self, rgb, base_depth, features):
        """
        Args:
            rgb: (B, 3, H, W)
            base_depth: (B, 1, H, W)
            features: (B, C, H, W) from encoder
        
        Returns:
            thin_mask: (B, 1, H, W) - probability of thin structure
        """
        # 1. Depth gradient analysis
        depth_grad = self.compute_depth_gradients(base_depth)
        grad_features = self.gradient_conv(depth_grad)
        
        # 2. Texture analysis
        # Resize RGB to match feature spatial dimensions if needed
        if rgb.shape[2:] != features.shape[2:]:
            rgb_resized = F.interpolate(rgb, size=features.shape[2:], mode='bilinear', align_corners=False)
        else:
            rgb_resized = rgb
        texture_features = self.texture_conv(rgb_resized)
        
        # Resize gradient features to match if needed
        if grad_features.shape[2:] != features.shape[2:]:
            grad_features = F.interpolate(grad_features, size=features.shape[2:], mode='bilinear', align_corners=False)
        
        # 3. Combine all cues
        combined = torch.cat([grad_features, texture_features, features], dim=1)
        
        # 4. Predict thin structure probability
        thin_mask = self.detector(combined)
        
        return thin_mask

