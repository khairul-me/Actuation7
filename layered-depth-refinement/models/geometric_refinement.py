"""
Multi-View Geometric Refinement Module

Provides geometric consistency checking and refinement
when dual-camera views are available. Uses Fisher information
framework for uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricConsistencyRefinement(nn.Module):
    """
    Refines layered depth using geometric consistency between views.
    
    - Warps layers from view2 to view1
    - Computes consistency error
    - Applies learned residual correction (max ±5cm)
    """
    def __init__(self, d_model=256, max_correction=0.05):
        super().__init__()
        self.max_correction = max_correction
        
        # Consistency-guided refinement network
        self.refine_net = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),  # layers + consistency_error
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1),  # One correction per layer
            nn.Tanh()  # Bounded [-1, 1] * max_correction
        )
    
    def warp_depth(self, depth, K_src, K_tgt, R, T):
        """
        Warp depth map from source to target view.
        
        Args:
            depth: (B, 1, H, W) source depth
            K_src: (B, 3, 3) source intrinsics
            K_tgt: (B, 3, 3) target intrinsics
            R: (B, 3, 3) rotation from source to target
            T: (B, 3, 1) translation from source to target
        
        Returns:
            warped_depth: (B, 1, H, W) in target frame
            valid_mask: (B, 1, H, W) boolean
        """
        B, _, H, W = depth.shape
        device = depth.device
        
        # Create pixel grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        ones = torch.ones_like(x_coords)
        pixel_coords = torch.stack([x_coords, y_coords, ones], dim=0)  # (3, H, W)
        pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 3, H, W)
        
        # Backproject to 3D
        K_src_inv = torch.inverse(K_src)
        pixel_flat = pixel_coords.reshape(B, 3, -1)  # (B, 3, H*W)
        cam_coords = torch.bmm(K_src_inv, pixel_flat)  # (B, 3, H*W)
        depth_flat = depth.reshape(B, 1, -1)  # (B, 1, H*W)
        points_3d = cam_coords * depth_flat  # (B, 3, H*W)
        
        # Transform to target view
        points_tgt = torch.bmm(R, points_3d) + T  # (B, 3, H*W)
        
        # Project to target image
        pixel_tgt = torch.bmm(K_tgt, points_tgt)  # (B, 3, H*W)
        pixel_tgt = pixel_tgt[:, :2] / (pixel_tgt[:, 2:3] + 1e-8)  # (B, 2, H*W)
        
        # Normalize to [-1, 1] for grid_sample
        pixel_tgt_norm = pixel_tgt.clone()
        pixel_tgt_norm[:, 0] = 2.0 * pixel_tgt[:, 0] / (W - 1) - 1.0
        pixel_tgt_norm[:, 1] = 2.0 * pixel_tgt[:, 1] / (H - 1) - 1.0
        
        grid = pixel_tgt_norm.reshape(B, 2, H, W).permute(0, 2, 3, 1)  # (B, H, W, 2)
        
        # Warp depth
        warped_depth = F.grid_sample(
            depth, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )
        
        # Valid mask (within image bounds)
        valid = (grid[:, :, :, 0].abs() <= 1) & (grid[:, :, :, 1].abs() <= 1)
        valid = valid.unsqueeze(1)  # (B, 1, H, W)
        
        # Target depth for consistency check
        target_depth = points_tgt[:, 2:3].reshape(B, 1, H, W)
        
        return warped_depth, valid, target_depth
    
    def forward(self, layer_depths1, layer_depths2, K1, K2, R, T):
        """
        Args:
            layer_depths1: (B, K, H, W) layers from view 1
            layer_depths2: (B, K, H, W) layers from view 2
            K1, K2: (B, 3, 3) intrinsics
            R: (B, 3, 3) rotation from view2 to view1
            T: (B, 3, 1) translation from view2 to view1
        
        Returns:
            refined_layers: (B, K, H, W) refined layer depths
            consistency_map: (B, 1, H, W) consistency error
        """
        B, K, H, W = layer_depths1.shape
        
        # Compute consistency error for composite depth
        composite1 = layer_depths1.mean(dim=1, keepdim=True)
        composite2 = layer_depths2.mean(dim=1, keepdim=True)
        
        warped2, valid, target_depth = self.warp_depth(composite2, K2, K1, R, T)
        
        consistency_error = torch.abs(composite1 - warped2) * valid.float()
        
        # Refinement network input: layers + consistency error
        refine_input = torch.cat([layer_depths1, consistency_error], dim=1)  # (B, K+1, H, W)
        
        # Predict corrections
        corrections = self.refine_net(refine_input) * self.max_correction  # (B, K, H, W)
        
        # Apply corrections
        refined_layers = layer_depths1 + corrections
        
        return refined_layers, consistency_error


class GeometricUncertaintyPredictor(nn.Module):
    """
    Fisher information-based uncertainty estimation.
    
    σ² ∝ d² / (b² · cos²θ₁ · cos²θ₂)
    
    Where:
    - d = depth
    - b = baseline
    - θ₁, θ₂ = viewing angles from surface normal
    """
    def __init__(self):
        super().__init__()
        
        # Learned calibration of geometric uncertainty
        self.calibration_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # depth + geometric_sigma + consistency
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Softplus()  # Positive uncertainty
        )
    
    def compute_geometric_uncertainty(self, depth, K1, K2, R, T):
        """
        Compute Fisher information-based depth uncertainty.
        
        Args:
            depth: (B, 1, H, W)
            K1, K2: (B, 3, 3)
            R: (B, 3, 3)
            T: (B, 3, 1)
        
        Returns:
            sigma: (B, 1, H, W) geometric uncertainty
        """
        B, _, H, W = depth.shape
        
        # Baseline magnitude
        baseline = torch.norm(T.squeeze(-1), dim=1, keepdim=True)  # (B, 1)
        baseline = baseline.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        
        # σ² ∝ d² / b²  (simplified, ignoring viewing angles for now)
        sigma_sq = (depth ** 2) / (baseline ** 2 + 1e-8)
        sigma = torch.sqrt(sigma_sq + 1e-8)
        
        return sigma
    
    def forward(self, layer_depths1, layer_depths2, K1, K2, R, T, consistency_error=None):
        """
        Args:
            layer_depths1, layer_depths2: (B, K, H, W)
            K1, K2: (B, 3, 3)
            R: (B, 3, 3), T: (B, 3, 1)
            consistency_error: (B, 1, H, W) optional
        
        Returns:
            uncertainty: (B, 1, H, W) calibrated uncertainty
        """
        # Use composite depth for uncertainty
        composite = layer_depths1.mean(dim=1, keepdim=True)
        
        # Geometric uncertainty
        geo_sigma = self.compute_geometric_uncertainty(composite, K1, K2, R, T)
        
        # Consistency-based uncertainty
        if consistency_error is None:
            consistency_error = torch.zeros_like(composite)
        
        # Combine and calibrate
        calib_input = torch.cat([composite, geo_sigma, consistency_error], dim=1)
        uncertainty = self.calibration_net(calib_input)
        
        return uncertainty

