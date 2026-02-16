"""
Geometric Utility Functions

Essential 3D geometry operations for multi-view processing.
"""

import torch
import torch.nn.functional as F
import numpy as np


def backproject_depth(depth, K_inv, return_homogeneous=False):
    """
    Backproject depth map to 3D points.
    
    Args:
        depth: (B, 1, H, W)
        K_inv: (B, 3, 3) inverse intrinsics
        return_homogeneous: if True, return (B, 4, H*W)
    
    Returns:
        points_3d: (B, 3, H*W) or (B, 4, H*W)
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
    pixel_flat = pixel_coords.reshape(B, 3, -1)  # (B, 3, H*W)
    
    # Backproject
    cam_coords = torch.bmm(K_inv, pixel_flat)  # (B, 3, H*W)
    depth_flat = depth.reshape(B, 1, -1)  # (B, 1, H*W)
    points_3d = cam_coords * depth_flat  # (B, 3, H*W)
    
    if return_homogeneous:
        ones_flat = torch.ones(B, 1, H * W, device=device)
        points_3d = torch.cat([points_3d, ones_flat], dim=1)
    
    return points_3d


def project_points(points_3d, K):
    """
    Project 3D points to 2D pixel coordinates.
    
    Args:
        points_3d: (B, 3, N)
        K: (B, 3, 3) intrinsics
    
    Returns:
        pixels: (B, 2, N) pixel coordinates
        depths: (B, 1, N) projected depths
    """
    projected = torch.bmm(K, points_3d)  # (B, 3, N)
    depths = projected[:, 2:3, :]  # (B, 1, N)
    pixels = projected[:, :2, :] / (depths + 1e-8)  # (B, 2, N)
    
    return pixels, depths


def transform_points(points_3d, R, T):
    """
    Transform 3D points from one frame to another.
    
    Args:
        points_3d: (B, 3, N)
        R: (B, 3, 3) rotation
        T: (B, 3, 1) translation
    
    Returns:
        transformed: (B, 3, N)
    """
    return torch.bmm(R, points_3d) + T


def compute_epipolar_error(pts1, pts2, F_matrix):
    """
    Compute epipolar error for point correspondences.
    
    Args:
        pts1: (B, N, 2) points in image 1
        pts2: (B, N, 2) points in image 2
        F_matrix: (B, 3, 3) fundamental matrix
    
    Returns:
        error: (B, N) epipolar error per point
    """
    B, N, _ = pts1.shape
    
    # Homogeneous coordinates
    ones = torch.ones(B, N, 1, device=pts1.device)
    pts1_h = torch.cat([pts1, ones], dim=-1)  # (B, N, 3)
    pts2_h = torch.cat([pts2, ones], dim=-1)  # (B, N, 3)
    
    # Epipolar lines: l2 = F @ p1
    lines2 = torch.bmm(pts1_h, F_matrix.transpose(1, 2))  # (B, N, 3)
    
    # Distance from point to line
    numerator = torch.abs((pts2_h * lines2).sum(dim=-1))  # (B, N)
    denominator = torch.sqrt(lines2[:, :, 0] ** 2 + lines2[:, :, 1] ** 2 + 1e-8)
    
    error = numerator / denominator
    return error

