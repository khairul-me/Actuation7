"""
Classical Baseline: Variance-Weighted and Median Fusion

Simple multi-view depth fusion baselines for comparison.
"""

import torch
import torch.nn.functional as F


def variance_weighted_fusion(depth_maps, weights=None):
    """
    Classical variance-weighted depth fusion.
    
    Args:
        depth_maps: list of (B, 1, H, W) depth maps
        weights: optional list of (B, 1, H, W) weights
    
    Returns:
        fused_depth: (B, 1, H, W)
    """
    if len(depth_maps) == 1:
        return depth_maps[0]
    
    stacked = torch.stack(depth_maps, dim=0)  # (N, B, 1, H, W)
    
    if weights is not None:
        weights_stacked = torch.stack(weights, dim=0)
    else:
        # Inverse variance weighting
        # Estimate variance from the disagreement between views
        mean = stacked.mean(dim=0, keepdim=True)
        variance = ((stacked - mean) ** 2).mean(dim=0, keepdim=True) + 1e-8
        weights_stacked = 1.0 / variance
        weights_stacked = weights_stacked.expand_as(stacked)
    
    # Normalized weighted average
    weights_sum = weights_stacked.sum(dim=0, keepdim=True)
    fused = (stacked * weights_stacked).sum(dim=0) / (weights_sum.squeeze(0) + 1e-8)
    
    return fused


def median_fusion(depth_maps):
    """
    Simple median fusion baseline.
    
    Args:
        depth_maps: list of (B, 1, H, W) depth maps
    
    Returns:
        fused_depth: (B, 1, H, W)
    """
    if len(depth_maps) == 1:
        return depth_maps[0]
    
    stacked = torch.stack(depth_maps, dim=0)  # (N, B, 1, H, W)
    fused = stacked.median(dim=0).values
    
    return fused

