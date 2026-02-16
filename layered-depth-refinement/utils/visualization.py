"""
Visualization Utilities

Helpers for qualitative results and debugging.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def visualize_layered_depth(layer_depths, layer_alphas, thin_mask=None, 
                           save_path=None, base_depth=None):
    """
    Visualize layered depth prediction.
    
    Args:
        layer_depths: (K, H, W) or (B, K, H, W) - take first batch
        layer_alphas: same shape as layer_depths
        thin_mask: (1, H, W) optional
        save_path: optional path to save figure
        base_depth: (1, H, W) optional base depth for comparison
    """
    if isinstance(layer_depths, torch.Tensor):
        layer_depths = layer_depths.detach().cpu().numpy()
        layer_alphas = layer_alphas.detach().cpu().numpy()
        if thin_mask is not None:
            thin_mask = thin_mask.detach().cpu().numpy()
        if base_depth is not None:
            base_depth = base_depth.detach().cpu().numpy()
    
    # Take first batch if batched
    if layer_depths.ndim == 4:
        layer_depths = layer_depths[0]
        layer_alphas = layer_alphas[0]
    if thin_mask is not None and thin_mask.ndim == 4:
        thin_mask = thin_mask[0]
    if base_depth is not None and base_depth.ndim == 4:
        base_depth = base_depth[0]
    
    K = layer_depths.shape[0]
    n_cols = K + 2 + (1 if thin_mask is not None else 0) + (1 if base_depth is not None else 0)
    
    fig, axes = plt.subplots(2, max(n_cols, 4), figsize=(4 * max(n_cols, 4), 8))
    
    # Row 1: Depth layers
    for k in range(K):
        axes[0, k].imshow(layer_depths[k], cmap='turbo')
        axes[0, k].set_title(f'Layer {k+1} Depth')
        axes[0, k].axis('off')
    
    # Composite depth
    composite = (layer_depths * layer_alphas).sum(axis=0)
    axes[0, K].imshow(composite, cmap='turbo')
    axes[0, K].set_title('Composite Depth')
    axes[0, K].axis('off')
    
    if base_depth is not None:
        axes[0, K + 1].imshow(base_depth.squeeze(), cmap='turbo')
        axes[0, K + 1].set_title('Base Depth')
        axes[0, K + 1].axis('off')
    
    # Row 2: Alpha weights
    for k in range(K):
        axes[1, k].imshow(layer_alphas[k], cmap='hot', vmin=0, vmax=1)
        axes[1, k].set_title(f'Layer {k+1} Alpha')
        axes[1, k].axis('off')
    
    if thin_mask is not None:
        axes[1, K].imshow(thin_mask.squeeze(), cmap='hot', vmin=0, vmax=1)
        axes[1, K].set_title('Thin Mask')
        axes[1, K].axis('off')
    
    # Hide unused axes
    for i in range(2):
        for j in range(max(n_cols, 4)):
            if not axes[i, j].has_data():
                axes[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def visualize_comparison(pred_depth, gt_depth, base_depth=None, 
                        method_name='Ours', save_path=None):
    """
    Side-by-side comparison visualization.
    """
    if isinstance(pred_depth, torch.Tensor):
        pred_depth = pred_depth.detach().cpu().numpy().squeeze()
        gt_depth = gt_depth.detach().cpu().numpy().squeeze()
        if base_depth is not None:
            base_depth = base_depth.detach().cpu().numpy().squeeze()
    
    n_cols = 3 if base_depth is not None else 2
    fig, axes = plt.subplots(1, n_cols + 1, figsize=(5 * (n_cols + 1), 5))
    
    idx = 0
    if base_depth is not None:
        axes[idx].imshow(base_depth, cmap='turbo')
        axes[idx].set_title('Base Depth')
        axes[idx].axis('off')
        idx += 1
    
    axes[idx].imshow(pred_depth, cmap='turbo')
    axes[idx].set_title(f'{method_name} Prediction')
    axes[idx].axis('off')
    idx += 1
    
    axes[idx].imshow(gt_depth, cmap='turbo')
    axes[idx].set_title('Ground Truth')
    axes[idx].axis('off')
    idx += 1
    
    # Error map
    valid = gt_depth > 0
    error = np.abs(pred_depth - gt_depth)
    error[~valid] = 0
    axes[idx].imshow(error, cmap='hot')
    axes[idx].set_title('Absolute Error')
    axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig

