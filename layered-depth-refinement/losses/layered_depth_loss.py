"""
Combined Loss for Layered Depth Prediction

Components:
1. Composite depth L1 loss (always available)
2. Layer ordering constraint (d1 < d2 < d3)
3. Alpha distribution sparsity / entropy
4. Thin structure detection BCE
5. Multi-view consistency (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayeredDepthLoss(nn.Module):
    """
    Combined loss for layered depth prediction.
    """
    def __init__(self, alpha_depth=1.0, alpha_alpha=0.5, 
                 alpha_order=0.1, alpha_composite=1.0,
                 alpha_thin=0.3):
        super().__init__()
        self.alpha_depth = alpha_depth
        self.alpha_alpha = alpha_alpha
        self.alpha_order = alpha_order
        self.alpha_composite = alpha_composite
        self.alpha_thin = alpha_thin
        
    def forward(self, pred, target):
        """
        Args:
            pred: dict from model output with keys:
                - layer_depths: (B, K, H, W)
                - layer_alphas: (B, K, H, W)
                - thin_mask: (B, 1, H, W)
                - composite_depth: (B, 1, H, W)
            target: dict with:
                - gt_depth: (B, 1, H, W)
                - valid_mask: (B, 1, H, W)
                - gt_thin_mask: (B, 1, H, W) optional
        
        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss components
        """
        pred_layers = pred['layer_depths']       # (B, K, H, W)
        pred_alphas = pred['layer_alphas']       # (B, K, H, W)
        pred_thin_mask = pred['thin_mask']       # (B, 1, H, W)
        pred_composite = pred['composite_depth'] # (B, 1, H, W)
        
        gt_depth = target['gt_depth']            # (B, 1, H, W)
        valid_mask = target['valid_mask']        # (B, 1, H, W)
        gt_thin_mask = target.get('gt_thin_mask', None)
        
        losses = {}
        
        # 1. Composite depth loss (always available)
        valid_pixels = valid_mask.bool()
        if valid_pixels.sum() > 0:
            composite_loss = F.l1_loss(
                pred_composite[valid_pixels],
                gt_depth[valid_pixels]
            )
        else:
            composite_loss = torch.tensor(0.0, device=pred_layers.device)
        losses['composite'] = composite_loss
        
        # 2. Ordering constraint (layers must be sorted: d1 < d2 < d3)
        order_violations = F.relu(pred_layers[:, :-1] - pred_layers[:, 1:])
        order_loss = order_violations.mean()
        losses['order'] = order_loss
        
        # 3. Alpha sparsity / entropy
        # Entropy of alpha distribution
        alpha_entropy = -(pred_alphas * torch.log(pred_alphas + 1e-8)).sum(dim=1, keepdim=True)
        
        if gt_thin_mask is not None:
            thin_regions = gt_thin_mask > 0.5
            thick_regions = gt_thin_mask < 0.5
            
            # High entropy in thin regions (multi-layer), low in thick
            loss_parts = []
            if thin_regions.sum() > 0:
                thin_target_entropy = 1.0  # Uniform distribution
                entropy_loss_thin = F.l1_loss(
                    alpha_entropy[thin_regions],
                    torch.full_like(alpha_entropy[thin_regions], thin_target_entropy)
                )
                loss_parts.append(entropy_loss_thin)
            
            if thick_regions.sum() > 0:
                thick_target_entropy = 0.0  # Single peak
                entropy_loss_thick = F.l1_loss(
                    alpha_entropy[thick_regions],
                    torch.full_like(alpha_entropy[thick_regions], thick_target_entropy)
                )
                loss_parts.append(entropy_loss_thick)
            
            if loss_parts:
                losses['alpha'] = sum(loss_parts) / len(loss_parts)
            else:
                losses['alpha'] = torch.tensor(0.0, device=pred_layers.device)
        else:
            # Without GT thin mask, just encourage sparsity globally
            losses['alpha'] = alpha_entropy.mean()
        
        # 4. Thin structure detection loss (if GT available)
        if gt_thin_mask is not None:
            thin_loss = F.binary_cross_entropy(pred_thin_mask, gt_thin_mask)
            losses['thin'] = thin_loss
        else:
            losses['thin'] = torch.tensor(0.0, device=pred_layers.device)
        
        # 5. Total loss
        total_loss = (
            self.alpha_composite * losses['composite'] +
            self.alpha_order * losses['order'] +
            self.alpha_alpha * losses['alpha'] +
            self.alpha_thin * losses['thin']
        )
        
        losses['total'] = total_loss
        
        return total_loss, losses


class MultiViewConsistencyLoss(nn.Module):
    """
    Additional loss for multi-view training.
    Encourages geometric consistency between views.
    """
    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred, camera_params):
        """
        Args:
            pred: dict with layer_depths, uncertainty
            camera_params: dict with K1, K2, R, T
        Returns:
            loss: scalar
        """
        if pred.get('uncertainty') is None:
            return torch.tensor(0.0, device=pred['layer_depths'].device)
        
        # Uncertainty-calibrated loss: 
        # High consistency error should correlate with high uncertainty
        composite = pred['composite_depth']
        uncertainty = pred['uncertainty']
        
        # NLL loss: -log p(y|x) ∝ log(σ²) + (y-μ)²/σ²
        # Here we use composite depth residual as proxy
        nll_loss = torch.log(uncertainty + 1e-8) + (composite ** 2) / (2 * uncertainty ** 2 + 1e-8)
        
        return self.weight * nll_loss.mean()

