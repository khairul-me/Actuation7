"""
Depth Evaluation Metrics

Includes standard metrics (MAE, RMSE, AbsRel, δ accuracy)
plus thin-structure-specific metrics and uncertainty calibration.
"""

import numpy as np
import torch


class DepthMetrics:
    """
    Comprehensive depth evaluation metrics.
    
    Supports evaluation on:
    - All pixels
    - Thin structure pixels only
    - Thick structure pixels only
    """
    
    @staticmethod
    def compute_all(pred_depth, gt_depth, valid_mask, thin_mask=None):
        """
        Compute all metrics.
        
        Args:
            pred_depth: (B, 1, H, W) or (H, W)
            gt_depth: same shape as pred_depth
            valid_mask: same shape, boolean/float
            thin_mask: optional, same shape
        
        Returns:
            dict with metric names and values
        """
        # Flatten to 1D
        if isinstance(pred_depth, torch.Tensor):
            pred = pred_depth.detach().cpu().numpy()
            gt = gt_depth.detach().cpu().numpy()
            valid = valid_mask.detach().cpu().numpy().astype(bool)
            if thin_mask is not None:
                thin = thin_mask.detach().cpu().numpy() > 0.5
            else:
                thin = None
        else:
            pred = pred_depth
            gt = gt_depth
            valid = valid_mask.astype(bool)
            thin = thin_mask > 0.5 if thin_mask is not None else None
        
        pred = pred.flatten()
        gt = gt.flatten()
        valid = valid.flatten()
        
        results = {}
        
        # All pixels
        if valid.sum() > 0:
            p = pred[valid]
            g = gt[valid]
            results.update(DepthMetrics._compute_subset(p, g, prefix='all'))
        
        # Thin pixels only
        if thin is not None:
            thin = thin.flatten()
            thin_valid = valid & thin
            if thin_valid.sum() > 0:
                p = pred[thin_valid]
                g = gt[thin_valid]
                results.update(DepthMetrics._compute_subset(p, g, prefix='thin'))
            
            # Thick pixels
            thick_valid = valid & ~thin
            if thick_valid.sum() > 0:
                p = pred[thick_valid]
                g = gt[thick_valid]
                results.update(DepthMetrics._compute_subset(p, g, prefix='thick'))
        
        return results
    
    @staticmethod
    def _compute_subset(pred, gt, prefix='all'):
        """Compute metrics for a subset of pixels."""
        results = {}
        
        # MAE
        mae = np.mean(np.abs(pred - gt))
        results[f'{prefix}_mae'] = float(mae)
        
        # RMSE
        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        results[f'{prefix}_rmse'] = float(rmse)
        
        # AbsRel
        abs_rel = np.mean(np.abs(pred - gt) / (gt + 1e-8))
        results[f'{prefix}_absrel'] = float(abs_rel)
        
        # Delta accuracy thresholds
        ratio = np.maximum(pred / (gt + 1e-8), gt / (pred + 1e-8))
        
        for threshold, name in [(1.05, 'delta_105'), (1.10, 'delta_110'), (1.25, 'delta_125')]:
            results[f'{prefix}_{name}'] = float(np.mean(ratio < threshold) * 100)
        
        return results
    
    @staticmethod
    def compute_uncertainty_calibration(pred_depth, gt_depth, uncertainty, valid_mask, n_bins=10):
        """
        Compute uncertainty calibration metrics.
        
        Well-calibrated uncertainty: predicted σ should correlate with actual error.
        
        Returns:
            ece: Expected Calibration Error
            bin_info: dict with per-bin statistics
        """
        if isinstance(pred_depth, torch.Tensor):
            pred = pred_depth.detach().cpu().numpy().flatten()
            gt = gt_depth.detach().cpu().numpy().flatten()
            unc = uncertainty.detach().cpu().numpy().flatten()
            valid = valid_mask.detach().cpu().numpy().flatten().astype(bool)
        else:
            pred = pred_depth.flatten()
            gt = gt_depth.flatten()
            unc = uncertainty.flatten()
            valid = valid_mask.flatten().astype(bool)
        
        pred = pred[valid]
        gt = gt[valid]
        unc = unc[valid]
        
        # Sort by predicted uncertainty
        sort_idx = np.argsort(unc)
        pred_sorted = pred[sort_idx]
        gt_sorted = gt[sort_idx]
        unc_sorted = unc[sort_idx]
        
        # Bin
        bin_size = len(pred_sorted) // n_bins
        bin_info = []
        ece = 0.0
        
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(pred_sorted)
            
            bin_pred = pred_sorted[start:end]
            bin_gt = gt_sorted[start:end]
            bin_unc = unc_sorted[start:end]
            
            actual_error = np.mean(np.abs(bin_pred - bin_gt))
            predicted_unc = np.mean(bin_unc)
            
            bin_info.append({
                'bin': i,
                'actual_error': float(actual_error),
                'predicted_uncertainty': float(predicted_unc),
                'n_samples': end - start
            })
            
            ece += np.abs(actual_error - predicted_unc) * (end - start)
        
        ece /= len(pred_sorted)
        
        return float(ece), bin_info

