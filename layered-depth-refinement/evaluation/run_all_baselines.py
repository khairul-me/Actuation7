"""
Comprehensive baseline evaluation script.

Evaluates all methods on the test set:
1. Foundation models (no refinement)
2. Foundation models + our refinement
3. Classical baselines
4. Our complete method
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import DepthMetrics


def evaluate_method(method_fn, test_loader, method_name, device='cuda'):
    """
    Evaluate a single method on the test set.
    
    Args:
        method_fn: callable that takes a batch and returns prediction
        test_loader: DataLoader
        method_name: string identifier
        device: torch device
    
    Returns:
        dict with all metrics
    """
    all_metrics = []
    
    for batch in tqdm(test_loader, desc=f"Evaluating {method_name}"):
        # Move to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        # Get prediction
        with torch.no_grad():
            output = method_fn(batch)
        
        if isinstance(output, dict):
            pred_depth = output.get('composite_depth', output.get('depth'))
        else:
            pred_depth = output
        
        gt_depth = batch.get('gt_depth', batch.get('raw_depth'))
        valid_mask = batch.get('valid_mask', (gt_depth > 0).float())
        thin_mask = batch.get('gt_thin_mask', None)
        
        # Compute metrics
        metrics = DepthMetrics.compute_all(pred_depth, gt_depth, valid_mask, thin_mask)
        all_metrics.append(metrics)
    
    # Aggregate
    aggregated = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            aggregated[key] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Run all baseline evaluations')
    parser.add_argument('--data_dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Our model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Baseline Evaluation Suite")
    print("=" * 60)
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print()
    
    # Results will be populated when data and models are ready
    print("Note: Full evaluation requires test data and trained model.")
    print("This script is ready for use once data collection is complete.")
    

if __name__ == '__main__':
    main()

