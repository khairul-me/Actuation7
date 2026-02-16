"""
Evaluation Script

Evaluates trained model on test set with comprehensive metrics.

Usage:
    python scripts/evaluate.py --config configs/evaluate.yaml --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import argparse
import yaml
import json
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ThinStructureDepthRefinement
from evaluation.metrics import DepthMetrics
from datasets.real_dataset import RealDualCameraDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/evaluate.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = ThinStructureDepthRefinement.from_pretrained(args.checkpoint)
    model = model.to(device).eval()
    
    print(f"Model loaded from {args.checkpoint}")
    print(f"Evaluating on {device}")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create test dataset
    test_dataset = RealDualCameraDataset(
        data_dir=config['evaluation']['data_dir'],
        split='test',
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Test samples: {len(test_dataset)}")
    
    if len(test_dataset) == 0:
        print("No test data found.")
        return
    
    # Evaluate
    all_metrics = []
    
    for batch in test_loader:
        rgb = batch['rgb'].to(device)
        base_depth = batch.get('raw_depth')
        if base_depth is not None:
            base_depth = base_depth.to(device)
        
        with torch.no_grad():
            output = model(rgb=rgb, base_depth=base_depth)
        
        pred = output['composite_depth']
        gt = batch.get('gt_depth', batch['raw_depth']).to(device)
        valid = batch.get('valid_mask', (gt > 0).float()).to(device)
        thin_mask = batch.get('gt_thin_mask')
        if thin_mask is not None:
            thin_mask = thin_mask.to(device)
        
        metrics = DepthMetrics.compute_all(pred, gt, valid, thin_mask)
        all_metrics.append(metrics)
    
    # Aggregate
    import numpy as np
    results = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        results[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
        }
    
    # Save
    output_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for key, val in results.items():
        print(f"  {key}: {val['mean']:.4f} +/- {val['std']:.4f}")
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()

