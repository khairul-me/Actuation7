"""
Training Script for Layered Depth Refinement

Supports all 4 training stages:
1. Synthetic pretraining
2. Real data fine-tuning
3. Multi-view geometric refinement
4. Self-supervised on unlabeled data

Usage:
    python scripts/train.py --config configs/train_synthetic.yaml
    python scripts/train.py --config configs/finetune_real.yaml --pretrained checkpoints/synthetic_best.pth
"""

import os
import sys
import argparse
import yaml
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ThinStructureDepthRefinement
from losses import LayeredDepthLoss
from datasets.synthetic_dataset import SyntheticThinStructureDataset
from datasets.real_dataset import RealDualCameraDataset


def load_config(config_path):
    """Load YAML config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    """Create model from config."""
    model_cfg = config['model']
    model = ThinStructureDepthRefinement(
        use_foundation_fusion=model_cfg.get('use_foundation_fusion', True),
        use_multiview=model_cfg.get('use_multiview', False),
        num_layers=model_cfg.get('num_layers', 3),
        d_model=model_cfg.get('d_model', 256),
        use_dino=model_cfg.get('use_dino', True),
        use_depth_anything=model_cfg.get('use_depth_anything', True),
    )
    return model


def create_dataset(config, split):
    """Create dataset from config."""
    data_cfg = config['data']
    data_dir = data_cfg['data_dir']
    image_size = tuple(data_cfg.get('image_size', [480, 848]))
    max_depth = data_cfg.get('max_depth', 1.0)
    
    if 'synthetic' in data_dir:
        return SyntheticThinStructureDataset(
            data_dir=data_dir,
            split=split,
            image_size=image_size,
            max_depth=max_depth,
        )
    else:
        mode = data_cfg.get('mode', 'single')
        return RealDualCameraDataset(
            data_dir=data_dir,
            split=split,
            mode=mode,
            image_size=image_size,
            max_depth=max_depth,
        )


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    loss_components = {}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        rgb = batch['rgb'].to(device)
        
        # Optional inputs
        base_depth = batch.get('raw_depth')
        if base_depth is not None:
            base_depth = base_depth.to(device)
        
        rgb2 = batch.get('rgb2')
        if rgb2 is not None:
            rgb2 = rgb2.to(device)
        
        camera_params = batch.get('camera_params')
        if camera_params is not None:
            camera_params = {k: v.to(device) for k, v in camera_params.items()}
        
        # Forward pass
        output = model(
            rgb=rgb,
            base_depth=base_depth,
            rgb2=rgb2,
            camera_params=camera_params,
        )
        
        # Build target dict
        target = {
            'valid_mask': batch.get('valid_mask', torch.ones_like(output['composite_depth'])).to(device),
        }
        if 'gt_depth' in batch:
            target['gt_depth'] = batch['gt_depth'].to(device)
        else:
            target['gt_depth'] = batch.get('raw_depth', output['composite_depth'].detach()).to(device)
        
        if 'gt_thin_mask' in batch:
            target['gt_thin_mask'] = batch['gt_thin_mask'].to(device)
        
        # Compute loss
        loss, losses = criterion(output, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_clip = config['training'].get('grad_clip', 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Logging
        total_loss += loss.item()
        for k, v in losses.items():
            if k not in loss_components:
                loss_components[k] = 0
            loss_components[k] += v.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'composite': f'{losses["composite"].item():.4f}',
        })
    
    n = len(train_loader)
    avg_loss = total_loss / max(n, 1)
    avg_components = {k: v / max(n, 1) for k, v in loss_components.items()}
    
    return avg_loss, avg_components


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    for batch in val_loader:
        rgb = batch['rgb'].to(device)
        base_depth = batch.get('raw_depth')
        if base_depth is not None:
            base_depth = base_depth.to(device)
        
        output = model(rgb=rgb, base_depth=base_depth)
        
        target = {
            'valid_mask': batch.get('valid_mask', torch.ones_like(output['composite_depth'])).to(device),
        }
        if 'gt_depth' in batch:
            target['gt_depth'] = batch['gt_depth'].to(device)
        else:
            target['gt_depth'] = batch.get('raw_depth', output['composite_depth'].detach()).to(device)
        
        loss, _ = criterion(output, target)
        total_loss += loss.item()
    
    return total_loss / max(len(val_loader), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Layered Depth Refinement Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    
    # Create model
    model = create_model(config).to(device)
    
    # Load pretrained if specified
    pretrained = args.pretrained or config['training'].get('pretrained')
    if pretrained and os.path.exists(pretrained):
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded pretrained: {pretrained}")
    
    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
    
    # Create datasets
    train_dataset = create_dataset(config, 'train')
    val_dataset = create_dataset(config, 'val')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("\nNo training data found. Please prepare data first.")
        print("For synthetic: Run data_generation/blender_thin_structures.py")
        print("For real: Collect dual-camera frames")
        return
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
    
    # Loss and optimizer
    loss_cfg = config['loss']
    criterion = LayeredDepthLoss(
        alpha_composite=loss_cfg.get('alpha_composite', 1.0),
        alpha_order=loss_cfg.get('alpha_order', 0.1),
        alpha_alpha=loss_cfg.get('alpha_alpha', 0.5),
        alpha_thin=loss_cfg.get('alpha_thin', 0.3),
    )
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 1e-5),
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )
    
    # Training loop
    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss, train_components = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config['model'],
            }, 'checkpoints/best_model.pth')
            print(f"  Saved best model (val_loss={val_loss:.4f})")
        
        # Save periodic
        save_interval = config['logging'].get('save_interval', 5)
        if epoch % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, f'checkpoints/epoch_{epoch}.pth')
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()

