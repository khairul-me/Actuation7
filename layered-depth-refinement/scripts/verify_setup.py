"""
Setup Verification Script

Checks all prerequisites for the Layered Depth Refinement project:
1. Python environment & GPU
2. Required packages
3. Pretrained model weights
4. Project structure
5. Core component instantiation
6. Forward pass tests

Usage:
    python scripts/verify_setup.py
"""

import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
WEED_ROOT = os.path.dirname(PROJECT_ROOT)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    symbol = "[+]" if condition else "[-]"
    print(f"  {symbol} {name}: {status}", end="")
    if detail:
        print(f"  ({detail})", end="")
    print()
    return condition


def main():
    results = {}
    
    print("\n" + "#" * 60)
    print("  Layered Depth Refinement - Setup Verification")
    print("#" * 60)
    
    # ─── 1. Environment ─────────────────────────────────────────
    section("1. Python Environment & GPU")
    
    import platform
    results['python'] = check("Python 3.10+", 
                               sys.version_info >= (3, 10),
                               f"Python {platform.python_version()}")
    
    try:
        import torch
        results['pytorch'] = check("PyTorch 2.0+", 
                                    int(torch.__version__.split('.')[0]) >= 2,
                                    f"v{torch.__version__}")
        
        results['cuda'] = check("CUDA available",
                                 torch.cuda.is_available(),
                                 torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
        
        if torch.cuda.is_available():
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            results['vram'] = check("GPU VRAM >= 12GB", mem_gb >= 12, f"{mem_gb:.1f} GB")
    except ImportError:
        results['pytorch'] = check("PyTorch", False, "NOT INSTALLED")
        results['cuda'] = False
    
    # ─── 2. Required Packages ───────────────────────────────────
    section("2. Required Packages")
    
    packages = {
        'torchvision': 'torchvision',
        'opencv': 'cv2',
        'open3d': 'open3d',
        'kornia': 'kornia',
        'timm': 'timm',
        'einops': 'einops',
        'scipy': 'scipy',
        'wandb': 'wandb',
        'omegaconf': 'omegaconf',
        'diffusers': 'diffusers',
        'transformers': 'transformers',
        'accelerate': 'accelerate',
        'safetensors': 'safetensors',
        'huggingface_hub': 'huggingface_hub',
        'PIL': 'PIL',
        'imageio': 'imageio',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'yaml': 'yaml',
        'geffnet': 'geffnet',
    }
    
    for name, module in packages.items():
        try:
            m = __import__(module)
            ver = getattr(m, '__version__', 'ok')
            results[f'pkg_{name}'] = check(name, True, f"v{ver}")
        except ImportError:
            results[f'pkg_{name}'] = check(name, False, "NOT INSTALLED")
    
    # ─── 3. Pretrained Models ───────────────────────────────────
    section("3. Pretrained Model Weights")
    
    # Depth Anything V2
    da_path = os.path.join(WEED_ROOT, 'Depth-Anything-V2', 'checkpoints', 'depth_anything_v2_vitl.pth')
    da_size = os.path.getsize(da_path) / (1024**2) if os.path.exists(da_path) else 0
    results['da_v2'] = check("Depth Anything V2 ViT-L", 
                              os.path.exists(da_path) and da_size > 100,
                              f"{da_size:.0f} MB" if da_size > 0 else "NOT FOUND")
    
    # DINOv2
    dino_cache = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints', 'dinov2_vitb14_pretrain.pth')
    dino_size = os.path.getsize(dino_cache) / (1024**2) if os.path.exists(dino_cache) else 0
    results['dinov2'] = check("DINOv2 ViT-B/14",
                               os.path.exists(dino_cache) and dino_size > 100,
                               f"{dino_size:.0f} MB" if dino_size > 0 else "NOT FOUND")
    
    # DSINE
    dsine_paths = [
        os.path.join(WEED_ROOT, 'geometry-aware-depth-fusion', 'pretrained', 'dsine.pt'),
        os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints', 'dsine.pt'),
    ]
    dsine_found = any(os.path.exists(p) for p in dsine_paths)
    results['dsine'] = check("DSINE surface normals", dsine_found)
    
    # DUSt3R
    dust3r_path = os.path.join(WEED_ROOT, 'geometry-aware-depth-fusion', 'pretrained', 'dust3r', 'model.safetensors')
    dust3r_size = os.path.getsize(dust3r_path) / (1024**2) if os.path.exists(dust3r_path) else 0
    results['dust3r'] = check("DUSt3R weights",
                                os.path.exists(dust3r_path) and dust3r_size > 100,
                                f"{dust3r_size:.0f} MB" if dust3r_size > 0 else "NOT FOUND")
    
    # Marigold (cached by HuggingFace)
    hf_cache = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
    marigold_dirs = [d for d in os.listdir(hf_cache) if 'marigold' in d.lower()] if os.path.exists(hf_cache) else []
    results['marigold'] = check("Marigold LCM (cached)", len(marigold_dirs) > 0,
                                  f"{len(marigold_dirs)} cache dir(s)")
    
    # ─── 4. Project Structure ───────────────────────────────────
    section("4. Project Structure")
    
    expected_dirs = [
        'models', 'losses', 'datasets', 'evaluation', 'baselines',
        'scripts', 'configs', 'utils', 'data_collection', 'data_generation',
        'checkpoints', 'pretrained', 'results', 'data'
    ]
    
    for d in expected_dirs:
        full_path = os.path.join(PROJECT_ROOT, d)
        results[f'dir_{d}'] = check(f"Directory: {d}/", os.path.isdir(full_path))
    
    expected_files = [
        'models/depth_conditioned_transformer.py',
        'models/thin_structure_detector.py',
        'models/foundation_fusion.py',
        'models/foundation_enhancement_module.py',
        'models/geometric_refinement.py',
        'losses/layered_depth_loss.py',
        'datasets/synthetic_dataset.py',
        'datasets/real_dataset.py',
        'baselines/depth_anything_v2.py',
        'baselines/dust3r.py',
        'baselines/marigold.py',
        'baselines/variance_weighted.py',
        'evaluation/metrics.py',
        'evaluation/run_all_baselines.py',
        'scripts/train.py',
        'scripts/evaluate.py',
        'configs/train_synthetic.yaml',
        'configs/finetune_real.yaml',
        'configs/train_multiview.yaml',
        'configs/evaluate.yaml',
        'data_collection/calibrate_cameras.py',
        'data_collection/capture_dual_camera.py',
        'data_generation/blender_thin_structures.py',
        'requirements.txt',
    ]
    
    all_files_ok = True
    for f in expected_files:
        full_path = os.path.join(PROJECT_ROOT, f)
        exists = os.path.isfile(full_path)
        if not exists:
            all_files_ok = False
            check(f"File: {f}", False)
    
    results['all_files'] = check(f"All {len(expected_files)} source files present", all_files_ok)
    
    # ─── 5. Core Component Tests ────────────────────────────────
    section("5. Core Component Instantiation")
    
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Layer Transformer
        from models.depth_conditioned_transformer import DepthConditionedLayerTransformer
        layer_transformer = DepthConditionedLayerTransformer(
            num_layers=3, d_model=256, num_heads=8
        ).to(device)
        params = sum(p.numel() for p in layer_transformer.parameters())
        results['layer_transformer'] = check("DepthConditionedLayerTransformer", True, f"{params:,} params")
        
        # Thin Structure Detector
        from models.thin_structure_detector import ThinStructureDetector
        thin_detector = ThinStructureDetector(feature_dim=256).to(device)
        params = sum(p.numel() for p in thin_detector.parameters())
        results['thin_detector'] = check("ThinStructureDetector", True, f"{params:,} params")
        
        # Loss
        from losses.layered_depth_loss import LayeredDepthLoss
        loss_fn = LayeredDepthLoss()
        results['loss'] = check("LayeredDepthLoss", True)
        
        # Metrics
        from evaluation.metrics import DepthMetrics
        results['metrics'] = check("DepthMetrics", True)
        
    except Exception as e:
        results['components'] = check("Component instantiation", False, str(e))
    
    # ─── 6. Forward Pass Test ───────────────────────────────────
    section("6. Forward Pass Tests (GPU)")
    
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test Layer Transformer forward pass
        B, C, H, W = 1, 256, 30, 40  # Small resolution for speed
        rgb_features = torch.randn(B, C, H, W, device=device)
        base_depth = torch.rand(B, 1, H, W, device=device) * 0.5 + 0.1
        
        start = time.time()
        layer_depths, layer_alphas, aux = layer_transformer(rgb_features, base_depth)
        elapsed = (time.time() - start) * 1000
        
        results['fwd_transformer'] = check(
            "LayerTransformer forward", 
            layer_depths.shape == (B, 3, H, W) and layer_alphas.shape == (B, 3, H, W),
            f"output: ({B},{3},{H},{W}), {elapsed:.0f}ms"
        )
        
        # Check ordering constraint
        ordered = (layer_depths[:, 1:] >= layer_depths[:, :-1]).all().item()
        results['ordering'] = check("Layer ordering (d1 < d2 < d3)", ordered)
        
        # Check alpha normalization
        alpha_sum = layer_alphas.sum(dim=1)
        alpha_ok = torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5)
        results['alpha_sum'] = check("Alpha weights sum to 1", alpha_ok)
        
        # Test thin detector
        rgb = torch.randn(B, 3, H, W, device=device)
        thin_mask = thin_detector(rgb, base_depth, rgb_features)
        results['fwd_thin'] = check(
            "ThinStructureDetector forward",
            thin_mask.shape == (B, 1, H, W),
            f"output: {thin_mask.shape}"
        )
        
        # Test loss computation
        pred = {
            'layer_depths': layer_depths,
            'layer_alphas': layer_alphas,
            'thin_mask': thin_mask,
            'composite_depth': (layer_depths * layer_alphas).sum(dim=1, keepdim=True),
        }
        target = {
            'gt_depth': base_depth,
            'valid_mask': torch.ones(B, 1, H, W, device=device),
        }
        total_loss, losses = loss_fn(pred, target)
        results['fwd_loss'] = check(
            "Loss computation",
            total_loss.requires_grad and not torch.isnan(total_loss),
            f"loss={total_loss.item():.4f}"
        )
        
        # Test metrics
        metrics = DepthMetrics.compute_all(
            pred['composite_depth'],
            target['gt_depth'],
            target['valid_mask']
        )
        results['fwd_metrics'] = check(
            "Metrics computation",
            'all_mae' in metrics and 'all_rmse' in metrics,
            f"MAE={metrics['all_mae']:.4f}, RMSE={metrics['all_rmse']:.4f}"
        )
        
    except Exception as e:
        import traceback
        print(f"  [-] Forward pass test FAILED: {e}")
        traceback.print_exc()
        results['fwd_pass'] = False
    
    # ─── Summary ────────────────────────────────────────────────
    section("SUMMARY")
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    total = passed + failed
    
    print(f"\n  Results: {passed}/{total} checks passed")
    
    if failed > 0:
        print(f"\n  Failed checks ({failed}):")
        for k, v in results.items():
            if v is False:
                print(f"    - {k}")
    
    if failed == 0:
        print("\n  ALL CHECKS PASSED!")
        print("  The project is fully set up and ready for development.")
        print("\n  Next steps:")
        print("    1. Generate synthetic data: blender --background --python data_generation/blender_thin_structures.py")
        print("    2. Or wait for real dual-camera videos")
        print("    3. Then: python scripts/train.py --config configs/train_synthetic.yaml")
    else:
        print(f"\n  {failed} checks failed. Please fix the issues above.")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

