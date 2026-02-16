"""
DUSt3R Baseline

Multi-view pointmap regression model (CVPR 2024).
Used as a multi-view stereo baseline.
"""

import os
import sys
import torch


class DUSt3RBaseline:
    """
    Wrapper for DUSt3R as a baseline method.
    
    Note: DUSt3R requires two views. For single-view evaluation,
    it will return None.
    """
    def __init__(self, weights_path=None, device='cuda'):
        self.device = device
        self.model = None
        self._weights_path = weights_path
        
        # Setup path to DUSt3R repo
        dust3r_repo = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'geometry-aware-depth-fusion', 'pretrained', 'dust3r_repo'
        )
        
        if os.path.exists(dust3r_repo):
            if dust3r_repo not in sys.path:
                sys.path.insert(0, dust3r_repo)
            
            croco_path = os.path.join(dust3r_repo, 'croco')
            if croco_path not in sys.path:
                sys.path.insert(0, croco_path)
    
    def load_model(self):
        """Lazy-load model (heavy, so only load when needed)."""
        if self.model is not None:
            return
        
        try:
            from dust3r.model import AsymmetricCroCo3DStereo
            from dust3r.inference import inference
            
            if self._weights_path and os.path.exists(self._weights_path):
                self.model = AsymmetricCroCo3DStereo.from_pretrained(self._weights_path)
            else:
                # Try default location
                default_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    'geometry-aware-depth-fusion', 'pretrained', 'dust3r', 'model.safetensors'
                )
                if os.path.exists(default_path):
                    self.model = AsymmetricCroCo3DStereo.from_pretrained(default_path)
            
            if self.model is not None:
                self.model = self.model.to(self.device).eval()
                print("[DUSt3R] Model loaded successfully")
            else:
                print("[DUSt3R] Warning: Could not load model weights")
                
        except ImportError as e:
            print(f"[DUSt3R] Import error: {e}")
            print("[DUSt3R] Make sure dust3r repo is in the pretrained directory")
    
    @torch.no_grad()
    def predict_dual_view(self, rgb1, rgb2):
        """
        Predict depth from two views.
        
        Args:
            rgb1, rgb2: (B, 3, H, W) tensors
        Returns:
            depth1, depth2: (B, 1, H, W) depth maps
        """
        self.load_model()
        if self.model is None:
            return None, None
        
        # DUSt3R expects specific input format
        # This is a simplified wrapper - full implementation needed
        # when running actual experiments
        raise NotImplementedError("Full DUSt3R inference pipeline to be implemented with data")
    
    def __call__(self, batch):
        """Process a batch (requires dual views)."""
        if 'rgb2' not in batch:
            return None
        
        rgb1 = batch['rgb'].to(self.device)
        rgb2 = batch['rgb2'].to(self.device)
        
        depth1, depth2 = self.predict_dual_view(rgb1, rgb2)
        return {'depth': depth1, 'composite_depth': depth1, 'depth2': depth2}

