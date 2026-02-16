"""
ThinStructureDepthRefinement - Complete Enhancement Module

Plug-and-play module that enhances ANY base depth estimator for thin structures.

Works with:
- Depth Anything V2 (monocular foundation model)
- DUSt3R (multi-view pointmap regression)
- Marigold (diffusion-based depth)
- Raw stereo depth
- Any other depth source

Key features:
1. Plug-and-play (frozen foundation models)
2. Layered depth output for thin structures
3. Optional multi-view geometric refinement
4. Adaptive layer collapse (thin vs. thick regions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth_conditioned_transformer import DepthConditionedLayerTransformer
from .thin_structure_detector import ThinStructureDetector
from .foundation_fusion import DINODepthFeatureFusion, SimpleFeatureEncoder
from .geometric_refinement import GeometricConsistencyRefinement, GeometricUncertaintyPredictor


class ThinStructureDepthRefinement(nn.Module):
    """
    Complete module that enhances ANY base depth estimator.
    """
    def __init__(self, 
                 use_foundation_fusion=True,
                 use_multiview=False,
                 num_layers=3,
                 d_model=256,
                 use_dino=True,
                 use_depth_anything=True,
                 depth_anything_weights=None):
        super().__init__()
        
        self.use_foundation_fusion = use_foundation_fusion
        self.use_multiview = use_multiview
        self.num_layers = num_layers
        self.d_model = d_model
        
        # Foundation model feature extraction (optional)
        if use_foundation_fusion:
            self.foundation_fusion = DINODepthFeatureFusion(
                use_dino=use_dino,
                use_depth_anything=use_depth_anything,
                output_dim=d_model,
                depth_anything_weights=depth_anything_weights,
            )
        else:
            # Use custom encoder (ConvNeXt-Tiny)
            self.feature_encoder = SimpleFeatureEncoder(output_dim=d_model)
        
        # Thin structure detection
        self.thin_detector = ThinStructureDetector(feature_dim=d_model)
        
        # Core layer transformer (NOVEL)
        self.layer_transformer = DepthConditionedLayerTransformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=8,
            dropout=0.1,
            rgb_feature_dim=d_model
        )
        
        # Multi-view components (optional)
        if use_multiview:
            self.geometric_refiner = GeometricConsistencyRefinement(max_correction=0.05)
            self.uncertainty_predictor = GeometricUncertaintyPredictor()
        
        # Adaptive layer collapse gate
        self.collapse_gate = nn.Sequential(
            nn.Conv2d(num_layers + 1, 64, 3, padding=1),  # layers + thin_mask
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()  # Gate value [0, 1]
        )
        
    @classmethod
    def from_pretrained(cls, checkpoint_path, **kwargs):
        """Load a pretrained model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get model config from checkpoint
        config = checkpoint.get('config', {})
        model = cls(**{**config, **kwargs})
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
        
    def forward(self, 
                rgb,
                base_depth=None,
                rgb2=None,
                base_depth2=None,
                camera_params=None):
        """
        Args:
            rgb: (B, 3, H, W) - primary view RGB
            base_depth: (B, 1, H, W) - optional base depth estimate.
                       If None and using foundation fusion, will use Depth Anything V2.
            rgb2: (B, 3, H, W) - optional second view (for multi-view)
            base_depth2: (B, 1, H, W) - optional second view depth
            camera_params: dict with K1, K2, R, T for multi-view
        
        Returns:
            output: dict with:
                - layer_depths: (B, K, H, W)
                - layer_alphas: (B, K, H, W)
                - thin_mask: (B, 1, H, W)
                - base_depth: (B, 1, H, W)
                - uncertainty: (B, 1, H, W) if multi-view
                - composite_depth: (B, 1, H, W)
                - collapse_gate: (B, 1, H, W)
                - transformer_aux: dict with attention weights
        """
        B, _, H, W = rgb.shape
        
        # Step 1: Extract features and base depth
        if self.use_foundation_fusion:
            result = self.foundation_fusion(rgb, return_base_depth=True)
            if isinstance(result, tuple):
                features, base_depth_pred = result
            else:
                features = result
                base_depth_pred = None
            
            if base_depth is None:
                base_depth = base_depth_pred
        else:
            if base_depth is None:
                raise ValueError("base_depth required when not using foundation fusion")
            features = self.feature_encoder(rgb, base_depth)
            # Upsample features to input resolution
            if features.shape[2:] != (H, W):
                features = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)
        
        # Step 2: Detect thin structures
        thin_mask = self.thin_detector(rgb, base_depth, features)
        
        # Step 3: Predict layered depth using transformer
        layer_depths, layer_alphas, transformer_aux = self.layer_transformer(
            rgb_features=features,
            base_depth=base_depth,
            thin_mask=thin_mask
        )
        
        # Step 4: Multi-view refinement (if available)
        uncertainty = None
        if self.use_multiview and rgb2 is not None and camera_params is not None:
            # Extract features for second view
            if self.use_foundation_fusion:
                result2 = self.foundation_fusion(rgb2, return_base_depth=True)
                if isinstance(result2, tuple):
                    features2, base_depth2_pred = result2
                else:
                    features2 = result2
                    base_depth2_pred = None
                
                if base_depth2 is None:
                    base_depth2 = base_depth2_pred
            else:
                features2 = self.feature_encoder(rgb2, base_depth2)
                if features2.shape[2:] != (H, W):
                    features2 = F.interpolate(features2, size=(H, W), mode='bilinear', align_corners=False)
            
            # Detect thin structures in view 2
            thin_mask2 = self.thin_detector(rgb2, base_depth2, features2)
            
            # Predict layers for view 2
            layer_depths2, layer_alphas2, _ = self.layer_transformer(
                rgb_features=features2,
                base_depth=base_depth2,
                thin_mask=thin_mask2
            )
            
            # Geometric consistency refinement
            layer_depths, consistency_map = self.geometric_refiner(
                layer_depths, layer_depths2,
                camera_params['K1'], camera_params['K2'],
                camera_params['R'], camera_params['T']
            )
            
            # Uncertainty estimation
            uncertainty = self.uncertainty_predictor(
                layer_depths, layer_depths2,
                camera_params['K1'], camera_params['K2'],
                camera_params['R'], camera_params['T'],
                consistency_error=consistency_map
            )
        
        # Step 5: Adaptive layer collapse
        collapse_input = torch.cat([layer_depths, thin_mask], dim=1)
        collapse_gate = self.collapse_gate(collapse_input)  # (B, 1, H, W)
        
        # Compute composite depth
        composite_depth = (layer_depths * layer_alphas).sum(dim=1, keepdim=True)
        
        return {
            'layer_depths': layer_depths,
            'layer_alphas': layer_alphas,
            'thin_mask': thin_mask,
            'base_depth': base_depth,
            'composite_depth': composite_depth,
            'uncertainty': uncertainty,
            'collapse_gate': collapse_gate,
            'transformer_aux': transformer_aux
        }

