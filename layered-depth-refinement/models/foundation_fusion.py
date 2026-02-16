"""
DINOv2 + Depth Anything V2 Feature Fusion Module

Combines DINOv2 semantic features with Depth Anything V2 geometric features.
Novel: Nobody has fused these two SOTA foundation models for layered depth.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINODepthFeatureFusion(nn.Module):
    """
    Combines DINOv2 semantic features with Depth Anything V2 geometric features.
    
    Novel: Nobody has fused these two SOTA foundation models for layered depth.
    """
    def __init__(self, use_dino=True, use_depth_anything=True,
                 dino_model_name='dinov2_vitb14',
                 depth_anything_weights=None,
                 output_dim=256):
        super().__init__()
        
        self.use_dino = use_dino
        self.use_depth_anything = use_depth_anything
        self.output_dim = output_dim
        
        # Load frozen foundation models
        if use_dino:
            self.dino = torch.hub.load('facebookresearch/dinov2', dino_model_name)
            self.dino.eval()
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino_dim = 768  # ViT-B/14 output dimension
        
        if use_depth_anything:
            # Load Depth Anything V2 ViT-L
            da_path = depth_anything_weights
            if da_path is None:
                # Try default path
                import os
                da_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    '..', 'Depth-Anything-V2', 'checkpoints', 'depth_anything_v2_vitl.pth'
                )
            
            # Add DA-V2 to path
            da_repo_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                '..', 'Depth-Anything-V2'
            )
            if da_repo_dir not in sys.path:
                sys.path.insert(0, da_repo_dir)
            
            from depth_anything_v2.dpt import DepthAnythingV2 as DAv2Model
            
            model_configs = {
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }
            self.depth_anything = DAv2Model(**model_configs['vitl'])
            if os.path.exists(da_path):
                self.depth_anything.load_state_dict(torch.load(da_path, map_location='cpu'))
            
            self.depth_anything.eval()
            for param in self.depth_anything.parameters():
                param.requires_grad = False
            
            self.da_feature_dim = 256  # DPT head output dim
        
        # Learnable fusion modules (these are what you train)
        if use_dino and use_depth_anything:
            # Fuse DINO (768) + DA features (256) -> output_dim
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(self.dino_dim + self.da_feature_dim, output_dim * 2, 1),
                nn.GroupNorm(16, output_dim * 2),
                nn.ReLU(),
                nn.Conv2d(output_dim * 2, output_dim, 3, padding=1),
                nn.GroupNorm(16, output_dim),
                nn.ReLU(),
            )
        elif use_dino:
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(self.dino_dim, output_dim, 1),
                nn.GroupNorm(16, output_dim),
                nn.ReLU(),
            )
        elif use_depth_anything:
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(self.da_feature_dim, output_dim, 1),
                nn.GroupNorm(16, output_dim),
                nn.ReLU(),
            )
        
    def extract_dino_features(self, rgb):
        """Extract multi-scale features from DINOv2."""
        B = rgb.shape[0]
        
        with torch.no_grad():
            # Resize to DINOv2 expected size (multiple of 14)
            rgb_resized = F.interpolate(rgb, size=(518, 518), mode='bilinear', align_corners=False)
            
            # Get last 4 intermediate layer features
            # n=int returns last n layers; n=list would be specific indices
            dino_features = self.dino.get_intermediate_layers(
                rgb_resized, 
                n=4,  # Return last 4 layers
                return_class_token=False
            )
            
            # Reshape to spatial format
            # N = (518/14) * (518/14) = 37 * 37 = 1369
            dino_feat = dino_features[-1]  # Last layer: (B, 1369, 768)
            dino_feat = dino_feat.transpose(1, 2).reshape(B, self.dino_dim, 37, 37)
        
        return dino_feat
    
    def extract_da_features(self, rgb):
        """Extract features and depth from Depth Anything V2."""
        with torch.no_grad():
            # Depth Anything V2 expects (B, 3, H, W) normalized
            depth = self.depth_anything.infer_image(rgb)
            
            # Also extract intermediate features from the encoder
            # We hook into the pretrained model's forward
            features = self.depth_anything.pretrained.get_intermediate_layers(
                rgb, n=[9, 12], return_class_token=False
            )
            
            # Use last layer features
            da_feat = features[-1]  # (B, N, C)
            B = rgb.shape[0]
            h = w = int(da_feat.shape[1] ** 0.5)
            da_feat = da_feat.transpose(1, 2).reshape(B, -1, h, w)
        
        return da_feat, depth
    
    def forward(self, rgb, return_base_depth=True):
        """
        Args:
            rgb: (B, 3, H, W) normalized to [0, 1]
            return_base_depth: if True, return base depth from Depth Anything
        
        Returns:
            fused_features: (B, output_dim, H, W)
            base_depth: (B, 1, H, W) if return_base_depth=True
        """
        B, _, H, W = rgb.shape
        
        features_to_fuse = []
        base_depth = None
        
        # Extract DINOv2 features
        if self.use_dino:
            dino_features = self.extract_dino_features(rgb)
            # Resize to target resolution
            dino_features = F.interpolate(dino_features, size=(H, W), mode='bilinear', align_corners=False)
            features_to_fuse.append(dino_features)
        
        # Extract Depth Anything V2 features + base depth
        if self.use_depth_anything:
            da_features, depth_pred = self.extract_da_features(rgb)
            da_features = F.interpolate(da_features, size=(H, W), mode='bilinear', align_corners=False)
            
            # Truncate or pad to expected dim
            if da_features.shape[1] != self.da_feature_dim:
                da_features = F.adaptive_avg_pool1d(
                    da_features.flatten(2), self.da_feature_dim
                ).reshape(B, self.da_feature_dim, H, W)
            
            features_to_fuse.append(da_features)
            
            if isinstance(depth_pred, torch.Tensor):
                if depth_pred.dim() == 2:
                    base_depth = depth_pred.unsqueeze(0).unsqueeze(0)
                elif depth_pred.dim() == 3:
                    base_depth = depth_pred.unsqueeze(1)
                else:
                    base_depth = depth_pred
                base_depth = F.interpolate(base_depth.float(), size=(H, W), mode='bilinear', align_corners=False)
        
        # Fuse features
        if len(features_to_fuse) > 1:
            combined = torch.cat(features_to_fuse, dim=1)
        elif len(features_to_fuse) == 1:
            combined = features_to_fuse[0]
        else:
            raise ValueError("No features to fuse - enable at least one backbone")
        
        fused_features = self.fusion_conv(combined)
        
        if return_base_depth and base_depth is not None:
            return fused_features, base_depth
        else:
            return fused_features


class SimpleFeatureEncoder(nn.Module):
    """
    Fallback encoder when not using foundation models.
    Uses ConvNeXt-Tiny for feature extraction from RGB + base depth.
    """
    def __init__(self, output_dim=256, input_channels=4):
        super().__init__()
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        
        # Modify first conv to accept 4 channels (RGB + depth)
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            input_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )
        # Copy RGB weights, init depth channel from mean
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
        backbone.features[0][0] = new_conv
        
        self.backbone = backbone.features
        
        # Project to output dim
        self.proj = nn.Conv2d(768, output_dim, 1)
    
    def forward(self, rgb, base_depth):
        """
        Args:
            rgb: (B, 3, H, W)
            base_depth: (B, 1, H, W)
        Returns:
            features: (B, output_dim, H//32, W//32)
        """
        x = torch.cat([rgb, base_depth], dim=1)
        features = self.backbone(x)
        features = self.proj(features)
        return features

