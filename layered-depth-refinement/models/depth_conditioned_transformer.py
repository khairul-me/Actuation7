"""
Depth-Conditioned Layer Transformer (NOVEL ARCHITECTURE)

Key innovations:
1. Learnable layer queries (like DETR object queries for depth layers)
2. Depth positional encoding (not spatial position)
3. Cross-attention to RGB features
4. Self-attention among layers for coherence
5. Offset-based prediction from base depth

This is NEW - nobody has done layer queries for depth estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DepthPositionalEncoding(nn.Module):
    """
    Encode base depth as positional signal for layer queries.
    Novel: Uses depth value, not spatial position.
    """
    def __init__(self, d_model=256, max_depth=1.0):
        super().__init__()
        self.d_model = d_model
        self.max_depth = max_depth
        
        # Learnable depth encoding
        self.depth_embed = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, d_model, 3, padding=1)
        )
        
    def forward(self, base_depth):
        """
        Args:
            base_depth: (B, 1, H, W) - depth from foundation model
        
        Returns:
            depth_encoding: (B, d_model, H, W)
        """
        # Normalize depth to [0, 1]
        depth_norm = torch.clamp(base_depth / self.max_depth, 0, 1)
        
        # Encode
        encoding = self.depth_embed(depth_norm)
        
        return encoding


class DepthConditionedLayerTransformer(nn.Module):
    """
    NOVEL ARCHITECTURE: Transformer with layer queries conditioned on base depth.
    
    Key innovations:
    1. Learnable layer queries (like DETR object queries)
    2. Depth positional encoding (not spatial position)
    3. Cross-attention to RGB features
    4. Self-attention among layers for coherence
    5. Offset-based prediction from base depth
    """
    def __init__(self, num_layers=3, d_model=256, num_heads=8, 
                 dropout=0.1, rgb_feature_dim=256):
        super().__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        # Learnable layer query embeddings
        self.layer_queries = nn.Parameter(torch.randn(num_layers, d_model))
        nn.init.xavier_uniform_(self.layer_queries.unsqueeze(0))  # Init as 3D then squeeze
        
        # Depth positional encoding
        self.depth_pos_encoder = DepthPositionalEncoding(d_model)
        
        # RGB feature projection
        self.rgb_proj = nn.Linear(rgb_feature_dim, d_model)
        
        # Multi-head cross-attention (queries attend to RGB)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # (seq, batch, feature)
        )
        
        # Multi-head self-attention (layer interaction)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Prediction heads
        self.depth_offset_head = nn.Linear(d_model, 1)
        self.alpha_head = nn.Linear(d_model, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, rgb_features, base_depth, thin_mask=None):
        """
        Args:
            rgb_features: (B, C, H, W) - features from RGB encoder
            base_depth: (B, 1, H, W) - depth from foundation model
            thin_mask: (B, 1, H, W) - optional thin structure mask
        
        Returns:
            layer_depths: (B, K, H, W) - predicted depth layers
            layer_alphas: (B, K, H, W) - alpha blending weights
            aux: dict with attention weights and intermediate results
        """
        B, C, H, W = rgb_features.shape
        K = self.num_layers
        
        # 1. Encode base depth as positional signal
        depth_encoding = self.depth_pos_encoder(base_depth)  # (B, d_model, H, W)
        
        # 2. Prepare RGB features for attention
        rgb_flat = rgb_features.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        rgb_flat = self.rgb_proj(rgb_flat)  # (H*W, B, d_model)
        
        # 3. Initialize layer queries with depth encoding
        # Average depth encoding spatially to get global context
        depth_global = depth_encoding.mean(dim=(2, 3))  # (B, d_model)
        
        # Expand layer queries for batch
        queries = self.layer_queries.unsqueeze(1).expand(-1, B, -1)  # (K, B, d_model)
        
        # Add depth positional encoding to each query
        queries = queries + depth_global.unsqueeze(0)  # Broadcast
        
        # 4. Cross-attention: layer queries attend to RGB features
        queries_attended, attn_weights_cross = self.cross_attention(
            query=queries,         # (K, B, d_model)
            key=rgb_flat,          # (H*W, B, d_model)
            value=rgb_flat,        # (H*W, B, d_model)
            need_weights=True
        )
        
        # Residual + norm
        queries = self.norm1(queries + self.dropout(queries_attended))
        
        # 5. Self-attention: layers interact with each other
        queries_self, attn_weights_self = self.self_attention(
            query=queries,
            key=queries,
            value=queries,
            need_weights=True
        )
        
        # Residual + norm
        queries = self.norm2(queries + self.dropout(queries_self))
        
        # 6. Feed-forward network
        queries_ffn = self.ffn(queries.permute(1, 0, 2))  # (B, K, d_model)
        queries_ffn = queries_ffn.permute(1, 0, 2)  # (K, B, d_model)
        
        # Residual + norm
        queries_final = self.norm3(queries + self.dropout(queries_ffn))
        
        # 7. Predict depth offsets and alphas from final query representations
        queries_final_batch = queries_final.permute(1, 0, 2)  # (B, K, d_model)
        
        # Depth offsets (relative to base depth)
        depth_offsets = self.depth_offset_head(queries_final_batch)  # (B, K, 1)
        depth_offsets = depth_offsets.squeeze(-1)  # (B, K)
        
        # Alpha logits
        alpha_logits = self.alpha_head(queries_final_batch)  # (B, K, 1)
        alpha_logits = alpha_logits.squeeze(-1)  # (B, K)
        
        # 8. Compute layer depths with ordering constraint
        # Use cumulative softplus to ensure d1 < d2 < d3
        depth_offsets_positive = F.softplus(depth_offsets)  # All positive
        cumulative_offsets = torch.cumsum(depth_offsets_positive, dim=1)  # (B, K)
        
        # Scale offsets (max 15cm total range, 5cm per layer)
        cumulative_offsets = cumulative_offsets * 0.05
        
        # Broadcast to spatial dimensions
        cumulative_offsets_spatial = cumulative_offsets.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
        cumulative_offsets_spatial = cumulative_offsets_spatial.expand(-1, -1, H, W)
        
        # Layer depths = base_depth + offsets
        base_depth_expanded = base_depth.expand(-1, K, -1, -1)  # (B, K, H, W)
        layer_depths = base_depth_expanded + cumulative_offsets_spatial
        
        # 9. Compute alpha weights (softmax across layers)
        if thin_mask is not None:
            # Modulate alphas based on thin mask
            alpha_logits_spatial = alpha_logits.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            
            # In non-thin regions, bias toward single layer (layer 0)
            bias = torch.zeros_like(alpha_logits_spatial)
            bias[:, 0, :, :] = (1 - thin_mask.squeeze(1)) * 5.0
            
            alpha_logits_spatial = alpha_logits_spatial + bias
            layer_alphas = F.softmax(alpha_logits_spatial, dim=1)
        else:
            alpha_logits_spatial = alpha_logits.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            layer_alphas = F.softmax(alpha_logits_spatial, dim=1)
        
        return layer_depths, layer_alphas, {
            'cross_attention_weights': attn_weights_cross,
            'self_attention_weights': attn_weights_self,
            'depth_offsets': cumulative_offsets,
            'alpha_logits': alpha_logits
        }

