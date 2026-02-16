"""
Marigold Baseline

Diffusion-based monocular depth estimation.
Uses Hugging Face diffusers pipeline.

Reference: "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation" (CVPR 2024)
"""

import os
import torch
import torch.nn.functional as F
import numpy as np


class MarigoldBaseline:
    """
    Wrapper for Marigold as a baseline method.
    
    Uses the Hugging Face diffusers pipeline for inference.
    Model is downloaded automatically on first use.
    """
    def __init__(self, model_id="prs-eth/marigold-lcm-v1-0", device='cuda', 
                 dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.pipeline = None
    
    def load_model(self):
        """Lazy-load the Marigold pipeline."""
        if self.pipeline is not None:
            return
        
        try:
            from diffusers import MarigoldDepthPipeline
            
            print(f"[Marigold] Loading pipeline from {self.model_id}...")
            self.pipeline = MarigoldDepthPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                variant="fp16" if self.dtype == torch.float16 else None,
            ).to(self.device)
            
            # Enable memory optimizations
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass  # xformers not available, that's ok
            
            print("[Marigold] Pipeline loaded successfully")
            
        except ImportError as e:
            print(f"[Marigold] Import error: {e}")
            print("[Marigold] Install with: pip install diffusers transformers accelerate")
        except Exception as e:
            print(f"[Marigold] Error loading model: {e}")
    
    @torch.no_grad()
    def predict(self, rgb_image):
        """
        Predict depth from a single RGB image.
        
        Args:
            rgb_image: PIL Image or (H, W, 3) numpy array or (B, 3, H, W) tensor
        Returns:
            depth: (B, 1, H, W) tensor
        """
        self.load_model()
        if self.pipeline is None:
            return None
        
        from PIL import Image
        
        # Convert tensor to PIL Image(s)
        if isinstance(rgb_image, torch.Tensor):
            if rgb_image.dim() == 4:
                # Batch processing
                depths = []
                for i in range(rgb_image.shape[0]):
                    img = rgb_image[i].cpu()
                    # Denormalize if needed (assume ImageNet normalization)
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img = img * std + mean
                    img = img.clamp(0, 1)
                    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img)
                    
                    output = self.pipeline(
                        pil_img,
                        num_inference_steps=4,  # LCM uses few steps
                        ensemble_size=1,
                    )
                    depth = output.prediction[0]  # numpy array
                    depths.append(torch.from_numpy(depth).float())
                
                depths = torch.stack(depths).unsqueeze(1).to(self.device)
                return depths
            else:
                img = rgb_image.cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img * std + mean
                img = img.clamp(0, 1)
                img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                rgb_image = Image.fromarray(img)
        
        if isinstance(rgb_image, np.ndarray):
            rgb_image = Image.fromarray(rgb_image)
        
        output = self.pipeline(
            rgb_image,
            num_inference_steps=4,
            ensemble_size=1,
        )
        
        depth = output.prediction[0]
        depth = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        return depth
    
    def __call__(self, batch):
        """Process a batch."""
        rgb = batch['rgb']
        depth = self.predict(rgb)
        if depth is None:
            return None
        return {'depth': depth, 'composite_depth': depth}

