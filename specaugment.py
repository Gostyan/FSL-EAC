
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FrequencyMask(nn.Module):
    def __init__(self, freq_mask_param: int = 27, num_masks: int = 1, 
                 replace_with_zero: bool = True):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks
        self.replace_with_zero = replace_with_zero
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        spec = spec.clone()
        num_freq = spec.shape[-2]
        
        for _ in range(self.num_masks):
            f = random.randint(0, min(self.freq_mask_param, num_freq - 1))
            if f == 0:
                continue
            f0 = random.randint(0, num_freq - f)
            
            if self.replace_with_zero:
                spec[..., f0:f0+f, :] = 0
            else:
                spec[..., f0:f0+f, :] = spec.mean()
        
        return spec


class TimeMask(nn.Module):
    def __init__(self, time_mask_param: int = 100, num_masks: int = 1,
                 replace_with_zero: bool = True):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks
        self.replace_with_zero = replace_with_zero
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        spec = spec.clone()
        num_time = spec.shape[-1]
        
        for _ in range(self.num_masks):
            t = random.randint(0, min(self.time_mask_param, num_time - 1))
            if t == 0:
                continue
            t0 = random.randint(0, num_time - t)
            
            if self.replace_with_zero:
                spec[..., t0:t0+t] = 0
            else:
                spec[..., t0:t0+t] = spec.mean()
        
        return spec


class TimeWarp(nn.Module):
    def __init__(self, warp_param: int = 80):
        super().__init__()
        self.warp_param = warp_param
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.dim() == 3:
            spec = spec.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        B, C, F, T = spec.shape
        
        if T <= 2 * self.warp_param:
            return spec.squeeze(0) if squeeze else spec
        
        center = random.randint(self.warp_param, T - self.warp_param)
        offset = random.randint(-self.warp_param, self.warp_param)
        
        if offset == 0:
            return spec.squeeze(0) if squeeze else spec
        
        src_points = torch.tensor([0, center, T - 1], dtype=torch.float32)
        dst_points = torch.tensor([0, center + offset, T - 1], dtype=torch.float32)
        
        x = torch.arange(T, dtype=torch.float32)
        warped_x = torch.zeros_like(x)
        
        mask1 = x <= center
        if mask1.any():
            scale1 = (dst_points[1] - dst_points[0]) / (src_points[1] - src_points[0] + 1e-8)
            warped_x[mask1] = (x[mask1] - src_points[0]) * scale1 + dst_points[0]
        
        mask2 = x > center
        if mask2.any():
            scale2 = (dst_points[2] - dst_points[1]) / (src_points[2] - src_points[1] + 1e-8)
            warped_x[mask2] = (x[mask2] - src_points[1]) * scale2 + dst_points[1]
        
        warped_x = 2 * warped_x / (T - 1) - 1
        
        grid_y = torch.linspace(-1, 1, F).view(1, F, 1).expand(B, F, T)
        grid_x = warped_x.view(1, 1, T).expand(B, F, T)
        grid = torch.stack([grid_x, grid_y], dim=-1).to(spec.device)
        
        warped = F.grid_sample(spec, grid, mode="bilinear", padding_mode="border", align_corners=True)
        
        return warped.squeeze(0) if squeeze else warped


class SpecAugment(nn.Module):
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        use_time_warp: bool = False,
        warp_param: int = 80,
        replace_with_zero: bool = True
    ):
        super().__init__()
        
        self.time_warp = TimeWarp(warp_param) if use_time_warp else None
        self.freq_mask = FrequencyMask(freq_mask_param, num_freq_masks, replace_with_zero)
        self.time_mask = TimeMask(time_mask_param, num_time_masks, replace_with_zero)
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.time_warp is not None:
            spec = self.time_warp(spec)
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec


def apply_spec_augment(
    spec: torch.Tensor,
    freq_mask_param: int = 27,
    time_mask_param: int = 100,
    num_freq_masks: int = 2,
    num_time_masks: int = 2
) -> torch.Tensor:
    augmenter = SpecAugment(
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        num_freq_masks=num_freq_masks,
        num_time_masks=num_time_masks
    )
    return augmenter(spec)


def get_specaugment_light() -> SpecAugment:
    return SpecAugment(
        freq_mask_param=15,
        time_mask_param=50,
        num_freq_masks=1,
        num_time_masks=1,
        use_time_warp=False
    )


def get_specaugment_standard() -> SpecAugment:
    return SpecAugment(
        freq_mask_param=27,
        time_mask_param=100,
        num_freq_masks=2,
        num_time_masks=2,
        use_time_warp=False
    )


def get_specaugment_strong() -> SpecAugment:
    return SpecAugment(
        freq_mask_param=27,
        time_mask_param=100,
        num_freq_masks=2,
        num_time_masks=2,
        use_time_warp=True,
        warp_param=80
    )


if __name__ == "__main__":
    print("=" * 60)
    print("SpecAugment Module Test")
    print("=" * 60)
    
    spec = torch.randn(2, 1, 80, 1000)
    print(f"\nInput shape: {spec.shape}")
    
    print("\n1. Test FrequencyMask:")
    freq_mask = FrequencyMask(freq_mask_param=27, num_masks=2)
    masked = freq_mask(spec)
    print(f"   Output shape: {masked.shape}")
    
    print("\n2. Test TimeMask:")
    time_mask = TimeMask(time_mask_param=100, num_masks=2)
    masked = time_mask(spec)
    print(f"   Output shape: {masked.shape}")
    
    print("\n3. Test SpecAugment:")
    augmenter = get_specaugment_standard()
    augmented = augmenter(spec)
    print(f"   Output shape: {augmented.shape}")
    
    diff = (spec - augmented).abs().mean().item()
    print(f"   Mean change: {diff:.4f}")
    
    print("\nSpecAugment test passed!")
