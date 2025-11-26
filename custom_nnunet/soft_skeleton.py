"""
Soft (differentiable) morphological skeletonization for PyTorch.

Implements Algorithm 1 from the clDice paper:
"clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"

This implementation uses differentiable operations (max/min pooling) instead of
traditional morphological operations to maintain gradient flow during backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonize(nn.Module):
    """
    Differentiable skeletonization using iterative morphological operations.
    
    The algorithm extracts the "skeleton" (centerline) of tubular structures
    by iteratively removing layers while preserving topology.
    
    Args:
        num_iterations (int): Number of morphological opening iterations (k in paper).
                              Higher k → thicker structures supported.
                              Default: 5 (good for road segmentation)
        kernel_size (int): Size of morphological kernels (default: 3)
    """
    
    def __init__(self, num_iterations: int = 5, kernel_size: int = 3):
        super(SoftSkeletonize, self).__init__()
        self.num_iterations = num_iterations
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
    
    def soft_erode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Soft erosion using min pooling.
        
        Min pooling is implemented as: -max_pool(-img)
        This is differentiable and approximates morphological erosion.
        
        Args:
            img: Input tensor [B, C, H, W]
        
        Returns:
            Eroded tensor [B, C, H, W]
        """
        # Min pooling = -MaxPool(-x)
        return -F.max_pool2d(
            -img, 
            kernel_size=self.kernel_size, 
            stride=1, 
            padding=self.padding
        )
    
    def soft_dilate(self, img: torch.Tensor) -> torch.Tensor:
        """
        Soft dilation using max pooling.
        
        Max pooling directly approximates morphological dilation.
        
        Args:
            img: Input tensor [B, C, H, W]
        
        Returns:
            Dilated tensor [B, C, H, W]
        """
        return F.max_pool2d(
            img, 
            kernel_size=self.kernel_size, 
            stride=1, 
            padding=self.padding
        )
    
    def soft_open(self, img: torch.Tensor) -> torch.Tensor:
        """
        Morphological opening: erosion followed by dilation.
        
        Opening removes small details while preserving larger structures.
        
        Args:
            img: Input tensor [B, C, H, W]
        
        Returns:
            Opened tensor [B, C, H, W]
        """
        return self.soft_dilate(self.soft_erode(img))
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Extract soft skeleton from input probability map.
        
        Algorithm (from paper):
        1. For k iterations:
           a. Compute morphological opening
           b. Subtract opening from current image → residual (fine details)
           c. Apply ReLU to clean negative values
           d. Accumulate residuals to build skeleton
        
        Args:
            img: Input probability map [B, C, H, W], values in [0, 1]
        
        Returns:
            Skeleton map [B, C, H, W], values in [0, 1]
        """
        # Initialize skeleton accumulator
        skeleton = torch.zeros_like(img)
        
        # Current image (will be progressively opened)
        current = img.clone()
        
        # Iteratively extract skeleton layers
        for _ in range(self.num_iterations):
            # Morphological opening
            opened = self.soft_open(current)
            
            # Extract residual (fine details removed by opening)
            residual = current - opened
            
            # Clean negative values (ReLU)
            residual = F.relu(residual)
            
            # Accumulate into skeleton
            skeleton = skeleton + residual
            
            # Update current image for next iteration
            current = opened
        
        # Clamp to [0, 1] range
        skeleton = torch.clamp(skeleton, 0, 1)
        
        return skeleton


def soft_skeletonize(img: torch.Tensor, num_iterations: int = 5, kernel_size: int = 3) -> torch.Tensor:
    """
    Functional interface for soft skeletonization.
    
    Args:
        img: Input probability map [B, C, H, W], values in [0, 1]
        num_iterations: Number of morphological iterations (default: 5)
        kernel_size: Size of morphological kernels (default: 3)
    
    Returns:
        Skeleton map [B, C, H, W], values in [0, 1]
    
    Example:
        >>> pred = torch.rand(2, 1, 256, 256)  # Batch of 2, single class
        >>> skel = soft_skeletonize(pred, num_iterations=5)
        >>> print(skel.shape)  # torch.Size([2, 1, 256, 256])
    """
    skeletonizer = SoftSkeletonize(num_iterations=num_iterations, kernel_size=kernel_size)
    return skeletonizer(img)

