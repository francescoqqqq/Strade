"""
Centerline Dice (clDice) Loss for topology-preserving segmentation.

Implements Algorithm 2 from the clDice paper:
"clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"

The clDice loss measures how well the centerlines (skeletons) of predicted and
ground truth masks overlap, penalizing topological errors like disconnections.
"""

import torch
import torch.nn as nn
from typing import Optional

from .soft_skeleton import soft_skeletonize


class SoftClDiceLoss(nn.Module):
    """
    Soft Centerline Dice Loss.
    
    Computes the clDice metric between prediction and ground truth skeletons,
    ensuring topology preservation in tubular structures (e.g., roads, vessels).
    
    Formula (from paper):
        T_prec = sum(skel_pred ∩ mask_gt) / sum(skel_pred)
        T_sens = sum(skel_gt ∩ mask_pred) / sum(skel_gt)
        clDice = 2 * (T_prec * T_sens) / (T_prec + T_sens)
        Loss = 1 - clDice
    
    Args:
        num_iterations (int): Skeleton extraction iterations (default: 5)
        kernel_size (int): Morphological kernel size (default: 3)
        smooth (float): Epsilon for numerical stability (default: 1e-5)
        apply_softmax (bool): Whether to apply softmax to predictions (default: True)
                             Set to True if input are logits, False if probabilities
    """
    
    def __init__(
        self,
        num_iterations: int = 5,
        kernel_size: int = 3,
        smooth: float = 1e-5,
        apply_softmax: bool = True
    ):
        super(SoftClDiceLoss, self).__init__()
        self.num_iterations = num_iterations
        self.kernel_size = kernel_size
        self.smooth = smooth
        self.apply_softmax = apply_softmax
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        class_index: Optional[int] = 1
    ) -> torch.Tensor:
        """
        Compute clDice loss.
        
        Args:
            pred: Predicted logits or probabilities [B, C, H, W]
                  - If logits: apply_softmax should be True
                  - If probabilities: apply_softmax should be False
            target: Ground truth one-hot encoded [B, C, H, W] or class indices [B, H, W]
            class_index: Index of the foreground class to compute clDice on (default: 1)
                        Set to None to compute on all classes and average
        
        Returns:
            clDice loss value (scalar tensor)
        """
        # Apply softmax to predictions if needed
        if self.apply_softmax:
            pred = torch.softmax(pred, dim=1)
        
        # Handle target encoding
        if target.ndim == 3:
            # Class indices [B, H, W] → one-hot [B, C, H, W]
            target = torch.nn.functional.one_hot(
                target.long(), 
                num_classes=pred.shape[1]
            ).permute(0, 3, 1, 2).float()
        elif target.ndim == 4 and target.shape[1] == 1:
            # Class indices [B, 1, H, W] → one-hot [B, C, H, W]
            target = torch.nn.functional.one_hot(
                target.squeeze(1).long(),
                num_classes=pred.shape[1]
            ).permute(0, 3, 1, 2).float()
        
        # Extract foreground class (roads) only
        if class_index is not None:
            pred = pred[:, class_index:class_index+1, :, :]      # [B, 1, H, W]
            target = target[:, class_index:class_index+1, :, :]  # [B, 1, H, W]
        
        # Extract skeletons
        skel_pred = soft_skeletonize(
            pred, 
            num_iterations=self.num_iterations,
            kernel_size=self.kernel_size
        )
        skel_target = soft_skeletonize(
            target,
            num_iterations=self.num_iterations,
            kernel_size=self.kernel_size
        )
        
        # Topology Precision: how much of predicted skeleton is covered by GT mask
        # T_prec = |skel_pred ∩ mask_target| / |skel_pred|
        intersection_prec = torch.sum(skel_pred * target, dim=(2, 3))
        skel_pred_sum = torch.sum(skel_pred, dim=(2, 3))
        tprec = (intersection_prec + self.smooth) / (skel_pred_sum + self.smooth)
        
        # Topology Sensitivity: how much of GT skeleton is covered by prediction
        # T_sens = |skel_target ∩ mask_pred| / |skel_target|
        intersection_sens = torch.sum(skel_target * pred, dim=(2, 3))
        skel_target_sum = torch.sum(skel_target, dim=(2, 3))
        tsens = (intersection_sens + self.smooth) / (skel_target_sum + self.smooth)
        
        # clDice: harmonic mean of T_prec and T_sens
        # clDice = 2 * (T_prec * T_sens) / (T_prec + T_sens)
        cldice = (2.0 * tprec * tsens + self.smooth) / (tprec + tsens + self.smooth)
        
        # Average over batch and channels
        cldice = torch.mean(cldice)
        
        # Return loss (1 - clDice) for minimization
        return 1.0 - cldice


class CombinedClDiceLoss(nn.Module):
    """
    Combined loss: Dice + CE + clDice
    
    Formula (Equation 3 from paper):
        Loss = (1 - α) * (Dice + CE) + α * clDice
    
    Args:
        alpha (float): Weight for clDice loss (default: 0.4)
                      alpha=0.0 → standard Dice+CE only
                      alpha=1.0 → clDice only
        base_loss (nn.Module): Base loss function (e.g., DC_and_CE_loss from nnU-Net)
        cldice_params (dict): Parameters for SoftClDiceLoss
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        alpha: float = 0.4,
        cldice_iterations: int = 5,
        cldice_smooth: float = 1e-5,
        apply_softmax: bool = True
    ):
        super(CombinedClDiceLoss, self).__init__()
        self.alpha = alpha
        self.base_loss = base_loss
        self.cldice_loss = SoftClDiceLoss(
            num_iterations=cldice_iterations,
            kernel_size=3,
            smooth=cldice_smooth,
            apply_softmax=apply_softmax
        )
    
    def forward(self, pred, target) -> torch.Tensor:
        """
        Compute combined loss.
        
        Handles both single predictions and deep supervision (list of predictions).
        
        Args:
            pred: Either:
                  - Single tensor [B, C, H, W] if no deep supervision
                  - List/tuple of tensors [(B, C, H, W), ...] if deep supervision
            target: Either:
                    - Single tensor [B, 1, H, W] if no deep supervision  
                    - List/tuple of tensors [(B, 1, H_i, W_i), ...] if deep supervision
        
        Returns:
            Combined loss value (scalar)
        """
        # Handle deep supervision case (pred is a list/tuple)
        if isinstance(pred, (list, tuple)):
            # Calculate weights for each resolution (exponentially decreasing)
            import numpy as np
            weights = np.array([1 / (2 ** i) for i in range(len(pred))])
            weights[-1] = 0  # Don't use lowest resolution
            weights = weights / weights.sum()
            
            # Target can also be a list for deep supervision
            target_list = target if isinstance(target, (list, tuple)) else [target] * len(pred)
            
            total_loss = 0.0
            for i, (pred_i, target_i) in enumerate(zip(pred, target_list)):
                if weights[i] == 0:
                    continue
                # Compute base loss (Dice + CE) on this prediction
                loss_base_i = self.base_loss(pred_i, target_i)
                # Compute clDice loss (only on foreground class, use highest resolution target)
                loss_cldice_i = self.cldice_loss(pred_i, target_i, class_index=1)
                # Weighted combination for this resolution
                combined_i = (1.0 - self.alpha) * loss_base_i + self.alpha * loss_cldice_i
                # Add weighted loss
                total_loss += weights[i] * combined_i
            
            return total_loss
        else:
            # Single prediction case
            loss_base = self.base_loss(pred, target)
            loss_cldice = self.cldice_loss(pred, target, class_index=1)
            total_loss = (1.0 - self.alpha) * loss_base + self.alpha * loss_cldice
            return total_loss

