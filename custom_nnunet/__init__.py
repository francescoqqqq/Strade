"""
Custom nnU-Net components for topology-preserving segmentation.

Implements clDice Loss as described in:
"clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"

Components:
- soft_skeleton: Differentiable morphological skeletonization
- cldice_loss: Centerline Dice loss module
- trainer_cldice: Custom nnU-Net trainer with combined loss
"""

from .soft_skeleton import soft_skeletonize
from .cldice_loss import SoftClDiceLoss
from .trainer_cldice import nnUNetTrainerClDice

__all__ = [
    'soft_skeletonize',
    'SoftClDiceLoss',
    'nnUNetTrainerClDice',
]

__version__ = '1.0.0'

