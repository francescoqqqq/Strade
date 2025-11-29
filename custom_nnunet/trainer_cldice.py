"""
Custom nnU-Net Trainer with clDice Loss.

Extends the standard nnUNetTrainer to incorporate topology-preserving clDice loss
for tubular structure segmentation (e.g., roads, blood vessels).

Usage:
    nnUNetv2_train DATASET_ID 2d FOLD --trainer nnUNetTrainerClDice
"""

from typing import Union, Tuple
import torch
import numpy as np

# Configura torch.compile per fare fallback automatico se fallisce
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except:
    pass

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

from .cldice_loss import CombinedClDiceLoss


class nnUNetTrainerClDice(nnUNetTrainer):
    """
    nnU-Net Trainer with combined Dice + CE + clDice loss.
    
    This trainer uses a weighted combination of:
    - Standard nnU-Net loss (Dice + Cross Entropy)
    - clDice loss (for topology preservation)
    
    Formula: Loss = (1 - α) * (Dice + CE) + α * clDice
    
    The alpha parameter controls the trade-off:
    - alpha=0.0: Standard nnU-Net (no topology constraint)
    - alpha=0.4: Balanced (default, recommended for roads)
    - alpha=1.0: Pure clDice (only topology, no region overlap)
    
    Configuration:
        You can override these in __init__ or by subclassing:
        - self.cldice_alpha: Weight for clDice (default: 0.4)
        - self.cldice_iterations: Skeleton iterations (default: 5)
        - self.cldice_smooth: Numerical stability epsilon (default: 1e-5)
    """
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device('cuda')
    ):
        """Initialize trainer with clDice loss configuration."""
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # clDice hyperparameters (can be modified before training)
        self.cldice_alpha = 0.4          # Weight for clDice loss (40% clDice, 60% Dice+CE)
        self.cldice_iterations = 10      # Skeleton extraction iterations
        self.cldice_smooth = 1e-5        # Numerical stability epsilon
        
        print(f"\n{'='*70}")
        print(f"🔬 Custom Trainer: nnUNetTrainerClDice")
        print(f"{'='*70}")
        print(f"Loss Configuration:")
        print(f"  - Base Loss: Dice + Cross Entropy")
        print(f"  - Topology Loss: clDice (soft skeletonization)")
        print(f"  - Alpha (clDice weight): {self.cldice_alpha}")
        print(f"  - Skeleton iterations (k): {self.cldice_iterations}")
        print(f"  - Combined: Loss = {1-self.cldice_alpha:.1f} * (Dice+CE) + {self.cldice_alpha:.1f} * clDice")
        print(f"{'='*70}\n")
    
    def _do_i_compile(self):
        """
        Disable torch.compile for this trainer.
        
        torch.compile can cause issues with custom losses and CUDA linking,
        so we disable it for the clDice trainer.
        """
        return False
    
    def _build_loss(self):
        """
        Override loss function to use combined Dice + CE + clDice.
        
        This method is called during trainer initialization to set up the loss function.
        We replace the standard nnU-Net loss with our combined loss that includes clDice.
        """
        # Get the standard nnU-Net loss (Dice + CE)
        # Note: nnU-Net v2 uses DC_and_CE_loss by default
        base_loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 
             'do_bg': False,  # Don't compute loss on background
             'ddp': self.is_ddp},
            {},
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label if self.label_manager.has_ignore_label else None
        )
        
        # Wrap with clDice loss
        # Note: CombinedClDiceLoss handles deep supervision internally
        combined_loss = CombinedClDiceLoss(
            base_loss=base_loss,
            alpha=self.cldice_alpha,
            cldice_iterations=self.cldice_iterations,
            cldice_smooth=self.cldice_smooth,
            apply_softmax=True  # Ensure input logits are converted to probabilities for skeletonization
        )
        
        return combined_loss
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Inherits the standard nnU-Net optimizer configuration.
        No changes needed for clDice loss.
        """
        return super().configure_optimizers()
    
    def on_train_start(self):
        """
        Called at the start of training.
        
        Log additional information about clDice configuration.
        """
        super().on_train_start()
        self.print_to_log_file(f"\nclDice Loss Configuration:")
        self.print_to_log_file(f"  Alpha: {self.cldice_alpha}")
        self.print_to_log_file(f"  Skeleton iterations: {self.cldice_iterations}")
        self.print_to_log_file(f"  Expected behavior: Improved topology (fewer disconnections)")


class nnUNetTrainerClDice_HighAlpha(nnUNetTrainerClDice):
    """
    Variant with higher clDice weight (alpha=0.6).
    
    Use this if you want to prioritize topology preservation over region overlap.
    May sacrifice some Dice score for better connectivity.
    
    Usage:
        nnUNetv2_train DATASET_ID 2d FOLD --trainer nnUNetTrainerClDice_HighAlpha
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cldice_alpha = 0.6


class nnUNetTrainerClDice_LowAlpha(nnUNetTrainerClDice):
    """
    Variant with lower clDice weight (alpha=0.2).
    
    Use this for a more conservative approach if the standard alpha=0.4 
    degrades Dice performance too much.
    
    Usage:
        nnUNetv2_train DATASET_ID 2d FOLD --trainer nnUNetTrainerClDice_LowAlpha
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cldice_alpha = 0.2


class nnUNetTrainerClDice_ThickStructures(nnUNetTrainerClDice):
    """
    Variant optimized for thicker tubular structures.
    
    Uses more skeleton iterations (k=10) to handle wider roads/vessels.
    
    Usage:
        nnUNetv2_train DATASET_ID 2d FOLD --trainer nnUNetTrainerClDice_ThickStructures
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cldice_iterations = 10

