"""
Loss functions for Edge Detection, including weighted BCE and Dice Loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from config import TrainConfig

def weighted_bce_loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: float, neg_weight: float) -> torch.Tensor:
    """
    Calculates a weighted binary cross-entropy loss.
    This is useful for imbalanced datasets where one class (e.g., edges) is much rarer than the other (background).

    Args:
        pred (torch.Tensor): The model's prediction (logits), shape (B, 1, H, W).
        target (torch.Tensor): The ground truth label, shape (B, 1, H, W).
        pos_weight (float): The weight for positive samples (edges).
        neg_weight (float): The weight for negative samples (non-edges).

    Returns:
        torch.Tensor: The calculated weighted BCE loss.
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    
    # Create a weight map based on the target
    weight_map = torch.ones_like(target)
    weight_map[target > 0.5] = pos_weight
    weight_map[target <= 0.5] = neg_weight
    
    # Apply weights and calculate mean loss
    weighted_loss = bce * weight_map
    return weighted_loss.mean()

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Calculates the Dice loss, which is effective for segmentation tasks.
    It measures the overlap between the predicted and ground truth regions.

    Args:
        pred (torch.Tensor): The model's prediction (probabilities), shape (B, 1, H, W).
        target (torch.Tensor): The ground truth label, shape (B, 1, H, W).
        smooth (float): A smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: The calculated Dice loss.
    """
    pred = torch.sigmoid(pred) # Ensure pred is probability
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice_coeff = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1 - dice_coeff

class EdgeDetectionLoss(nn.Module):
    """
    A combined loss function for edge detection that includes weighted BCE and Dice loss.
    The final loss is a weighted sum of the two individual losses.
    """
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculates the combined loss.

        Args:
            pred_logits (torch.Tensor): The raw output (logits) from the model.
            target (torch.Tensor): The ground truth edge map.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the total loss and individual loss components.
        """
        # BCE Loss
        bce = weighted_bce_loss(
            pred_logits, 
            target, 
            self.config.bce_pos_weight, 
            self.config.bce_neg_weight
        )
        
        # Dice Loss
        dice = dice_loss(pred_logits, target)
        
        # Total Loss
        total_loss = (self.config.bce_weight * bce) + (self.config.dice_weight * dice)
        
        return {
            'total_loss': total_loss,
            'bce_loss': bce,
            'dice_loss': dice
        }