"""
Model definition for FaceTrait-ViT.

This module contains the MultiTaskViT model class used for demographic analysis.
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class MultiTaskViT(nn.Module):
    """
    Multi-task Vision Transformer for demographic prediction.
    
    This model uses a Vision Transformer backbone to predict three demographic attributes:
    - Age (regression)
    - Gender (binary classification)
    - Race (multi-class classification)
    """
    
    def __init__(self, backbone=None, num_races=5):
        """
        Initialize the MultiTaskViT model.
        
        Args:
            backbone: A Vision Transformer backbone. If None, will use vit_b_16 with ImageNet weights.
            num_races: Number of race categories to predict.
        """
        super().__init__()
        
        if backbone is None:
            backbone = vit_b_16(weights="IMAGENET1K_V1")
            
        self.backbone = backbone
        self.backbone.heads = nn.Identity()  # Remove classification head
        hidden_dim = self.backbone.hidden_dim
        
        # Age prediction head
        self.age_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # Gender prediction head
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # Race prediction head
        self.race_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_races)
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (age, gender, race) predictions:
            - age: Tensor of shape (batch_size) with age predictions
            - gender: Tensor of shape (batch_size) with gender logits
            - race: Tensor of shape (batch_size, num_races) with race logits
        """
        feats = self.backbone(x)
        age = self.age_head(feats)
        gender = self.gender_head(feats)
        race = self.race_head(feats)
        return age.squeeze(1), gender.squeeze(1), race