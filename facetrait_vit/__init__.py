"""
FaceTrait-ViT: A Vision Transformer for Demographic Analysis

This package provides easy access to the FaceTrait-ViT model, which can predict
age, gender, and race from facial images using a Vision Transformer architecture.
"""

__version__ = "0.1.0"

# Import key components to make them available at the package level
from .model import MultiTaskViT
from .predictor import (
    predict_image,
    predict_batch,
    visualize_prediction,
    load_model,
    get_transform,
    RACE_LABELS,
    GENDER_LABELS
)

# For CLI usage
from .cli import main as cli_main