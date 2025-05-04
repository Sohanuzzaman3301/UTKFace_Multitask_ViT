"""
Prediction utilities for FaceTrait-ViT.

This module provides functions to easily load the model and make predictions.
"""

import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from .model import MultiTaskViT

# Default paths
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vit_multitask_model.pth")

# Race and gender label mappings
RACE_LABELS = ['White', 'Black', 'Asian', 'Indian', 'Others']
GENDER_LABELS = ['Male', 'Female']

def load_model(model_path=None, device=None):
    """
    Load the FaceTrait-ViT model.
    
    Args:
        model_path: Path to the model weights. If None, uses the default path.
        device: Device to load the model on. If None, uses CUDA if available, else CPU.
        
    Returns:
        model: Loaded MultiTaskViT model
        device: Device the model is loaded on
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
        
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Create and load model
    model = MultiTaskViT()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

def get_transform():
    """
    Get the image transformation pipeline for inference.
    
    Returns:
        transform: torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

def predict_image(image_path, model=None, device=None, transform=None, return_confidence=False):
    """
    Predict demographics from an image.
    
    Args:
        image_path: Path to the image file
        model: Pre-loaded model. If None, will load the default model.
        device: Device to run inference on. If None, uses CUDA if available, else CPU.
        transform: Image transformation pipeline. If None, uses the default transform.
        return_confidence: Whether to return confidence scores
        
    Returns:
        Dictionary with predictions:
        - age: Predicted age in years
        - gender: Predicted gender (Male/Female)
        - race: Predicted race category
        - gender_confidence: (Optional) Confidence in gender prediction
        - race_confidence: (Optional) Confidence in race prediction
    """
    # Load model if not provided
    if model is None or device is None:
        model, device = load_model(device=device)
        
    # Get transform if not provided
    if transform is None:
        transform = get_transform()
    
    # Prepare image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run prediction
    with torch.no_grad():
        age_pred, gender_pred, race_pred = model(image_tensor)
        
        # Process results
        age = int(age_pred.item())
        gender_prob = torch.sigmoid(gender_pred).item()
        gender = GENDER_LABELS[1] if gender_prob > 0.5 else GENDER_LABELS[0]
        gender_conf = max(gender_prob, 1-gender_prob)
        
        race_probs = torch.softmax(race_pred, dim=1)[0].cpu().numpy()
        race_idx = torch.argmax(race_pred, dim=1).item()
        race = RACE_LABELS[race_idx]
        race_conf = float(race_probs[race_idx])
    
    result = {
        "age": age,
        "gender": gender,
        "race": race
    }
    
    if return_confidence:
        result["gender_confidence"] = gender_conf
        result["race_confidence"] = race_conf
        
    return result

def visualize_prediction(image_path, prediction=None, output_path=None):
    """
    Visualize the prediction on an image.
    
    Args:
        image_path: Path to the image file
        prediction: Prediction dictionary. If None, a prediction will be made.
        output_path: Path to save the visualization. If None, the image is displayed.
        
    Returns:
        output_path: Path to the saved visualization if output_path was provided, None otherwise.
    """
    # Get prediction if not provided
    if prediction is None:
        prediction = predict_image(image_path, return_confidence=True)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.set_title("FaceTrait-ViT Prediction", fontsize=14)
    
    # Set prediction text with confidence
    if "gender_confidence" in prediction and "race_confidence" in prediction:
        prediction_text = (
            f"Age: {prediction['age']}\n"
            f"Gender: {prediction['gender']} ({prediction['gender_confidence']:.2f})\n"
            f"Race: {prediction['race']} ({prediction['race_confidence']:.2f})"
        )
    else:
        prediction_text = (
            f"Age: {prediction['age']}\n"
            f"Gender: {prediction['gender']}\n"
            f"Race: {prediction['race']}"
        )
    
    # Add text box with predictions
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.05, 0.95, prediction_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.tight_layout()
        plt.show()
        plt.close()
        return None

def predict_batch(image_paths, batch_size=16, model=None, device=None, transform=None, return_confidence=False):
    """
    Predict demographics for a batch of images.
    
    Args:
        image_paths: List of paths to image files
        batch_size: Batch size for processing
        model: Pre-loaded model. If None, will load the default model.
        device: Device to run inference on. If None, uses CUDA if available, else CPU.
        transform: Image transformation pipeline. If None, uses the default transform.
        return_confidence: Whether to return confidence scores
        
    Returns:
        List of dictionaries with predictions for each image
    """
    # Load model if not provided
    if model is None or device is None:
        model, device = load_model(device=device)
        
    # Get transform if not provided
    if transform is None:
        transform = get_transform()
    
    results = []
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        # Prepare batch
        for path in batch_paths:
            try:
                image = Image.open(path).convert('RGB')
                image_tensor = transform(image)
                batch_images.append(image_tensor)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Stack tensors and predict
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            age_preds, gender_preds, race_preds = model(batch_tensor)
            
            # Process predictions
            for j, path in enumerate(batch_paths[:len(batch_images)]):
                age = int(age_preds[j].item())
                
                gender_prob = torch.sigmoid(gender_preds[j]).item()
                gender = GENDER_LABELS[1] if gender_prob > 0.5 else GENDER_LABELS[0]
                gender_conf = max(gender_prob, 1-gender_prob)
                
                race_probs = torch.softmax(race_preds[j], dim=0).cpu().numpy()
                race_idx = torch.argmax(race_preds[j], dim=0).item()
                race = RACE_LABELS[race_idx]
                race_conf = float(race_probs[race_idx])
                
                result = {
                    "path": path,
                    "age": age,
                    "gender": gender,
                    "race": race
                }
                
                if return_confidence:
                    result["gender_confidence"] = gender_conf
                    result["race_confidence"] = race_conf
                
                results.append(result)
    
    return results