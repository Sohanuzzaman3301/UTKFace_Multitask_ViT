# FaceTrait-ViT: A Vision Transformer for Demographic Analysis

<img src="/home/prime/Vit_utk/images/facetrait_logo.png" alt="FaceTrait-ViT Logo" width="500"/>

FaceTrait-ViT is a multi-task learning model based on Vision Transformer (ViT) architecture, designed to simultaneously predict age, gender, and race from facial images using the UTKFace dataset.

## Key Features

- **Multi-Task Learning**: Single model for age regression, gender and race classification
- **ViT Architecture**: Leverages self-attention mechanisms for better feature extraction
- **High Accuracy**: Achieves strong performance across all three demographic tasks

<img src="/home/prime/Vit_utk/images/architecture_diagram.png" alt="Model Architecture" width="700"/>

## Dataset

The UTKFace dataset consists of over 20,000 face images with annotations of:
- **Age**: 0-116 years
- **Gender**: Male/Female
- **Race**: White, Black, Asian, Indian, Others

## Performance Metrics

Based on extensive evaluation, FaceTrait-ViT achieves the following performance:

<img src="/home/prime/Vit_utk/images/performance_metrics.png" alt="Performance Visualization" width="700"/>


### Confusion Matrices

The model's classification performance can be visualized through these confusion matrices:

<img src="/home/prime/Vit_utk/images/confusion_matrices.png" alt="Confusion Matrices" width="700"/>

For detailed performance metrics, including precision, recall, and F1-scores for each class, please see the [performance report](images/performance_report.txt).

## Sample Predictions

Here are some sample predictions made by FaceTrait-ViT across different demographic groups:

<img src="/home/prime/Vit_utk/images/sample_predictions.png" alt="Sample Predictions" width="700"/>

## Installation

```bash
# Clone the repository
git clone https://github.com/Sohanuzzaman3301/facetrait-vit-.git
cd facetrait-vit-

# Install dependencies
pip install torch torchvision pandas numpy pillow matplotlib scikit-learn tqdm

# Install the package in development mode (optional)
pip install -e .
```

## Using the Library

FaceTrait-ViT provides a simple API for making demographic predictions from facial images:

```python
# Import the library
from facetrait_vit import predict_image, visualize_prediction

# Make a prediction
result = predict_image("path/to/image.jpg")
print(f"Age: {result['age']}")
print(f"Gender: {result['gender']}")
print(f"Race: {result['race']}")

# Visualize the prediction
visualize_prediction("path/to/image.jpg")
```

### Batch Processing

```python
from facetrait_vit import predict_batch
import glob

# Get all images in a directory
image_paths = glob.glob("path/to/images/*.jpg")

# Process in batch
results = predict_batch(image_paths, batch_size=16)

# Print results
for result in results:
    print(f"Image: {result['path']}")
    print(f"  Age: {result['age']}")
    print(f"  Gender: {result['gender']}")
    print(f"  Race: {result['race']}")
```

### Advanced Usage

```python
from facetrait_vit import load_model, get_transform, predict_image

# Load model once for multiple predictions
model, device = load_model()
transform = get_transform()

# Make predictions with the same loaded model
result1 = predict_image("image1.jpg", model=model, device=device, transform=transform)
result2 = predict_image("image2.jpg", model=model, device=device, transform=transform)
```

## Command-line Usage

You can also use FaceTrait-ViT directly from the command line:

```bash
# Process a single image
python -m facetrait_vit.cli --input path/to/image.jpg --visualize

# Process all images in a directory
python -m facetrait_vit.cli --input path/to/images/ --output results/ --visualize --confidence
```

Command-line options:
- `--input`: Path to input image or directory of images (required)
- `--model`: Path to model weights (optional)
- `--output`: Output directory for visualizations (optional)
- `--visualize`: Visualize predictions
- `--batch-size`: Batch size for processing multiple images (default: 16)
- `--confidence`: Show confidence scores

## Limitations

- The model's predictions may exhibit biases present in the training data
- Age prediction accuracy is lower for very young and very old age groups
- Performance may vary across different racial groups due to dataset imbalances
- Face images must be properly aligned for best results

## Citation

If you use FaceTrait-ViT in your research or project, please cite:

```
@misc{facetrait_vit,
  author = {Md Sohanuzzaman, Shanto},
  title = {FaceTrait-ViT: Demographic Analysis with Vision Transformers},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Sohanuzzaman3301/facetrait-vit-}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

