"""
Command-line interface for FaceTrait-ViT.

This module provides a CLI for using the FaceTrait-ViT model.
"""

import argparse
import os
import glob
import sys
from tqdm import tqdm

from .predictor import predict_image, predict_batch, visualize_prediction


def main():
    """Main entry point for the FaceTrait-ViT CLI."""
    parser = argparse.ArgumentParser(description='FaceTrait-ViT: Demographic Analysis with Vision Transformers')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory of images')
    parser.add_argument('--model', type=str, default=None, help='Path to model weights (optional)')
    parser.add_argument('--output', type=str, default=None, help='Output directory for visualizations')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for processing multiple images')
    parser.add_argument('--confidence', action='store_true', help='Show confidence scores')
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist")
        sys.exit(1)
    
    # Process input
    if os.path.isdir(args.input):
        # Process a directory of images
        image_paths = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_paths.extend(glob.glob(os.path.join(args.input, f'*.{ext}')))
            image_paths.extend(glob.glob(os.path.join(args.input, f'*.{ext.upper()}')))
        
        if not image_paths:
            print(f"No images found in '{args.input}'")
            sys.exit(1)
            
        print(f"Found {len(image_paths)} images. Processing...")
        
        # Process in batch
        results = predict_batch(
            image_paths,
            batch_size=args.batch_size,
            model_path=args.model,
            return_confidence=args.confidence
        )
        
        # Visualize if requested
        if args.visualize and args.output:
            os.makedirs(args.output, exist_ok=True)
            print(f"Generating visualizations...")
            
            for result in tqdm(results):
                image_path = result['path']
                filename = os.path.basename(image_path)
                base_name, _ = os.path.splitext(filename)
                output_path = os.path.join(args.output, f"{base_name}_pred.jpg")
                visualize_prediction(image_path, result, output_path)
        
        # Print summary
        print("\nResults summary:")
        for result in results:
            base_name = os.path.basename(result['path'])
            if args.confidence:
                print(f"{base_name}: Age={result['age']}, Gender={result['gender']} ({result['gender_confidence']:.2f}), Race={result['race']} ({result['race_confidence']:.2f})")
            else:
                print(f"{base_name}: Age={result['age']}, Gender={result['gender']}, Race={result['race']}")
    
    else:
        # Process a single image
        result = predict_image(args.input, model_path=args.model, return_confidence=args.confidence)
        
        # Print result
        print("\nPrediction:")
        if args.confidence:
            print(f"Age: {result['age']}")
            print(f"Gender: {result['gender']} (confidence: {result['gender_confidence']:.2f})")
            print(f"Race: {result['race']} (confidence: {result['race_confidence']:.2f})")
        else:
            print(f"Age: {result['age']}")
            print(f"Gender: {result['gender']}")
            print(f"Race: {result['race']}")
        
        # Visualize if requested
        if args.visualize:
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                filename = os.path.basename(args.input)
                base_name, _ = os.path.splitext(filename)
                output_path = os.path.join(args.output, f"{base_name}_pred.jpg")
                visualize_prediction(args.input, result, output_path)
                print(f"\nVisualization saved to: {output_path}")
            else:
                print("\nDisplaying visualization:")
                visualize_prediction(args.input, result)
    
    print("\nDone!")


if __name__ == "__main__":
    main()