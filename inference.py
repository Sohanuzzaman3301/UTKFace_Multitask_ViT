import os
import argparse
import torch
from torchvision.models import vit_b_16
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

class MultiTaskViT(torch.nn.Module):
    def __init__(self, backbone, num_races=5):
        super().__init__()
        self.backbone = backbone
        self.backbone.heads = torch.nn.Identity()  # Remove classification head
        hidden_dim = self.backbone.hidden_dim
        self.age_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 1)
        )
        self.gender_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 1)
        )
        self.race_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, num_races)
        )
    def forward(self, x):
        feats = self.backbone(x)
        age = self.age_head(feats)
        gender = self.gender_head(feats)
        race = self.race_head(feats)
        return age.squeeze(1), gender.squeeze(1), race

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MultiTaskViT(vit_b_16(weights="IMAGENET1K_V1"))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

def preprocess_image():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform

def predict_demographics(image_path, model, device, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        age_pred, gender_pred, race_pred = model(image_tensor)
        
        # Process the predictions
        age = int(age_pred.item())
        gender_prob = torch.sigmoid(gender_pred).item()
        gender = "Female" if gender_prob > 0.5 else "Male"
        gender_conf = max(gender_prob, 1-gender_prob)
        
        race_idx = torch.argmax(race_pred, dim=1).item()
        race_probs = torch.softmax(race_pred, dim=1)[0].cpu().numpy()
        race_conf = float(race_probs[race_idx])
        
        race_labels = ['White', 'Black', 'Asian', 'Indian', 'Others']
        race = race_labels[race_idx]
        
    return {
        "age": age,
        "gender": gender,
        "gender_confidence": gender_conf,
        "race": race,
        "race_confidence": race_conf,
        "image": image
    }

def visualize_prediction(result, output_dir=None, filename=None):
    # Create a figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Display the image
    ax.imshow(result["image"])
    ax.set_title("Prediction Results", fontsize=14)
    
    # Set prediction text with confidence
    prediction_text = (
        f"Age: {result['age']}\n"
        f"Gender: {result['gender']} ({result['gender_confidence']:.2f})\n"
        f"Race: {result['race']} ({result['race_confidence']:.2f})"
    )
    
    # Add text box with predictions
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.05, 0.95, prediction_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_pred.jpg")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.tight_layout()
        plt.show()
        plt.close()
        return None

def main():
    parser = argparse.ArgumentParser(description='UTKFace Demographics Prediction with ViT')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory of images')
    parser.add_argument('--model', type=str, default='vit_multitask_model.pth', help='Path to model weights')
    parser.add_argument('--output', type=str, default='predictions', help='Output directory for visualization')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    args = parser.parse_args()
    
    # Load model
    model, device = load_model(args.model)
    transform = preprocess_image()
    
    # Check if input is directory or file
    if os.path.isdir(args.input):
        image_paths = glob.glob(os.path.join(args.input, '*.jpg')) + \
                      glob.glob(os.path.join(args.input, '*.jpeg')) + \
                      glob.glob(os.path.join(args.input, '*.png'))
        print(f"Found {len(image_paths)} images")
    else:
        image_paths = [args.input]
    
    # Process each image
    results = []
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            result = predict_demographics(image_path, model, device, transform)
            
            # Add filename to result
            filename = os.path.basename(image_path)
            result["filename"] = filename
            
            results.append(result)
            
            # Visualize prediction
            if args.visualize:
                output_path = visualize_prediction(result, args.output, filename)
                if output_path:
                    print(f"Visualization saved to {output_path}")
            
            # Print prediction
            print(f"\nPrediction for {filename}:")
            print(f"  Age: {result['age']}")
            print(f"  Gender: {result['gender']} (confidence: {result['gender_confidence']:.2f})")
            print(f"  Race: {result['race']} (confidence: {result['race_confidence']:.2f})")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print(f"\nProcessed {len(results)} images successfully")

if __name__ == "__main__":
    main()