import torch
from PIL import Image
from torchvision import transforms
import json
import argparse
import platform

# Import the model builder function from our new .py module
from EN_B3_model import build_efficientnet_b3

# --- CONFIGURATION ---
# Path to the checkpoint file saved from your fine-tuning notebook
DEFAULT_MODEL_PATH = "models/EfficientNetB3_FineTuned/best_model.pth"
DEFAULT_CLASS_NAMES_PATH = "class_names.json"
NUM_CLASSES = 116

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """Loads the fine-tuned model from a checkpoint file."""
    print(f"Loading model from: {model_path}")
    
    # 1. Build the model architecture using our helper function
    # We pass pretrained=False because we are about to load our own fine-tuned weights.
    # The function will still download the architecture correctly.
    model = build_efficientnet_b3(num_classes=NUM_CLASSES, pretrained=False)
    
    # ✨ Optional: Compile the model for faster inference on PyTorch 2.x ✨
    if int(torch.__version__.split('.')[0]) >= 2:
        if platform.system() != "Windows":
            print("Compiling the model for faster inference...")
            model = torch.compile(model)

    # 2. Load the trained weights from our fine-tuning checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 3. Set the model to evaluation mode
    model.eval()
    model.to(device)
    
    print("Model loaded successfully and is ready for inference.")
    return model

def predict(model, image_path, class_names):
    """
    Takes an image path and returns the model's prediction and confidence.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return f"Error: Image not found at {image_path}", 0.0

    # Define the image transformation (must match validation transform)
    # Note: EfficientNet models have specific input sizes they were trained on.
    # B3 was trained on 300x300, but fine-tuning on 224x224 is common and works well.
    # We will stick to 224x224 to match our training setup.
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = inference_transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs.float(), dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted_idx.item()]
    
    return predicted_class, confidence.item()

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a plant disease using a fine-tuned EfficientNet-B3 model.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the trained model .pth file.")
    parser.add_argument("--class_names", type=str, default=DEFAULT_CLASS_NAMES_PATH, help="Path to the class names JSON file.")
    
    args = parser.parse_args()

    try:
        with open(args.class_names, 'r') as f:
            class_names = json.load(f)
    except FileNotFoundError:
        print(f"Error: Class names file not found at {args.class_names}")
        exit()

    model = load_model(args.model_path)
    prediction, confidence = predict(model, args.image_path, class_names)
    
    print("\n--- Prediction Result ---")
    print(f"Image Path:      {args.image_path}")
    print(f"Predicted Class: {prediction}")
    print(f"Confidence:      {confidence:.4f}")