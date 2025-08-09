import torch
from PIL import Image
from torchvision import transforms
import json
import argparse
import platform # Import platform to check the OS

# Import the model architecture from our .py module
from model import ComplexCustomCNN, ResidualBlock 

# --- CONFIGURATION ---
DEFAULT_MODEL_PATH = "models/CustomCNN_v1/best_model.pth"
DEFAULT_CLASS_NAMES_PATH = "class_names.json"
NUM_CLASSES = 116

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """Loads the trained model from a checkpoint file."""
    print(f"Loading model from: {model_path}")
    
    # 1. Load the model architecture
    # The list [2, 2, 2, 2] must match the one used during training.
    model = ComplexCustomCNN(ResidualBlock, [2, 2, 2, 2], num_classes=NUM_CLASSES)
    
    # =========================================================================
    # ✨✨✨ THE FIX: Platform-Aware Compilation for Inference ✨✨✨
    # =========================================================================
    if int(torch.__version__.split('.')[0]) >= 2:
        # Check the operating system
        if platform.system() == "Windows":
            print("Windows OS detected. Skipping torch.compile() for inference.")
        else:
            # On Linux or MacOS, compile the model for a potential inference speed boost.
            print("Linux or MacOS detected. Attempting to compile the model...")
            try:
                model = torch.compile(model)
                print("Inference model compiled successfully!")
            except Exception as e:
                print(f"Model compilation failed: {e}. Continuing with un-compiled model.")
    
    # 2. Load the trained weights
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
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Apply transformations and add a batch dimension
    image_tensor = inference_transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad(): # Disables gradient calculation for efficiency
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs.float(), dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted_idx.item()]
    
    return predicted_class, confidence.item()

# --- Main execution block ---
if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Predict a plant disease from an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the trained model .pth file.")
    parser.add_argument("--class_names", type=str, default=DEFAULT_CLASS_NAMES_PATH, help="Path to the class names JSON file.")
    
    args = parser.parse_args()

    # Load class names from the JSON file
    try:
        with open(args.class_names, 'r') as f:
            class_names = json.load(f)
    except FileNotFoundError:
        print(f"Error: Class names file not found at {args.class_names}")
        exit()

    # Load the model
    model = load_model(args.model_path)
    
    # Make a prediction
    prediction, confidence = predict(model, args.image_path, class_names)
    
    # Print the results
    print("\n--- Prediction Result ---")
    print(f"Image Path:      {args.image_path}")
    print(f"Predicted Class: {prediction}")
    print(f"Confidence:      {confidence:.4f}")