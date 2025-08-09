import torch
import torch.nn as nn
from torchvision import models

# This file contains a function to build our fine-tuned EfficientNet-B3 model.

def build_efficientnet_b3(num_classes=116, pretrained=True):
    """
    Builds a fine-tuned EfficientNet-B3 model.

    Args:
        num_classes (int): The number of output classes for the final layer.
        pretrained (bool): If True, loads weights pre-trained on ImageNet.
                           For inference, this should always be True.

    Returns:
        A PyTorch model instance.
    """
    # 1. Load the pre-trained EfficientNet-B3 model
    if pretrained:
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
    else:
        weights = None
        
    model = models.efficientnet_b3(weights=weights)

    # 2. Freeze all the parameters in the feature extractor
    # This is important for inference as well, to ensure layers like BatchNorm are in eval mode.
    for param in model.parameters():
        param.requires_grad = False

    # 3. Replace the final classifier layer
    # Get the number of input features from the original classifier
    num_ftrs = model.classifier[1].in_features
    
    # Create a new classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model