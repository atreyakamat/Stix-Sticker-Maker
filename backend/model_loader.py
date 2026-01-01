"""
BiRefNet Model Loader
Loads the model once at startup and keeps it in memory
"""

import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

# Global model instance
_model = None
_transform = None
_device = None


def get_device():
    """Determine the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # AMD ROCm support
    if hasattr(torch, 'hip') and torch.hip.is_available():
        return torch.device("cuda")  # ROCm uses cuda device
    return torch.device("cpu")


def load_model():
    """Load BiRefNet model - called once at startup"""
    global _model, _transform, _device
    
    _device = get_device()
    print(f"🔧 Using device: {_device}")
    
    # Load BiRefNet from HuggingFace
    _model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet",
        trust_remote_code=True
    )
    _model.to(_device)
    _model.eval()
    
    # Standard transform for BiRefNet
    _transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return _model


def get_model():
    """Get the loaded model instance"""
    return _model


def get_transform():
    """Get the image transform"""
    return _transform


def get_device_instance():
    """Get the current device"""
    return _device
