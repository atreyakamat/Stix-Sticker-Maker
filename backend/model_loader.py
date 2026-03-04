"""
Model Loader — Multi-Model Architecture
=========================================
Loads the segmentation model once at startup and keeps it in memory.
Supports BiRefNet (primary) with configuration for future model switching.
The model selection is driven by config.MODEL_NAME and can be overridden
at runtime via the /api/model endpoint.
"""

import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

from config import MODEL_NAME, MODEL_RESOLUTION, QUALITY_TIERS, DEFAULT_QUALITY_TIER

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_model = None
_transform = None
_device = None
_model_name = None          # The human-readable name of the loaded model
_model_resolution = None    # Resolution used for inference

# Map of supported model identifiers → HuggingFace repo IDs
MODEL_REGISTRY = {
    "birefnet":         "ZhengPeng7/BiRefNet",           # Standard BiRefNet
    "birefnet-massive": "ZhengPeng7/BiRefNet",           # High-resolution optimized
    "birefnet-general": "ZhengPeng7/BiRefNet-general",   # General purpose
    "birefnet-portrait":"ZhengPeng7/BiRefNet-portrait",  # Optimized for portraits
    "rmbg-2.0":         "briaai/RMBG-2.0",               # SOTA commercial model (BiRefNet architecture)
}


def get_device():
    """Determine the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, 'hip') and torch.hip.is_available():
        return torch.device("cuda")  # ROCm uses cuda device
    return torch.device("cpu")


def load_model(model_key: str | None = None, resolution: int | None = None):
    """
    Load a segmentation model — called once at startup.
    
    Parameters
    ----------
    model_key : str, optional
        Key from MODEL_REGISTRY. Defaults to config.MODEL_NAME.
    resolution : int, optional
        Inference resolution (square). Defaults to config.MODEL_RESOLUTION.
    """
    global _model, _transform, _device, _model_name, _model_resolution

    model_key = model_key or MODEL_NAME
    resolution = resolution or MODEL_RESOLUTION

    _device = get_device()
    _model_name = model_key
    _model_resolution = resolution
    print(f"🔧 Using device: {_device}")

    repo_id = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY["birefnet"])
    print(f"📦 Loading model '{model_key}' from {repo_id} @ {resolution}px ...")

    _model = AutoModelForImageSegmentation.from_pretrained(
        repo_id,
        trust_remote_code=True,
    )
    _model.to(_device)
    _model.eval()

    _transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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


def get_model_info() -> dict:
    """Return metadata about the currently loaded model"""
    return {
        "name": _model_name,
        "resolution": _model_resolution,
        "device": str(_device),
        "available_models": list(MODEL_REGISTRY.keys()),
        "quality_tiers": QUALITY_TIERS,
        "default_tier": DEFAULT_QUALITY_TIER,
    }
