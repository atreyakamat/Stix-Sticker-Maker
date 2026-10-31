import os
import tempfile
from pathlib import Path

# Use system temp directory (works on Windows/Linux/Docker)
# This keeps files OUT of your project folder
TEMP_BASE = Path(tempfile.gettempdir()) / "stix_app"
UPLOAD_DIR = TEMP_BASE / "uploads"
OUTPUT_DIR = TEMP_BASE / "output"
MASK_DIR = TEMP_BASE / "masks"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

# Cleanup settings
FILE_RETENTION_SECONDS = 3600  # Delete files older than 1 hour
CLEANUP_INTERVAL_SECONDS = 600 # Run cleanup every 10 minutes

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# The backend keeps a small LRU cache of segmentation models in memory and
# loads them lazily the first time a quality tier needs them. See
# model_loader.py for the registry of available models.
#
# MODEL_NAME     The model loaded eagerly at startup (also the fallback).
# MODEL_RES      Default inference resolution (square) for that model.
# DEVICE         "auto" | "cpu" | "cuda" | "dml" (DirectML, Windows + AMD/Intel GPU).
# MODEL_CACHE    How many distinct models to keep resident in RAM at once.
MODEL_NAME = os.environ.get("STIX_MODEL", "birefnet")
MODEL_RESOLUTION = int(os.environ.get("STIX_MODEL_RES", "1024"))
DEVICE = os.environ.get("STIX_DEVICE", "auto").lower()
MODEL_CACHE_SIZE = int(os.environ.get("STIX_MODEL_CACHE", "2"))

# Kept for backwards compatibility with older code paths. The edge-first
# hybrid pipeline is always active; this flag no longer gates behaviour.
ENABLE_HYBRID_PIPELINE = os.environ.get("STIX_HYBRID", "false").lower() == "true"

# ============================================================================
# QUALITY TIERS
# ============================================================================
# Each tier is a complete inference recipe. Unlike the previous version, the
# `model` and `resolution` fields are actually honoured by the pipeline.
#
#   model       Key into model_loader.MODEL_REGISTRY.
#   resolution  Square inference resolution fed to the network.
#   matting     Use edge-aware soft-alpha refinement (great for hair/glow).
#   tta         Flip test-time augmentation: run the image + its mirror and
#               average. ~2x slower, removes stray specks. Use for "quality".
#   refine      Legacy flag kept so existing callers keep working; mirrors
#               `tta` so an old `refine=True` still means a higher-effort pass.
QUALITY_TIERS = {
    "fast": {
        "model": "birefnet_lite",
        "resolution": 1024,
        "matting": False,
        "tta": False,
        "refine": False,
        "description": "Fastest. Lightweight BiRefNet, good for previews and bulk runs.",
    },
    "balanced": {
        "model": "birefnet",
        "resolution": 1024,
        "matting": True,
        "tta": False,
        "refine": False,
        "description": "Default. Full BiRefNet at 1024px with soft edge matting.",
    },
    "quality": {
        "model": "birefnet_hr_matting",
        "resolution": 2048,
        "matting": True,
        "tta": True,
        "refine": True,
        "description": "Maximum accuracy. High-res matting model at 2048px with flip averaging (slow on CPU).",
    },
}

DEFAULT_QUALITY_TIER = os.environ.get("STIX_DEFAULT_TIER", "balanced")
