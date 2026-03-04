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
# Supported models: "birefnet", "birefnet-general", "u2net"
# The system loads the primary model at startup and keeps it in memory.
# Hybrid mode uses RMBG for fast inference and BiRefNet for refinement.
MODEL_NAME = os.environ.get("STIX_MODEL", "birefnet")
MODEL_RESOLUTION = int(os.environ.get("STIX_MODEL_RES", "1024"))
ENABLE_HYBRID_PIPELINE = os.environ.get("STIX_HYBRID", "false").lower() == "true"

# Quality tiers map to model configurations
QUALITY_TIERS = {
    "fast": {
        "model": "birefnet",
        "resolution": 512,
        "description": "Fast processing, good for previews",
    },
    "balanced": {
        "model": "birefnet",
        "resolution": 1024,
        "description": "Balanced quality and speed (default)",
    },
    "quality": {
        "model": "birefnet",
        "resolution": 1024,
        "description": "Maximum quality with refinement pass",
        "refine": True,
    },
}

DEFAULT_QUALITY_TIER = "balanced"
