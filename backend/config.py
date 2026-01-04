import os
import tempfile
from pathlib import Path

# Use system temp directory (works on Windows/Linux/Docker)
# This keeps files OUT of your project folder
TEMP_BASE = Path(tempfile.gettempdir()) / "stix_app"
UPLOAD_DIR = TEMP_BASE / "uploads"
OUTPUT_DIR = TEMP_BASE / "output"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cleanup settings
FILE_RETENTION_SECONDS = 3600  # Delete files older than 1 hour
CLEANUP_INTERVAL_SECONDS = 600 # Run cleanup every 10 minutes
