"""
Stix Backend - Main FastAPI Application
Sticker Background Removal & Generator Platform
"""

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routes import router
from model_loader import load_model, get_model

from config import UPLOAD_DIR, OUTPUT_DIR, FILE_RETENTION_SECONDS, CLEANUP_INTERVAL_SECONDS
import time
import threading

def run_auto_cleanup():
    """Background task to delete old files"""
    while True:
        try:
            now = time.time()
            cutoff = now - FILE_RETENTION_SECONDS
            
            # Check uploads and output
            for directory in [UPLOAD_DIR, OUTPUT_DIR]:
                if not directory.exists():
                    continue
                for file_path in directory.glob("*"):
                    if file_path.is_file():
                        # If modified time is older than cutoff
                        if file_path.stat().st_mtime < cutoff:
                            try:
                                file_path.unlink()
                                print(f"🧹 Cleaned up: {file_path.name}")
                            except Exception as e:
                                print(f"Error deleting {file_path.name}: {e}")
                                
        except Exception as e:
            print(f"Cleanup error: {e}")
            
        time.sleep(CLEANUP_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load AI model on startup, start cleanup thread"""
    print("🚀 Starting Stix Backend...")
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=run_auto_cleanup, daemon=True)
    cleanup_thread.start()
    print("🧹 Auto-cleanup task started")

    print("📦 Loading BiRefNet model (this may take a moment on first run)...")
    load_model()
    print("✅ Model loaded successfully!")
    yield
    print("👋 Shutting down Stix Backend...")


app = FastAPI(
    title="Stix API",
    description="Sticker Background Removal & Generator Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve processed images from TEMP dir
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# Include API routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "Stix API",
        "version": "1.0.0",
        "status": "running",
        "model": "BiRefNet"
    }


@app.get("/health")
async def health_check():
    model = get_model()
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }
