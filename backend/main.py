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

# Directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load AI model on startup, cleanup on shutdown"""
    print("🚀 Starting Stix Backend...")
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

# Serve processed images
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
