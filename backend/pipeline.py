"""
Image Processing Pipeline
5-stage sticker extraction: Preprocess → Edge → Segment → Refine → Output
"""

import numpy as np
from PIL import Image
import cv2
import torch
from pathlib import Path

from model_loader import get_model, get_transform, get_device_instance
from mask_utils import smooth_mask, clean_artifacts, apply_mask, guided_filter


def preprocess_image(image: Image.Image, max_size: int = 2048) -> Image.Image:
    """
    Stage 1: Preprocess image
    - Fix orientation
    - Resize to max dimension
    - Convert to RGB
    """
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if too large (preserve aspect ratio)
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    return image


def detect_edges(image: Image.Image) -> np.ndarray:
    """
    Stage 2: Robust Edge detection for sticker boundary.
    Uses a combination of Sobel/Canny and adaptive thresholding to find boundaries
    even in low contrast (white-on-white) scenarios.
    """
    # Convert to numpy
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 1. Edge detection via gradients (Canny)
    # Use bilateral filter to preserve edges while removing noise
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    v = np.median(filtered)
    lower = int(max(0, 0.7 * v))
    upper = int(min(255, 1.3 * v))
    canny = cv2.Canny(filtered, lower, upper)
    
    # 2. Local contrast detection (Adaptive Thresholding)
    # This helps find boundaries where color differences are subtle but local
    adap = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Combined edge map
    combined = cv2.bitwise_or(canny, adap)
    
    # Morphological closing to join gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Dilate slightly to ensure we cover the edge region
    dilated = cv2.dilate(closed, kernel, iterations=1)
    
    return dilated


def segment_with_birefnet(image: Image.Image) -> np.ndarray:
    """
    Stage 3: AI segmentation using BiRefNet
    Returns soft alpha mask
    """
    model = get_model()
    transform = get_transform()
    device = get_device_instance()
    
    # Store original size
    original_size = image.size
    
    # Transform for model
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid()
    
    # Convert to numpy mask
    pred = preds[0].squeeze().cpu().numpy()
    
    # Resize back to original size
    mask = Image.fromarray((pred * 255).astype(np.uint8))
    mask = mask.resize(original_size, Image.LANCZOS)
    
    return np.array(mask)


def refine_mask(mask: np.ndarray, original_image: Image.Image, edge_hints: np.ndarray = None) -> np.ndarray:
    """
    Stage 4: Mask refinement
    - Uses Guided Filtering for alpha matting (Canva-quality edges)
    - Incorporates edge hints as a rough geometric constraint
    - Morphological cleanup and multi-component extraction
    """
    # 1. Initial cleanup using morphological operations on the raw AI mask
    # We do this on a slightly thresholded version to remove floaters
    _, rough = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    
    # Incorporate edge hints as a constraint (Boundary Enforcer)
    # We use them to "fill in" the AI mask if it's too aggressive, or clip it
    if edge_hints is not None:
        # Find all reasonable contours from edge hints
        contours, _ = cv2.findContours(edge_hints, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Create a shell mask from all significant contours
            shell_mask = np.zeros_like(mask)
            # Area-based filter to avoid small noise
            total_area = mask.shape[0] * mask.shape[1]
            min_area = total_area * 0.001 
            
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            if valid_contours:
                cv2.drawContours(shell_mask, valid_contours, -1, 255, -1)
                # Ensure the AI mask stays roughly within the shell, but give it leeway
                # We use bitwise OR to prevent BiRefNet from missing parts found by edge detection
                rough = cv2.bitwise_or(rough, cv2.bitwise_and(rough, shell_mask))
    
    # 2. Smooth and Clean
    kernel = np.ones((3, 3), np.uint8)
    # Remove small noise and close tiny gaps
    refined = cv2.morphologyEx(rough, cv2.MORPH_OPEN, kernel, iterations=1)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 3. Guided Filtering (Alpha Matting)
    # This is where the "Smoothing" and "Edge Accuracy" comes from.
    # It uses the original image to refine the mask boundary.
    img_array = np.array(original_image)
    if img_array.shape[:2] != refined.shape:
        img_array = cv2.resize(img_array, (refined.shape[1], refined.shape[0]))
        
    matted_mask = guided_filter(img_array, refined, r=8, eps=1e-3)
    
    # 4. Final Cleanup
    # Ensure no floating artifacts
    final_mask = clean_artifacts(matted_mask, min_area=200)
    
    return final_mask


def process_sticker(image: Image.Image, use_edge_detection: bool = True) -> dict:
    """
    Full pipeline: Process a sticker image
    Returns dict with mask and transparent image
    """
    # Stage 1: Preprocess
    preprocessed = preprocess_image(image)
    
    # Stage 2: Edge detection (optional, helps with white-on-white)
    edge_hints = None
    if use_edge_detection:
        edge_hints = detect_edges(preprocessed)
    
    # Stage 3: AI segmentation
    raw_mask = segment_with_birefnet(preprocessed)
    
    # Stage 4: Refine mask
    refined_mask = refine_mask(raw_mask, preprocessed, edge_hints)
    
    # Stage 5: Apply mask to create transparent output
    transparent = apply_mask(preprocessed, refined_mask)
    
    return {
        "original": preprocessed,
        "mask": refined_mask,
        "transparent": transparent
    }


def save_results(results: dict, output_dir: Path, job_id: str) -> dict:
    """Save processed results to disk"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save transparent PNG
    transparent_path = output_dir / f"{job_id}_transparent.png"
    results["transparent"].save(transparent_path, "PNG")
    paths["transparent"] = str(transparent_path)
    
    # Save mask
    mask_path = output_dir / f"{job_id}_mask.png"
    Image.fromarray(results["mask"]).save(mask_path, "PNG")
    paths["mask"] = str(mask_path)
    
    # Save original (preprocessed)
    original_path = output_dir / f"{job_id}_original.png"
    results["original"].save(original_path, "PNG")
    paths["original"] = str(original_path)
    
    return paths
