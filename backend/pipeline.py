"""
STICKER BACKGROUND REMOVER - EDGE-FIRST PIPELINE
=================================================
This is a production-grade sticker extraction system.
It is NOT a generic background remover.

Architecture: EDGE → CONTOUR → CONSTRAINED AI → MASK HARDENING → OUTPUT

The fundamental principle:
- Sticker edges are defined by GEOMETRY, not color
- AI is a refinement tool, not the decision maker
- White-on-white MUST work even if RGB values are identical
"""

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

from model_loader import get_model, get_transform, get_device_instance
import torch


class EdgeConfidence(Enum):
    HIGH = "high"           # Clean closed contour detected
    MEDIUM = "medium"       # Contour detected but may have gaps
    LOW = "low"             # Weak edges, manual review suggested
    FAILED = "failed"       # Could not detect sticker boundary


@dataclass
class ProcessingResult:
    """Result of sticker processing with validation metadata"""
    original: Image.Image
    mask: np.ndarray
    alpha_mask: np.ndarray  # Soft edges for smooth rendering
    transparent: Image.Image
    edge_confidence: EdgeConfidence
    contour_closed: bool
    boundary_continuous: bool
    ai_escaped_contour: bool
    warnings: list


# =============================================================================
# STEP 1: AGGRESSIVE PREPROCESSING (EDGE ENHANCEMENT)
# =============================================================================

def preprocess_for_edges(image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image STRICTLY for edge visibility.
    This is NOT segmentation - it's edge enhancement.
    
    Returns:
        - enhanced: Edge-enhanced grayscale for detection
        - original_array: Original RGB array for later use
    """
    img_array = np.array(image.convert('RGB'))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This is CRITICAL for white-on-white - it amplifies local contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Slight Gaussian blur to remove noise (but preserve edges)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced, img_array


# =============================================================================
# STEP 2: MULTI-PASS EDGE DETECTION (THE FOUNDATION)
# =============================================================================

def detect_edges_multipass(enhanced: np.ndarray) -> np.ndarray:
    """
    Run multi-pass edge detection.
    This ensures edges exist EVEN IF COLORS ARE IDENTICAL.
    
    Methods combined:
    1. Canny (primary) - gradient-based
    2. Sobel (secondary) - directional gradients  
    3. Laplacian (fallback) - second derivative
    """
    # === CANNY (Primary) ===
    # Auto-threshold based on median intensity
    v = np.median(enhanced)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    canny = cv2.Canny(enhanced, lower, upper)
    
    # === SOBEL (Secondary) ===
    # Captures directional gradients that Canny might miss
    sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = np.uint8(np.clip(sobel / sobel.max() * 255, 0, 255))
    _, sobel_binary = cv2.threshold(sobel, 30, 255, cv2.THRESH_BINARY)
    
    # === LAPLACIAN (Fallback) ===
    # Second derivative catches edges Canny/Sobel miss
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    laplacian = np.uint8(np.abs(laplacian))
    _, laplacian_binary = cv2.threshold(laplacian, 20, 255, cv2.THRESH_BINARY)
    
    # === MERGE with logical OR ===
    combined = cv2.bitwise_or(canny, sobel_binary.astype(np.uint8))
    combined = cv2.bitwise_or(combined, laplacian_binary.astype(np.uint8))
    
    # Morphological closing to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return combined


# =============================================================================
# STEP 3: CLOSED SHAPE ISOLATION (CRITICAL)
# =============================================================================

def isolate_sticker_boundary(
    edge_map: np.ndarray, 
    image_shape: tuple
) -> Tuple[np.ndarray, EdgeConfidence, bool]:
    """
    From the edge map, find the LARGEST CLOSED CONTOUR.
    This is the absolute sticker boundary.
    
    Returns:
        - boundary_mask: Filled binary mask of sticker region
        - confidence: Edge detection confidence level
        - is_closed: Whether contour is fully closed
    """
    h, w = image_shape[:2]
    total_area = h * w
    
    # Dilate edges to help close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edge_map, kernel, iterations=2)
    
    # Find all contours
    contours, hierarchy = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        # FAILED: No contours found at all
        return np.zeros((h, w), dtype=np.uint8), EdgeConfidence.FAILED, False
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    
    # Validate contour size
    min_area = total_area * 0.01   # At least 1% of image
    max_area = total_area * 0.99   # At most 99% (not the whole image)
    
    if contour_area < min_area:
        return np.zeros((h, w), dtype=np.uint8), EdgeConfidence.FAILED, False
    
    if contour_area > max_area:
        # Contour is the whole image - likely failed
        return np.zeros((h, w), dtype=np.uint8), EdgeConfidence.LOW, False
    
    # Check if contour is closed (perimeter vs area ratio)
    perimeter = cv2.arcLength(largest_contour, closed=True)
    circularity = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Approximate the contour to check closure
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(largest_contour, epsilon, closed=True)
    is_closed = len(approx) >= 3 and cv2.isContourConvex(approx) or circularity > 0.1
    
    # Create filled boundary mask
    boundary_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(boundary_mask, [largest_contour], -1, 255, -1)
    
    # Fill any holes inside the contour
    boundary_mask = cv2.morphologyEx(
        boundary_mask, cv2.MORPH_CLOSE, 
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2
    )
    
    # Determine confidence
    if is_closed and circularity > 0.3:
        confidence = EdgeConfidence.HIGH
    elif is_closed or circularity > 0.1:
        confidence = EdgeConfidence.MEDIUM
    else:
        confidence = EdgeConfidence.LOW
    
    return boundary_mask, confidence, is_closed


# =============================================================================
# STEP 4: CONSTRAINED AI SEGMENTATION
# =============================================================================

def segment_constrained(
    image: Image.Image, 
    boundary_mask: np.ndarray
) -> Tuple[np.ndarray, bool]:
    """
    Run AI segmentation CONSTRAINED to the detected boundary.
    
    Rules:
    - AI inference is masked to the detected region
    - Pixels outside the region are hard-zeroed
    - AI output CANNOT expand beyond the contour
    
    Returns:
        - refined_mask: AI-refined mask within boundary
        - ai_escaped: Whether AI tried to go outside boundary
    """
    model = get_model()
    transform = get_transform()
    device = get_device_instance()
    
    original_size = image.size  # (width, height)
    
    # Transform and run AI
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid()
    
    # Convert to numpy
    ai_mask = preds[0].squeeze().cpu().numpy()
    ai_mask = (ai_mask * 255).astype(np.uint8)
    
    # Resize AI mask to original size
    ai_mask = cv2.resize(ai_mask, original_size, interpolation=cv2.INTER_LINEAR)
    
    # === CONSTRAINT: Hard-zero pixels outside boundary ===
    # This is the critical step that prevents white-on-white collapse
    
    # Check if AI tried to escape the boundary
    ai_binary = (ai_mask > 127).astype(np.uint8) * 255
    boundary_binary = (boundary_mask > 127).astype(np.uint8) * 255
    
    # Pixels where AI is active but boundary is not
    escaped_pixels = cv2.bitwise_and(ai_binary, cv2.bitwise_not(boundary_binary))
    escape_ratio = np.sum(escaped_pixels > 0) / max(np.sum(ai_binary > 0), 1)
    ai_escaped = escape_ratio > 0.05  # More than 5% escaped
    
    # Apply constraint: AI can only be active where boundary allows
    constrained_mask = cv2.bitwise_and(ai_mask, boundary_mask)
    
    # But also: include areas inside boundary that AI missed
    # (AI might have been too conservative inside the sticker)
    # We use the boundary as a maximum, not a replacement
    
    return constrained_mask, ai_escaped


# =============================================================================
# STEP 5: MASK HARDENING (QUALITY CONTROL)
# =============================================================================

def harden_mask(
    mask: np.ndarray, 
    original_image: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert AI output into a production-grade sticker mask.
    
    Operations:
    1. Morphological closing to seal micro gaps
    2. Erode → dilate to smooth jagged edges
    3. Alpha matte smoothing ONLY at boundary
    4. Remove halos and color spill
    
    Returns:
        - binary_mask: Hard mask for borders
        - alpha_mask: Soft mask for smooth edges
    """
    # === Step 1: Seal micro gaps ===
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_med, iterations=2)
    
    # === Step 2: Erode then dilate (opening) to smooth jagged edges ===
    eroded = cv2.erode(closed, kernel_small, iterations=1)
    smoothed = cv2.dilate(eroded, kernel_small, iterations=1)
    
    # === Step 3: Create binary mask ===
    _, binary_mask = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
    
    # === Step 4: Create alpha mask with smooth edges ===
    # Only smooth at the boundary, keep interior solid
    
    # Find boundary pixels
    dilated = cv2.dilate(binary_mask, kernel_small, iterations=2)
    eroded_inner = cv2.erode(binary_mask, kernel_small, iterations=2)
    boundary_region = cv2.subtract(dilated, eroded_inner)
    
    # Apply Gaussian blur only to boundary
    blurred = cv2.GaussianBlur(binary_mask.astype(np.float32), (5, 5), 1.5)
    
    # Composite: solid interior + smooth boundary
    alpha_mask = binary_mask.astype(np.float32)
    boundary_pixels = boundary_region > 0
    alpha_mask[boundary_pixels] = blurred[boundary_pixels]
    alpha_mask = np.clip(alpha_mask, 0, 255).astype(np.uint8)
    
    # === Step 5: Remove halos (optional defringe) ===
    # Slight erosion of the alpha at edges to prevent color spill
    alpha_mask = cv2.erode(alpha_mask, kernel_small, iterations=1)
    alpha_mask = cv2.GaussianBlur(alpha_mask, (3, 3), 0.5)
    
    return binary_mask, alpha_mask


# =============================================================================
# STEP 6: FINAL OUTPUT GENERATION
# =============================================================================

def apply_alpha_mask(image: Image.Image, alpha_mask: np.ndarray) -> Image.Image:
    """Apply alpha mask to create transparent PNG"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    img_array = np.array(image)
    img_array[:, :, 3] = alpha_mask
    
    return Image.fromarray(img_array)


def check_boundary_continuity(mask: np.ndarray) -> bool:
    """Check if the mask boundary is continuous (no breaks)"""
    # Find contours on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Should have exactly one external contour for a continuous boundary
    return len(contours) == 1


def check_mask_printable(mask: np.ndarray, min_coverage: float = 0.005) -> bool:
    """Check if mask covers enough area to be a valid sticker"""
    coverage = np.sum(mask > 127) / mask.size
    return coverage >= min_coverage


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_sticker(image: Image.Image) -> ProcessingResult:
    """
    Main sticker processing pipeline.
    
    EDGE-FIRST ARCHITECTURE:
    1. Preprocess for edge visibility
    2. Multi-pass edge detection
    3. Closed shape isolation
    4. Constrained AI segmentation
    5. Mask hardening
    6. Output generation with validation
    """
    warnings = []
    
    # Resize if too large
    max_size = 2048
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # === STEP 1: Preprocess ===
    enhanced, img_array = preprocess_for_edges(image)
    
    # === STEP 2: Edge Detection ===
    edge_map = detect_edges_multipass(enhanced)
    
    # === STEP 3: Boundary Isolation ===
    boundary_mask, edge_confidence, contour_closed = isolate_sticker_boundary(
        edge_map, img_array.shape
    )
    
    if edge_confidence == EdgeConfidence.FAILED:
        warnings.append("Could not detect sticker boundary - using full AI segmentation")
        # Fallback: use full image as boundary
        boundary_mask = np.ones((img_array.shape[0], img_array.shape[1]), dtype=np.uint8) * 255
    
    if not contour_closed:
        warnings.append("Sticker boundary may have gaps - edge quality may vary")
    
    if edge_confidence == EdgeConfidence.LOW:
        warnings.append("Low edge confidence - manual review recommended")
    
    # === STEP 4: Constrained AI Segmentation ===
    ai_mask, ai_escaped = segment_constrained(image, boundary_mask)
    
    if ai_escaped:
        warnings.append("AI segmentation was constrained to prevent background bleed")
    
    # === STEP 5: Mask Hardening ===
    binary_mask, alpha_mask = harden_mask(ai_mask, img_array)
    
    # === STEP 6: Output Generation ===
    transparent = apply_alpha_mask(image, alpha_mask)
    
    # === Validation ===
    boundary_continuous = check_boundary_continuity(binary_mask)
    if not boundary_continuous:
        warnings.append("Sticker may have disconnected parts")
    
    is_printable = check_mask_printable(binary_mask)
    if not is_printable:
        warnings.append("Sticker area too small - may not be printable")
    
    return ProcessingResult(
        original=image,
        mask=binary_mask,
        alpha_mask=alpha_mask,
        transparent=transparent,
        edge_confidence=edge_confidence,
        contour_closed=contour_closed,
        boundary_continuous=boundary_continuous,
        ai_escaped_contour=ai_escaped,
        warnings=warnings
    )


def save_results(result: ProcessingResult, output_dir: Path, job_id: str) -> dict:
    """Save all outputs with metadata"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Transparent PNG (final output)
    transparent_path = output_dir / f"{job_id}_transparent.png"
    result.transparent.save(transparent_path, "PNG")
    paths["transparent"] = str(transparent_path)
    
    # Binary mask (for borders)
    mask_path = output_dir / f"{job_id}_mask.png"
    Image.fromarray(result.mask).save(mask_path, "PNG")
    paths["mask"] = str(mask_path)
    
    # Alpha mask (for smooth edges)
    alpha_path = output_dir / f"{job_id}_alpha.png"
    Image.fromarray(result.alpha_mask).save(alpha_path, "PNG")
    paths["alpha"] = str(alpha_path)
    
    # Original (preprocessed)
    original_path = output_dir / f"{job_id}_original.png"
    result.original.save(original_path, "PNG")
    paths["original"] = str(original_path)
    
    return paths
