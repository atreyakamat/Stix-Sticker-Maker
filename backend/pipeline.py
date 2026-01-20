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
    Uses multiple enhancement strategies to maximize edge contrast.
    
    Returns:
        - enhanced: Edge-enhanced grayscale for detection
        - original_array: Original RGB array for later use
    """
    img_array = np.array(image.convert('RGB'))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # === STRATEGY 1: CLAHE Enhancement ===
    # Critical for white-on-white - amplifies local contrast
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced_clahe = clahe.apply(gray)
    
    # === STRATEGY 2: Bilateral Filter ===
    # Smooths while preserving edges
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # === STRATEGY 3: Unsharp Mask ===
    # Sharpens edges significantly
    gaussian = cv2.GaussianBlur(gray, (0, 0), 3)
    unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    
    # === STRATEGY 4: Local Variance Enhancement ===
    # Highlights areas with texture variation (edges)
    mean = cv2.blur(gray.astype(np.float32), (5, 5))
    sq_mean = cv2.blur(gray.astype(np.float32)**2, (5, 5))
    variance = np.sqrt(np.maximum(sq_mean - mean**2, 0))
    variance_enhanced = np.uint8(np.clip(variance * 3, 0, 255))
    
    # Combine all enhancements with weighted average
    enhanced = cv2.addWeighted(enhanced_clahe, 0.4, unsharp, 0.3, 0)
    enhanced = cv2.addWeighted(enhanced, 0.8, variance_enhanced, 0.2, 0)
    
    # Final light blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
    
    return enhanced, img_array


# =============================================================================
# STEP 2: MULTI-PASS EDGE DETECTION (ULTRA-STRONG)
# =============================================================================

def detect_edges_multipass(enhanced: np.ndarray, original_rgb: np.ndarray = None) -> np.ndarray:
    """
    Run ULTRA-STRONG multi-pass edge detection.
    Uses 5 different methods and combines them for maximum edge coverage.
    
    Methods:
    1. Canny (multiple thresholds)
    2. Sobel (gradient magnitude)
    3. Laplacian (second derivative)
    4. Scharr (optimized gradient)
    5. Morphological gradient
    """
    h, w = enhanced.shape
    
    # === METHOD 1: MULTI-THRESHOLD CANNY ===
    # Run Canny at multiple sensitivity levels
    v = np.median(enhanced)
    
    # Sensitive (catches weak edges)
    canny_sensitive = cv2.Canny(enhanced, max(0, v * 0.3), v * 0.7)
    
    # Standard
    canny_standard = cv2.Canny(enhanced, max(0, v * 0.5), v * 1.0)
    
    # Conservative (strong edges only)
    canny_conservative = cv2.Canny(enhanced, max(0, v * 0.7), v * 1.3)
    
    # Combine all Canny results
    canny_combined = cv2.bitwise_or(canny_sensitive, canny_standard)
    canny_combined = cv2.bitwise_or(canny_combined, canny_conservative)
    
    # === METHOD 2: SOBEL GRADIENT ===
    sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_norm = np.uint8(np.clip(sobel_mag / sobel_mag.max() * 255, 0, 255)) if sobel_mag.max() > 0 else np.zeros_like(enhanced)
    _, sobel_binary = cv2.threshold(sobel_norm, 20, 255, cv2.THRESH_BINARY)
    
    # === METHOD 3: SCHARR (More accurate than Sobel) ===
    scharr_x = cv2.Scharr(enhanced, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(enhanced, cv2.CV_64F, 0, 1)
    scharr_mag = np.sqrt(scharr_x**2 + scharr_y**2)
    scharr_norm = np.uint8(np.clip(scharr_mag / scharr_mag.max() * 255, 0, 255)) if scharr_mag.max() > 0 else np.zeros_like(enhanced)
    _, scharr_binary = cv2.threshold(scharr_norm, 20, 255, cv2.THRESH_BINARY)
    
    # === METHOD 4: LAPLACIAN ===
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F, ksize=3)
    laplacian_abs = np.uint8(np.abs(laplacian))
    _, laplacian_binary = cv2.threshold(laplacian_abs, 15, 255, cv2.THRESH_BINARY)
    
    # === METHOD 5: MORPHOLOGICAL GRADIENT ===
    # Dilation minus erosion = edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)
    _, morph_binary = cv2.threshold(morph_gradient, 15, 255, cv2.THRESH_BINARY)
    
    # === METHOD 6: COLOR EDGE DETECTION (if RGB available) ===
    color_edges = np.zeros((h, w), dtype=np.uint8)
    if original_rgb is not None:
        # Detect edges in each color channel and combine
        for c in range(3):
            channel = original_rgb[:, :, c]
            channel_edges = cv2.Canny(channel, 30, 100)
            color_edges = cv2.bitwise_or(color_edges, channel_edges)
    
    # === FUSION: Combine all methods ===
    combined = np.zeros((h, w), dtype=np.uint8)
    combined = cv2.bitwise_or(combined, canny_combined)
    combined = cv2.bitwise_or(combined, sobel_binary.astype(np.uint8))
    combined = cv2.bitwise_or(combined, scharr_binary.astype(np.uint8))
    combined = cv2.bitwise_or(combined, laplacian_binary.astype(np.uint8))
    combined = cv2.bitwise_or(combined, morph_binary.astype(np.uint8))
    combined = cv2.bitwise_or(combined, color_edges)
    
    # === POST-PROCESSING ===
    # Close small gaps in edges
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Dilate slightly to strengthen weak edges
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    combined = cv2.dilate(combined, kernel_dilate, iterations=1)
    
    # Clean up noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    return combined


# =============================================================================
# STEP 3: CLOSED SHAPE ISOLATION (ULTRA-STRONG)
# =============================================================================

def isolate_sticker_boundary(
    edge_map: np.ndarray, 
    image_shape: tuple
) -> Tuple[np.ndarray, EdgeConfidence, bool]:
    """
    From the edge map, find the LARGEST CLOSED CONTOUR.
    Uses multiple strategies to ensure robust contour detection.
    
    Returns:
        - boundary_mask: Filled binary mask of sticker region
        - confidence: Edge detection confidence level
        - is_closed: Whether contour is fully closed
    """
    h, w = image_shape[:2]
    total_area = h * w
    
    # === STRATEGY 1: Standard contour finding ===
    # Dilate edges progressively to close gaps
    best_contour = None
    best_area = 0
    
    for dilation_size in [3, 5, 7, 9, 11]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        dilated = cv2.dilate(edge_map, kernel, iterations=2)
        
        # Close operation to fill gaps
        dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            # Check if this is a valid sticker-sized contour
            if area > total_area * 0.02 and area < total_area * 0.98:
                if area > best_area:
                    best_contour = largest
                    best_area = area
    
    # === STRATEGY 2: Flood fill from corners ===
    # If contour detection failed, try inverse approach
    if best_contour is None or best_area < total_area * 0.02:
        # Create a mask by flood-filling from corners (assuming background)
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        temp = edge_map.copy()
        
        # Flood fill from all corners
        corners = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
        for (cx, cy) in corners:
            if temp[cy, cx] == 0:  # If corner is not on an edge
                cv2.floodFill(temp, flood_mask, (cx, cy), 128)
        
        # The un-flooded area is the sticker
        sticker_region = (temp != 128).astype(np.uint8) * 255
        
        # Find contours in this region
        contours, _ = cv2.findContours(sticker_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > best_area and area > total_area * 0.02:
                best_contour = largest
                best_area = area
    
    # === STRATEGY 3: Convex hull as fallback ===
    # If contour is irregular, use its convex hull
    if best_contour is not None:
        hull = cv2.convexHull(best_contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(best_contour)
        
        # If the contour is very non-convex (lots of gaps), prefer hull
        convexity_ratio = contour_area / hull_area if hull_area > 0 else 0
        
        if convexity_ratio < 0.7:
            # Contour has too many gaps, blend with hull
            # Use morphological operations to smooth
            pass
    
    if best_contour is None:
        return np.zeros((h, w), dtype=np.uint8), EdgeConfidence.FAILED, False
    
    # Validate contour size
    if best_area < total_area * 0.01:
        return np.zeros((h, w), dtype=np.uint8), EdgeConfidence.FAILED, False
    
    if best_area > total_area * 0.99:
        return np.zeros((h, w), dtype=np.uint8), EdgeConfidence.LOW, False
    
    # Check contour quality
    perimeter = cv2.arcLength(best_contour, closed=True)
    circularity = 4 * np.pi * best_area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Approximate contour
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(best_contour, epsilon, closed=True)
    is_closed = len(approx) >= 3 and circularity > 0.05
    
    # Create filled boundary mask
    boundary_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(boundary_mask, [best_contour], -1, 255, -1)
    
    # === POST-PROCESSING ===
    # Fill holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    boundary_mask = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    
    # Smooth edges
    boundary_mask = cv2.GaussianBlur(boundary_mask, (5, 5), 0)
    _, boundary_mask = cv2.threshold(boundary_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Determine confidence
    coverage = best_area / total_area
    if is_closed and circularity > 0.2 and coverage > 0.05:
        confidence = EdgeConfidence.HIGH
    elif is_closed or circularity > 0.08:
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
    
    # === STEP 2: Edge Detection (with color info) ===
    edge_map = detect_edges_multipass(enhanced, img_array)
    
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
