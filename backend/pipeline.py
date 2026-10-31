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

from model_loader import get_model, get_model_and_meta, get_transform, get_device_instance
from mask_utils import guided_filter
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
    analysis: dict = None


# =============================================================================
# PICTURE METROLOGY & DEEP ANALYSIS
# =============================================================================

def analyze_image_metrics(
    image: Image.Image,
    mask: np.ndarray,
    alpha_mask: np.ndarray,
    edge_confidence: EdgeConfidence
) -> dict:
    """
    Perform deep metrological and qualitative picture analysis for sticker creation.
    """
    w, h = image.size
    total_pixels = w * h
    megapixels = round(total_pixels / 1_000_000, 2)
    
    # 1. Print Dimensions at 300 DPI (crisp print) & 150 DPI (large print)
    print_w_300_in = round(w / 300, 2)
    print_h_300_in = round(h / 300, 2)
    print_w_300_cm = round(print_w_300_in * 2.54, 2)
    print_h_300_cm = round(print_h_300_in * 2.54, 2)

    print_w_150_in = round(w / 150, 2)
    print_h_150_in = round(h / 150, 2)
    
    # 2. Aspect Ratio Category
    ratio = w / h if h > 0 else 1.0
    if 0.9 <= ratio <= 1.1:
        aspect_category = "Square"
    elif 1.1 < ratio <= 1.5:
        aspect_category = "Landscape"
    elif ratio > 1.5:
        aspect_category = "Panoramic"
    elif 0.67 <= ratio < 0.9:
        aspect_category = "Portrait"
    else:
        aspect_category = "Tall Vertical"
        
    # 3. Subject Coverage Ratio
    fg_pixels = int(np.sum(mask > 127))
    coverage_ratio = round((fg_pixels / max(total_pixels, 1)) * 100, 1)
    
    # 4. Edge Sharpness Score (Laplacian variance on original grayscale image)
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = min(100, int(np.log1p(laplacian_var) * 15))
    
    # 5. Contrast & Colorfulness
    bg_pixels = mask <= 127
    fg_pixels_mask = mask > 127
    
    if np.any(fg_pixels_mask) and np.any(bg_pixels):
        fg_mean = img_array[fg_pixels_mask].mean(axis=0)
        bg_mean = img_array[bg_pixels].mean(axis=0)
        color_dist = float(np.linalg.norm(fg_mean - bg_mean))
        contrast_score = min(100, int((color_dist / 441.67) * 100))
    else:
        contrast_score = 50
        
    # Colorfulness metric (Hasler and Süsstrunk)
    (R, G, B) = cv2.split(img_array.astype(np.float32))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    std_rg, mean_rg = float(np.std(rg)), float(np.mean(rg))
    std_yb, mean_yb = float(np.std(yb)), float(np.mean(yb))
    std_rgyb = np.sqrt((std_rg ** 2) + (std_yb ** 2))
    mean_rgyb = np.sqrt((mean_rg ** 2) + (mean_yb ** 2))
    colorfulness = float(std_rgyb + (0.3 * mean_rgyb))
    colorfulness_rating = "Vibrant" if colorfulness > 35 else ("Moderate" if colorfulness > 15 else "Muted")

    # 6. Recommended Border Thickness
    min_dim = min(w, h)
    rec_border_px = max(6, min(45, int(min_dim * 0.025)))

    # 7. Overall Sticker Quality Score (0 to 100)
    confidence_weight = {"high": 40, "medium": 30, "low": 15, "failed": 5}.get(edge_confidence.value, 20)
    resolution_score = min(30, int((w * h / (1024 * 1024)) * 30))
    quality_score = min(99, max(25, confidence_weight + resolution_score + int(sharpness_score * 0.15) + int(contrast_score * 0.15)))

    if quality_score >= 85 and w >= 800:
        readiness = "Excellent (High DPI Print Ready)"
    elif quality_score >= 65:
        readiness = "Good (Standard Quality)"
    else:
        readiness = "Needs Review / Brush Touch-up"

    return {
        "dimensions": {"width": w, "height": h, "megapixels": megapixels, "aspect_ratio": round(ratio, 2), "aspect_category": aspect_category},
        "print_size_300dpi": {"width_in": print_w_300_in, "height_in": print_h_300_in, "width_cm": print_w_300_cm, "height_cm": print_h_300_cm},
        "print_size_150dpi": {"width_in": print_w_150_in, "height_in": print_h_150_in},
        "subject_coverage_pct": coverage_ratio,
        "edge_sharpness_score": sharpness_score,
        "contrast_score": contrast_score,
        "colorfulness": round(colorfulness, 1),
        "colorfulness_rating": colorfulness_rating,
        "recommended_border_px": rec_border_px,
        "quality_score": quality_score,
        "print_readiness": readiness,
    }


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
    """
    h, w = enhanced.shape
    
    v = np.median(enhanced)
    
    canny_sensitive = cv2.Canny(enhanced, max(0, v * 0.3), v * 0.7)
    canny_standard = cv2.Canny(enhanced, max(0, v * 0.5), v * 1.0)
    canny_conservative = cv2.Canny(enhanced, max(0, v * 0.7), v * 1.3)
    
    canny_combined = cv2.bitwise_or(canny_sensitive, canny_standard)
    canny_combined = cv2.bitwise_or(canny_combined, canny_conservative)
    
    sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_norm = np.uint8(np.clip(sobel_mag / sobel_mag.max() * 255, 0, 255)) if sobel_mag.max() > 0 else np.zeros_like(enhanced)
    _, sobel_binary = cv2.threshold(sobel_norm, 20, 255, cv2.THRESH_BINARY)
    
    scharr_x = cv2.Scharr(enhanced, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(enhanced, cv2.CV_64F, 0, 1)
    scharr_mag = np.sqrt(scharr_x**2 + scharr_y**2)
    scharr_norm = np.uint8(np.clip(scharr_mag / scharr_mag.max() * 255, 0, 255)) if scharr_mag.max() > 0 else np.zeros_like(enhanced)
    _, scharr_binary = cv2.threshold(scharr_norm, 20, 255, cv2.THRESH_BINARY)
    
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F, ksize=3)
    laplacian_abs = np.uint8(np.abs(laplacian))
    _, laplacian_binary = cv2.threshold(laplacian_abs, 15, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)
    _, morph_binary = cv2.threshold(morph_gradient, 15, 255, cv2.THRESH_BINARY)
    
    color_edges = np.zeros((h, w), dtype=np.uint8)
    if original_rgb is not None:
        for c in range(3):
            channel = original_rgb[:, :, c]
            channel_edges = cv2.Canny(channel, 30, 100)
            color_edges = cv2.bitwise_or(color_edges, channel_edges)
    
    combined = np.zeros((h, w), dtype=np.uint8)
    combined = cv2.bitwise_or(combined, canny_combined)
    combined = cv2.bitwise_or(combined, sobel_binary.astype(np.uint8))
    combined = cv2.bitwise_or(combined, scharr_binary.astype(np.uint8))
    combined = cv2.bitwise_or(combined, laplacian_binary.astype(np.uint8))
    combined = cv2.bitwise_or(combined, morph_binary.astype(np.uint8))
    combined = cv2.bitwise_or(combined, color_edges)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    combined = cv2.dilate(combined, kernel_dilate, iterations=1)
    
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
    h, w = image_shape[:2]
    total_area = h * w
    
    best_contour = None
    best_area = 0
    
    for dilation_size in [3, 5, 7, 9, 11]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        dilated = cv2.dilate(edge_map, kernel, iterations=2)
        dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            if total_area * 0.02 < area < total_area * 0.98:
                if area > best_area:
                    best_contour = largest
                    best_area = area
    
    if best_contour is None or best_area < total_area * 0.02:
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        temp = edge_map.copy()
        
        corners = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
        for (cx, cy) in corners:
            if temp[cy, cx] == 0:
                cv2.floodFill(temp, flood_mask, (cx, cy), 128)
        
        sticker_region = (temp != 128).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(sticker_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > best_area and area > total_area * 0.02:
                best_contour = largest
                best_area = area
    
    if best_contour is None:
        return np.zeros((h, w), dtype=np.uint8), EdgeConfidence.FAILED, False
    
    if best_area < total_area * 0.01:
        return np.zeros((h, w), dtype=np.uint8), EdgeConfidence.FAILED, False
    
    if best_area > total_area * 0.99:
        return np.zeros((h, w), dtype=np.uint8), EdgeConfidence.LOW, False
    
    perimeter = cv2.arcLength(best_contour, closed=True)
    circularity = 4 * np.pi * best_area / (perimeter * perimeter) if perimeter > 0 else 0
    
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(best_contour, epsilon, closed=True)
    is_closed = len(approx) >= 3 and circularity > 0.05
    
    boundary_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(boundary_mask, [best_contour], -1, 255, -1)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    boundary_mask = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    
    boundary_mask = cv2.GaussianBlur(boundary_mask, (5, 5), 0)
    _, boundary_mask = cv2.threshold(boundary_mask, 127, 255, cv2.THRESH_BINARY)
    
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

def _infer_soft_mask(model, input_tensor) -> np.ndarray:
    """Run one forward pass and return a HxW float32 mask in [0, 1]."""
    with torch.no_grad():
        output = model(input_tensor)

        if isinstance(output, (list, tuple)):
            target = output[-1]
        elif hasattr(output, 'logits'):
            target = output.logits
        elif hasattr(output, 'output'):
            target = output.output
        else:
            target = output

        if isinstance(target, (list, tuple)):
            target = target[-1]

        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)

        if float(target.min()) < 0.0 or float(target.max()) > 1.0:
            target = target.sigmoid()

    return target[0].squeeze().detach().cpu().float().numpy()


def segment_constrained(
    image: Image.Image,
    boundary_mask: np.ndarray,
    model_key: Optional[str] = None,
    resolution: int = 1024,
    tta: bool = False,
) -> Tuple[np.ndarray, bool]:
    """
    Run the segmentation model (tier-selected) and softly constrain its output
    to the geometric boundary. The returned mask keeps the model's continuous
    0-255 values so downstream matting can preserve soft edges.
    """
    model, meta = get_model_and_meta(model_key)
    device = get_device_instance()
    transform = get_transform(resolution, meta.get("norm", "imagenet"))

    original_size = image.size  # (width, height)

    try:
        model_dtype = next(model.parameters()).dtype
    except Exception:
        model_dtype = torch.float32

    input_tensor = transform(image).unsqueeze(0).to(device=device, dtype=model_dtype)

    pred = _infer_soft_mask(model, input_tensor)

    # Test-time augmentation: average the prediction with its mirror image.
    # Costs a second forward pass but suppresses stray specks on hard cases.
    if tta:
        flipped = torch.flip(input_tensor, dims=[3])
        pred_flip = _infer_soft_mask(model, flipped)
        pred = (pred + np.fliplr(pred_flip)) / 2.0

    ai_mask = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    ai_mask = cv2.resize(ai_mask, original_size, interpolation=cv2.INTER_LINEAR)

    # --- escape detection (did the AI leak outside the detected boundary?) ---
    ai_binary = (ai_mask > 127).astype(np.uint8) * 255
    if boundary_mask is not None and np.any(boundary_mask):
        boundary_binary = (boundary_mask > 127).astype(np.uint8) * 255
        escaped_pixels = cv2.bitwise_and(ai_binary, cv2.bitwise_not(boundary_binary))
        escape_ratio = np.sum(escaped_pixels > 0) / max(np.sum(ai_binary > 0), 1)
        ai_escaped = float(escape_ratio) > 0.05

        # Soft constraint: dilate the boundary a few px before intersecting so
        # the model's anti-aliased / hair pixels at the true edge survive.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        soft_boundary = cv2.dilate(boundary_binary, kernel, iterations=2)
        constrained_mask = cv2.bitwise_and(ai_mask, soft_boundary)
    else:
        ai_escaped = False
        constrained_mask = ai_mask

    return constrained_mask, ai_escaped


# =============================================================================
# STEP 5: MASK HARDENING (QUALITY CONTROL)
# =============================================================================

def harden_mask(
    mask: np.ndarray,
    original_image: np.ndarray,
    matting: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn the soft AI mask into a solid binary shape (for borders/analysis) and
    a soft alpha mask (for rendering).

    The previous implementation thresholded to binary and then faked a thin
    blur back at the edge, throwing away the model's real soft transition. This
    version keeps the model's soft values and refines them edge-awarely with a
    guided filter, so hair, glow and anti-aliasing survive.
    """
    soft = mask.astype(np.uint8)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # --- 1. Solid shape + speck removal -------------------------------------
    _, binary = cv2.threshold(soft, 127, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_med, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    binary_mask = np.zeros_like(binary)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = int(areas.max()) if len(areas) else 0
        # Keep the main shape and any sizeable detached parts (dots on an "i",
        # multi-part stickers); drop only genuine specks.
        threshold_area = max(80, int(max_area * 0.02))
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= threshold_area:
                binary_mask[labels == i] = 255
    else:
        binary_mask = binary

    # Gate: anything the speck filter removed must also vanish from the alpha.
    gate = cv2.dilate(binary_mask, kernel_small, iterations=2)
    soft = cv2.bitwise_and(soft, gate)

    # --- 2. Alpha -----------------------------------------------------------
    if not matting:
        # Firm, slightly anti-aliased edge from the clean binary shape.
        alpha_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (3, 3), 0.8)
        alpha_mask = np.clip(alpha_mask, 0, 255).astype(np.uint8)
        return binary_mask, alpha_mask

    # Edge-aware refinement: snap the soft transition to real image edges.
    if original_image.ndim == 3:
        guide = original_image
    else:
        guide = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    refined = guided_filter(guide, soft, r=8, eps=1e-3)

    # Mild S-curve: firm up confident interior/exterior, keep the mid soft.
    a = refined.astype(np.float32) / 255.0
    a = np.clip((a - 0.10) / 0.80, 0.0, 1.0)   # preserve faint hair above ~0.10
    a = a * a * (3.0 - 2.0 * a)                # smoothstep
    alpha_mask = np.clip(a * 255.0, 0, 255).astype(np.uint8)

    alpha_mask = cv2.bitwise_and(alpha_mask, gate)
    return binary_mask, alpha_mask


# =============================================================================
# STEP 6: FINAL OUTPUT GENERATION & MAIN PIPELINE
# =============================================================================

def apply_alpha_mask(image: Image.Image, alpha_mask: np.ndarray) -> Image.Image:
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    img_array = np.array(image)
    img_array[:, :, 3] = alpha_mask
    
    return Image.fromarray(img_array)


def check_boundary_continuity(mask: np.ndarray) -> bool:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) == 1


def check_mask_printable(mask: np.ndarray, min_coverage: float = 0.005) -> bool:
    coverage = np.sum(mask > 127) / mask.size
    return coverage >= min_coverage


def process_sticker(
    image: Image.Image,
    tier_config: Optional[dict] = None,
    refine: bool = False,
) -> ProcessingResult:
    warnings = []

    # Resolve the inference recipe. `tier_config` wins; `refine` is the legacy
    # fallback so old callers keep working.
    if tier_config is None:
        tier_config = {
            "model": None,
            "resolution": 2048 if refine else 1024,
            "matting": True,
            "tta": refine,
        }
    model_key = tier_config.get("model")
    resolution = int(tier_config.get("resolution", 1024))
    matting = bool(tier_config.get("matting", True))
    tta = bool(tier_config.get("tta", tier_config.get("refine", False)))

    # Cap source resolution to a sensible working size (headroom above the
    # inference resolution so detail survives the resize back).
    max_size = max(1024, min(resolution * 2, 3072))
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    enhanced, img_array = preprocess_for_edges(image)
    edge_map = detect_edges_multipass(enhanced, img_array)

    boundary_mask, edge_confidence, contour_closed = isolate_sticker_boundary(
        edge_map, img_array.shape
    )

    if edge_confidence == EdgeConfidence.FAILED:
        warnings.append("Could not detect sticker boundary - using full AI segmentation")
        boundary_mask = np.ones((img_array.shape[0], img_array.shape[1]), dtype=np.uint8) * 255

    if not contour_closed:
        warnings.append("Sticker boundary may have gaps - edge quality may vary")

    if edge_confidence == EdgeConfidence.LOW:
        warnings.append("Low edge confidence - manual review recommended")

    ai_mask, ai_escaped = segment_constrained(
        image, boundary_mask, model_key=model_key, resolution=resolution, tta=tta
    )

    if ai_escaped:
        warnings.append("AI segmentation was constrained to prevent background bleed")

    binary_mask, alpha_mask = harden_mask(ai_mask, img_array, matting=matting)
    transparent = apply_alpha_mask(image, alpha_mask)
    
    boundary_continuous = check_boundary_continuity(binary_mask)
    if not boundary_continuous:
        warnings.append("Sticker may have disconnected parts")
    
    is_printable = check_mask_printable(binary_mask)
    if not is_printable:
        warnings.append("Sticker area too small - may not be printable")
    
    # Deep image & sticker metrology analysis
    analysis = analyze_image_metrics(image, binary_mask, alpha_mask, edge_confidence)
    
    return ProcessingResult(
        original=image,
        mask=binary_mask,
        alpha_mask=alpha_mask,
        transparent=transparent,
        edge_confidence=edge_confidence,
        contour_closed=contour_closed,
        boundary_continuous=boundary_continuous,
        ai_escaped_contour=ai_escaped,
        warnings=warnings,
        analysis=analysis
    )


def save_results(result: ProcessingResult, output_dir: Path, job_id: str) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    transparent_path = output_dir / f"{job_id}_transparent.png"
    result.transparent.save(transparent_path, "PNG")
    paths["transparent"] = str(transparent_path)
    
    mask_path = output_dir / f"{job_id}_mask.png"
    Image.fromarray(result.mask).save(mask_path, "PNG")
    paths["mask"] = str(mask_path)
    
    alpha_path = output_dir / f"{job_id}_alpha.png"
    Image.fromarray(result.alpha_mask).save(alpha_path, "PNG")
    paths["alpha"] = str(alpha_path)
    
    original_path = output_dir / f"{job_id}_original.png"
    result.original.save(original_path, "PNG")
    paths["original"] = str(original_path)
    
    return paths

