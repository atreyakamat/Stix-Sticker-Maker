"""
Mask Utilities
Functions for mask manipulation, smoothing, and application
"""

import numpy as np
from PIL import Image
import cv2
from scipy import ndimage


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int = 4, eps: float = 1e-2) -> np.ndarray:
    """
    Guided Filter for alpha matting refinement.
    Uses the guide image to sharpen and smooth the mask edges based on local variance.
    """
    guide = guide.astype(np.float32) / 255.0
    src = src.astype(np.float32) / 255.0

    # If guide is RGB, use its grayscale version or handle channels
    if len(guide.shape) == 3:
        guide_gray = cv2.cvtColor((guide * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        guide_gray = guide

    mean_I = cv2.blur(guide_gray, (r, r))
    mean_p = cv2.blur(src, (r, r))
    mean_Ip = cv2.blur(guide_gray * src, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.blur(guide_gray * guide_gray, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.blur(a, (r, r))
    mean_b = cv2.blur(b, (r, r))

    q = mean_a * guide_gray + mean_b
    
    return (np.clip(q * 255, 0, 255)).astype(np.uint8)


def smooth_mask(mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to mask edges.
    Creates natural, soft edges.
    """
    if sigma <= 0:
        return mask
        
    # Apply soft gaussian blur
    smoothed = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    
    # We want to maintain 0 and 255 where possible but have a smooth transition
    return np.clip(smoothed, 0, 255).astype(np.uint8)


def clean_artifacts(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    Remove small disconnected regions (noise/artifacts)
    Keeps only the largest connected component
    """
    # Threshold
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    # Keep only components larger than min_area
    output = np.zeros_like(mask)
    for i in range(1, num_labels):  # Skip background (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            output[labels == i] = mask[labels == i]
    
    return output


def expand_mask(mask: np.ndarray, thickness: int) -> np.ndarray:
    """
    Expand mask outward by given thickness (for border generation).
    Uses distance transform for smooth, rounded expansion.
    """
    if thickness <= 0:
        return mask.copy()
    
    # 1. Start with binary mask
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 2. Use Distance Transform to find distance from every pixel to the nearest 0
    # Invert binary so sticker is 0 and background is 255
    inverted = cv2.bitwise_not(binary)
    dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
    
    # 3. Expansion: all pixels within 'thickness' distance of the original mask
    expanded = np.zeros_like(mask)
    expanded[dist < thickness] = 255
    
    # Ensure original mask is included and smoothed
    combined = cv2.bitwise_or(expanded, binary)
    
    return smooth_mask(combined, sigma=0.5)


def contract_mask(mask: np.ndarray, thickness: int) -> np.ndarray:
    """
    Contract mask inward by given thickness
    """
    if thickness <= 0:
        return mask.copy()
    
    # Threshold to binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Erode
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    contracted = cv2.erode(binary, kernel, iterations=thickness)
    
    return contracted


def apply_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Apply mask to image, creating transparent PNG
    """
    # Ensure RGBA
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Create output with alpha from mask
    img_array = np.array(image)
    
    # Set alpha channel from mask
    img_array[:, :, 3] = mask
    
    return Image.fromarray(img_array)


def create_border_layer(
    mask: np.ndarray,
    thickness: int,
    color: tuple
) -> Image.Image:
    """
    Create a border layer using mask expansion
    Returns RGBA image with just the border
    
    Algorithm:
    1. Expand original mask by thickness
    2. Subtract original mask to get border region
    3. Fill border region with color
    """
    if thickness <= 0:
        # Return empty transparent image
        h, w = mask.shape
        return Image.new('RGBA', (w, h), (0, 0, 0, 0))
    
    # Expand mask
    expanded = expand_mask(mask, thickness)
    
    # Get border region (expanded - original)
    _, original_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    border_region = cv2.subtract(expanded, original_binary)
    
    # Smooth the border edges
    border_region = smooth_mask(border_region, sigma=1.0)
    
    # Create RGBA image
    h, w = mask.shape
    border_image = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Fill with color where border exists
    border_image[:, :, 0] = color[0]  # R
    border_image[:, :, 1] = color[1]  # G
    border_image[:, :, 2] = color[2]  # B
    border_image[:, :, 3] = border_region  # Alpha from border mask
    
    return Image.fromarray(border_image)


def composite_with_border(
    sticker: Image.Image,
    mask: np.ndarray,
    border_thickness: int,
    border_color: tuple
) -> Image.Image:
    """
    Composite sticker with border
    Border layer goes behind sticker layer
    """
    # Ensure sticker is RGBA
    if sticker.mode != 'RGBA':
        sticker = sticker.convert('RGBA')
    
    # Create border layer
    border_layer = create_border_layer(mask, border_thickness, border_color)
    
    # Composite: border behind sticker
    result = Image.new('RGBA', sticker.size, (0, 0, 0, 0))
    result = Image.alpha_composite(result, border_layer)
    result = Image.alpha_composite(result, sticker)
    
    return result


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
