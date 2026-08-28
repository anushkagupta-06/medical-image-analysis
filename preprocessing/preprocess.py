"""
Preprocessing Module for Medical Image Analysis

Handles grayscale conversion, filtering, and image enhancement.
"""

import cv2
import numpy as np

def load_image(path):
    """
    Load an image from disk.

    Returns:
        img_color: Original BGR image
        img_gray: Grayscale image
    """

    img_color = cv2.imread(path)

    if img_color is None:
        raise ValueError(f"Could not load image: {path}")

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    return img_color, img_gray

def load_tif_image(path):
    """
    Load a .tif medical image.

    Converts non-8-bit images to uint8.
    Returns both original and grayscale versions.
    """

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Could not load TIF image: {path}")

    if img.dtype != np.uint8:
        img = cv2.normalize(
            img,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    return img, gray

def apply_gaussian_blur(img, ksize=5):
    """
    Apply Gaussian blur for noise reduction.
    """

    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_median_filter(img, ksize=5):
    """
    Apply median filtering for noise removal.
    """

    return cv2.medianBlur(img, ksize)

def apply_bilateral_filter(
    img,
    d=9,
    sigma_color=75,
    sigma_space=75
):
    """
    Apply bilateral filtering while preserving edges.
    """

    return cv2.bilateralFilter(
        img,
        d,
        sigma_color,
        sigma_space
    )

def apply_histogram_equalization(img):
    """
    Apply standard histogram equalization.
    """

    return cv2.equalizeHist(img)

def apply_clahe(
    img,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    Apply Contrast Limited Adaptive Histogram Equalization.
    """

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(img)

def apply_unsharp_mask(
    img,
    sigma=1.0,
    strength=1.5
):
    """
    Apply unsharp masking for edge/detail enhancement.
    """

    blurred = cv2.GaussianBlur(
        img,
        (0, 0),
        sigma
    )

    sharpened = cv2.addWeighted(
        img,
        1.0 + strength,
        blurred,
        -strength,
        0
    )

    return sharpened

def denoise(img):
    """
    Apply lightweight bilateral denoising.
    """

    return cv2.bilateralFilter(
        img,
        5,
        50,
        50
    )

def preprocess_pipeline(path, is_tif=False):
    """
    Run the complete preprocessing pipeline.

    Returns:
        Dictionary containing all intermediate images.
    """

    if is_tif:
        img_original, img_gray = load_tif_image(path)
    else:
        img_original, img_gray = load_image(path)

    # Resize large images to limit memory usage
    max_dim = 1024

    h, w = img_gray.shape[:2]

    if max(h, w) > max_dim:

        scale = max_dim / max(h, w)

        new_w = int(w * scale)
        new_h = int(h * scale)

        img_gray = cv2.resize(
            img_gray,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        if len(img_original.shape) == 3:

            img_original = cv2.resize(
                img_original,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

        else:
            img_original = img_gray.copy()

    results = {
        "original": img_original,
        "grayscale": img_gray,

        "gaussian_blur":
            apply_gaussian_blur(img_gray),

        "median_filter":
            apply_median_filter(img_gray),

        "bilateral_filter":
            apply_bilateral_filter(img_gray),

        "denoised":
            denoise(img_gray),

        "histogram_eq":
            apply_histogram_equalization(img_gray),

        "clahe":
            apply_clahe(img_gray),

        "sharpened":
            apply_unsharp_mask(img_gray),
    }

    return results

