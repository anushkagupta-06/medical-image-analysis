"""
Feature Extraction Module for Medical Image Analysis

Extracts:
- GLCM texture features
- Shape features
- Intensity/statistical features
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_glcm_features(img):
    """
    Extract GLCM texture features.
    """

    # Quantize image to 64 gray levels
    img_quantized = (img // 4).astype(np.uint8)

    glcm = graycomatrix(
        img_quantized,
        distances=[1, 3],
        angles=[
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ],
        levels=64,
        symmetric=True,
        normed=True
    )

    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation"
    ]

    features = {}

    for prop in properties:
        values = graycoprops(glcm, prop)

        features[f"glcm_{prop}_mean"] = float(np.mean(values))
        features[f"glcm_{prop}_std"] = float(np.std(values))

    return features


def extract_shape_features(img):
    """
    Extract shape features from the largest contour.
    """

    _, binary = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    features = {
        "shape_area": 0.0,
        "shape_perimeter": 0.0,
        "shape_circularity": 0.0,
        "shape_aspect_ratio": 0.0,
        "shape_solidity": 0.0,
        "shape_extent": 0.0,
        "shape_num_contours": len(contours)
    }

    if not contours:
        return features

    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)

    features["shape_area"] = float(area)
    features["shape_perimeter"] = float(perimeter)

    if perimeter > 0:
        features["shape_circularity"] = float(
            4 * np.pi * area / (perimeter ** 2)
        )

    x, y, w, h = cv2.boundingRect(largest)

    if h > 0:
        features["shape_aspect_ratio"] = float(w) / float(h)

    if w * h > 0:
        features["shape_extent"] = float(area) / float(w * h)

    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)

    if hull_area > 0:
        features["shape_solidity"] = float(area) / float(hull_area)

    return features


def extract_intensity_features(img):
    """
    Extract intensity/statistical features.
    """

    features = {
        "intensity_mean": float(np.mean(img)),
        "intensity_std": float(np.std(img)),
        "intensity_min": float(np.min(img)),
        "intensity_max": float(np.max(img)),
        "intensity_median": float(np.median(img)),
        "intensity_skewness": 0.0,
        "intensity_kurtosis": 0.0,
        "intensity_entropy": 0.0
    }

    mean_val = np.mean(img)
    std_val = np.std(img)

    if std_val > 0:
        features["intensity_skewness"] = float(
            np.mean(((img - mean_val) / std_val) ** 3)
        )

        features["intensity_kurtosis"] = float(
            np.mean(((img - mean_val) / std_val) ** 4) - 3
        )

    # Entropy
    hist = cv2.calcHist(
        [img],
        [0],
        None,
        [256],
        [0, 256]
    ).flatten()

    hist = hist / hist.sum()
    hist = hist[hist > 0]

    features["intensity_entropy"] = float(
        -np.sum(hist * np.log2(hist))
    )

    return features


def extract_all_features(img):
    """
    Extract and combine all features.
    """

    features = {}

    features.update(extract_glcm_features(img))
    features.update(extract_shape_features(img))
    features.update(extract_intensity_features(img))

    return features