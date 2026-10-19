"""
Segmentation Module for Medical Image Analysis

Handles thresholding, edge detection, morphological
operations, watershed segmentation, and ROI detection.
"""

import cv2
import numpy as np

def binary_threshold(img, thresh=127):
    """
    Apply fixed binary thresholding.
    """

    _, result = cv2.threshold(
        img,
        thresh,
        255,
        cv2.THRESH_BINARY
    )

    return result

def otsu_threshold(img):
    """
    Apply Otsu's automatic thresholding.
    """

    _, result = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return result

def adaptive_threshold(img, block_size=11, c=2):
    """
    Apply adaptive Gaussian thresholding.
    """

    return cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )

def edge_detection_canny(img, low=50, high=150):
    """
    Apply Canny edge detection.
    """

    return cv2.Canny(
        img,
        low,
        high
    )

def edge_detection_sobel(img):
    """
    Apply Sobel edge detection using
    combined X and Y gradients.
    """

    sobelx = cv2.Sobel(
        img,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobely = cv2.Sobel(
        img,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    magnitude = np.sqrt(
        sobelx ** 2 + sobely ** 2
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    return magnitude

def morphological_operations(img_binary):
    """
    Apply morphological opening and closing
    to clean the binary segmentation.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    opened = cv2.morphologyEx(
        img_binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return closed

def watershed_segmentation(img_gray, img_color=None):
    """
    Apply watershed segmentation.

    Returns:
        Image with watershed boundaries.
    """

    if img_color is None:
        img_color = cv2.cvtColor(
            img_gray,
            cv2.COLOR_GRAY2BGR
        )
    else:
        img_color = img_color.copy()

    # Otsu threshold
    _, thresh = cv2.threshold(
        img_gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological opening
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    opening = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    # Sure background
    sure_bg = cv2.dilate(
        opening,
        kernel,
        iterations=3
    )

    # Distance transform
    dist_transform = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5
    )

    # Sure foreground
    _, sure_fg = cv2.threshold(
        dist_transform,
        0.5 * dist_transform.max(),
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv2.subtract(
        sure_bg,
        sure_fg
    )

    # Markers
    _, markers = cv2.connectedComponents(
        sure_fg
    )

    markers = markers + 1

    markers[unknown == 255] = 0

    # Watershed
    markers = cv2.watershed(
        img_color,
        markers
    )

    # Draw boundaries in red
    result = img_color.copy()

    result[markers == -1] = [
        0,
        0,
        255
    ]

    return result

def find_contours(img_binary):
    """
    Find external contours from a binary image.
    """

    contours, hierarchy = cv2.findContours(
        img_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours, hierarchy

def draw_contours_on_image(
    img,
    contours,
    color=(0, 255, 0),
    thickness=2
):
    """
    Draw contours on an image.
    """

    if len(img.shape) == 2:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR
        )

    result = img.copy()

    cv2.drawContours(
        result,
        contours,
        -1,
        color,
        thickness
    )

    return result

def detect_roi(img_gray, img_color=None):
    """
    Detect Region of Interest using
    CLAHE + Otsu + largest contour.

    Returns:
        roi_info
        binary mask
        annotated image
    """

    if img_color is None:
        img_color = cv2.cvtColor(
            img_gray,
            cv2.COLOR_GRAY2BGR
        )

    # CLAHE enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(img_gray)

    # Otsu threshold
    _, binary = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=3
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    # Find contours
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # No contour found
    if not contours:
        return (
            None,
            cleaned,
            img_color.copy()
        )

    # Largest contour
    largest = max(
        contours,
        key=cv2.contourArea
    )

    # Bounding box
    x, y, w, h = cv2.boundingRect(
        largest
    )

    # Annotated image
    annotated = img_color.copy()

    cv2.rectangle(
        annotated,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.drawContours(
        annotated,
        [largest],
        -1,
        (0, 0, 255),
        1
    )

    roi_info = {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area": cv2.contourArea(largest),
        "perimeter": cv2.arcLength(
            largest,
            True
        )
    }

    return (
        roi_info,
        cleaned,
        annotated
    )

def segment_pipeline(img_gray, img_color=None):
    """
    Run all segmentation methods
    and return results in a dictionary.
    """

    results = {

        "binary_threshold":
            binary_threshold(img_gray),

        "otsu_threshold":
            otsu_threshold(img_gray),

        "adaptive_threshold":
            adaptive_threshold(img_gray),

        "canny_edges":
            edge_detection_canny(img_gray),

        "sobel_edges":
            edge_detection_sobel(img_gray),
    }

    # Morphological cleanup
    results["morphological"] = \
        morphological_operations(
            results["otsu_threshold"]
        )

    # Watershed
    try:

        results["watershed"] = \
            watershed_segmentation(
                img_gray,
                img_color
            )

    except Exception:

        results["watershed"] = img_gray

    # ROI detection
    (
        roi_info,
        roi_mask,
        roi_annotated
    ) = detect_roi(
        img_gray,
        img_color
    )

    results["roi_mask"] = roi_mask

    results["roi_annotated"] = \
        roi_annotated

    results["roi_info"] = roi_info

    return results