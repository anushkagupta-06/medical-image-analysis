"""
Dataset Exploratory Data Analysis

Generates:
1. Class distribution
2. Image resolution distribution
3. Intensity distribution
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)


def collect_images(root_dir):

    image_paths = []

    for root, _, files in os.walk(root_dir):

        for filename in files:

            if filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):
                image_paths.append(
                    os.path.join(root, filename)
                )

    return image_paths


def analyze_resolutions(image_paths):

    widths = []
    heights = []

    for path in image_paths:

        image = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        height, width = image.shape

        widths.append(width)
        heights.append(height)

    return widths, heights


def analyze_intensity(image_paths, sample_limit=500):

    intensity_values = []

    selected_paths = image_paths[:sample_limit]

    for path in selected_paths:

        image = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        # Sample pixels to avoid excessive memory usage
        pixels = image.flatten()

        if len(pixels) > 5000:

            pixels = np.random.choice(
                pixels,
                5000,
                replace=False
            )

        intensity_values.extend(pixels)

    return np.array(intensity_values)


def run_eda(dataset_dir):

    image_paths = collect_images(
        dataset_dir
    )

    print(
        f"Images found: {len(image_paths)}"
    )

    if not image_paths:
        print("No images found.")
        return

    # -------------------------------------------------
    # RESOLUTION ANALYSIS
    # -------------------------------------------------

    widths, heights = analyze_resolutions(
        image_paths
    )

    print(
        f"Average width: {np.mean(widths):.2f}"
    )

    print(
        f"Average height: {np.mean(heights):.2f}"
    )

    # -------------------------------------------------
    # RESOLUTION PLOT
    # -------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.scatter(
        widths,
        heights,
        alpha=0.4
    )

    plt.xlabel("Width")
    plt.ylabel("Height")

    plt.title(
        "Chest X-Ray Resolution Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        "resolution_distribution.png",
        dpi=200
    )

    plt.close()

    # -------------------------------------------------
    # INTENSITY ANALYSIS
    # -------------------------------------------------

    intensity = analyze_intensity(
        image_paths
    )

    print(
        f"Mean intensity: {np.mean(intensity):.2f}"
    )

    print(
        f"Intensity std: {np.std(intensity):.2f}"
    )

    # -------------------------------------------------
    # INTENSITY HISTOGRAM
    # -------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        intensity,
        bins=50
    )

    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.title(
        "Chest X-Ray Intensity Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        "intensity_distribution.png",
        dpi=200
    )

    plt.close()

    print(
        "EDA plots generated successfully."
    )


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage: python utils/eda.py <dataset_directory>"
        )

    else:

        run_eda(sys.argv[1])