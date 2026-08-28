import cv2

from preprocessing.preprocess import preprocess_pipeline
from segmentation.segment import segment_pipeline


image_path = "test_images/chest_xray.jpg"

# Phase 2
preprocessed = preprocess_pipeline(image_path)

gray = preprocessed["grayscale"]
color = preprocessed["original"]

# Phase 3
results = segment_pipeline(
    gray,
    color
)

for name, value in results.items():

    if isinstance(value, dict):
        print(f"{name}:")
        print(value)

    else:
        print(
            f"{name:20s}"
            f" shape={value.shape}"
            f" dtype={value.dtype}"
        )


# Save important outputs
cv2.imwrite(
    "test_otsu.png",
    results["otsu_threshold"]
)

cv2.imwrite(
    "test_canny.png",
    results["canny_edges"]
)

cv2.imwrite(
    "test_sobel.png",
    results["sobel_edges"]
)

cv2.imwrite(
    "test_morphological.png",
    results["morphological"]
)

cv2.imwrite(
    "test_watershed.png",
    results["watershed"]
)

cv2.imwrite(
    "test_roi.png",
    results["roi_annotated"]
)

print("\nSegmentation test completed.")