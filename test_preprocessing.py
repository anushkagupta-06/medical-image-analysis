import cv2

from preprocessing.preprocess import preprocess_pipeline


image_path = "test_images/chest_xray.jpg"

results = preprocess_pipeline(image_path)

for name, image in results.items():

    print(
        f"{name:20s} "
        f"shape={image.shape} "
        f"dtype={image.dtype}"
    )

cv2.imwrite(
    "test_gaussian.png",
    results["gaussian_blur"]
)

cv2.imwrite(
    "test_clahe.png",
    results["clahe"]
)

cv2.imwrite(
    "test_sharpened.png",
    results["sharpened"]
)

print("\nPreprocessing test completed.")