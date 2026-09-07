import cv2

from models.feature_extractor import (
    extract_glcm_features,
    extract_shape_features,
    extract_intensity_features,
    extract_all_features
)


IMAGE_PATH = "test_images/chest_xray.jpg"


img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(
        f"Could not load test image: {IMAGE_PATH}"
    )


print("Image shape:", img.shape)


glcm_features = extract_glcm_features(img)
print("\nGLCM Features:")
for key, value in glcm_features.items():
    print(f"{key}: {value}")


shape_features = extract_shape_features(img)
print("\nShape Features:")
for key, value in shape_features.items():
    print(f"{key}: {value}")


intensity_features = extract_intensity_features(img)
print("\nIntensity Features:")
for key, value in intensity_features.items():
    print(f"{key}: {value}")


all_features = extract_all_features(img)

print("\nTotal Features:", len(all_features))
print("\nAll Features:")
for key, value in all_features.items():
    print(f"{key}: {value}")