import cv2
import os

from utils.dsp_analysis import compute_fft


IMAGE_PATH = "test_images/chest_xray.jpg"

OUTPUT_PATH = "test_images/fft_spectrum.png"


image = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        f"Could not load: {IMAGE_PATH}"
    )


fft_image = compute_fft(image)

cv2.imwrite(
    OUTPUT_PATH,
    fft_image
)

print("FFT analysis completed.")
print("Output:", OUTPUT_PATH)
print("Image shape:", fft_image.shape)