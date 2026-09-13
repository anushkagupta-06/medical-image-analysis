"""
DSP Analysis Utilities

Contains frequency-domain analysis used in the
medical image processing pipeline.
"""

import cv2
import numpy as np


def compute_fft(image):
    """
    Compute the 2D FFT magnitude spectrum.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image.

    Returns
    -------
    magnitude_spectrum : numpy.ndarray
        Normalized FFT magnitude spectrum.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if len(image.shape) != 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    # Convert to floating point
    image_float = np.float32(image)

    # 2D Fourier Transform
    fft = np.fft.fft2(image_float)

    # Shift low frequencies to the center
    fft_shifted = np.fft.fftshift(fft)

    # Magnitude spectrum
    magnitude = np.abs(fft_shifted)

    # Log scaling for visualization
    magnitude = np.log1p(magnitude)

    # Normalize to 0-255
    magnitude_spectrum = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return magnitude_spectrum.astype(np.uint8)