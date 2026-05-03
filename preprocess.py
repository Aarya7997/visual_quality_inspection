import cv2
import numpy as np


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 BGR for OpenCV workflows."""
    if image is None:
        raise ValueError("Input image is None")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        # PIL images usually come in RGB; convert to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image



def preprocess_image(image: np.ndarray, size: int = 512):
    """
    Resize, denoise, grayscale conversion, and contrast enhancement.

    Returns:
        original_bgr: resized BGR image
        gray: grayscale image
        blur: denoised grayscale image
        clahe_gray: contrast-enhanced grayscale image
    """
    image_bgr = ensure_bgr(image)
    original_bgr = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(blur)

    return original_bgr, gray, blur, clahe_gray
