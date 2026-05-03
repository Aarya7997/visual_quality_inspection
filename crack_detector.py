import cv2
import numpy as np



def detect_cracks(gray_image: np.ndarray):
    """
    Detect crack-like structures using edge extraction + morphology.

    Returns:
        crack_mask: binary mask
        crack_ratio: fraction of crack pixels
        edge_density: fraction of edge pixels before dilation
    """
    edges = cv2.Canny(gray_image, 60, 160)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    crack_mask = cv2.dilate(cleaned, kernel, iterations=1)

    crack_pixels = np.count_nonzero(crack_mask)
    total_pixels = crack_mask.size

    edge_density = np.count_nonzero(edges) / total_pixels
    crack_ratio = crack_pixels / total_pixels

    return crack_mask, float(crack_ratio), float(edge_density)
