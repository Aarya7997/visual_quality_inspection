import cv2
import numpy as np



def generate_heatmap(original_bgr: np.ndarray, crack_mask: np.ndarray, color_mask: np.ndarray):
    """
    Combine crack and color defect masks into a single heatmap overlay.
    """
    if crack_mask.shape != color_mask.shape:
        color_mask = cv2.resize(color_mask, (crack_mask.shape[1], crack_mask.shape[0]))

    combined = cv2.addWeighted(crack_mask, 0.7, color_mask, 0.3, 0)
    heatmap = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 0.65, heatmap, 0.35, 0)
    return overlay, combined
