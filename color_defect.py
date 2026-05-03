import cv2
import numpy as np



def detect_color_defect(image_bgr: np.ndarray):
    """
    Estimate color and illumination inconsistency.

    Returns:
        color_mask: binary mask for suspicious color regions
        color_defect_score: normalized severity score
        hsv: HSV image
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_v = float(np.mean(v))
    std_v = float(np.std(v))
    mean_s = float(np.mean(s))
    std_s = float(np.std(s))

    local_v = cv2.GaussianBlur(v, (21, 21), 0)
    deviation = cv2.absdiff(v, local_v)
    _, color_mask = cv2.threshold(deviation, 28, 255, cv2.THRESH_BINARY)

    color_defect_score = (0.6 * std_v / (mean_v + 1e-5)) + (0.4 * std_s / (mean_s + 1e-5))

    return color_mask, float(color_defect_score), hsv
