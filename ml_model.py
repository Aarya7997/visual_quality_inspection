import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from preprocess import preprocess_image
from crack_detector import detect_cracks
from color_defect import detect_color_defect



def _safe_hist(channel: np.ndarray, bins: int = 16):
    hist = cv2.calcHist([channel], [0], None, [bins], [0, 256]).flatten()
    hist = hist / (np.sum(hist) + 1e-6)
    return hist.tolist()



def extract_features(image: np.ndarray):
    """
    Handcrafted feature vector for industrial visual inspection.
    This turns the project into an actual ML pipeline instead of pure thresholding.
    """
    original_bgr, gray, blur, clahe_gray = preprocess_image(image)
    crack_mask, crack_ratio, edge_density = detect_cracks(clahe_gray)
    color_mask, color_defect_score, hsv = detect_color_defect(original_bgr)

    glcm = graycomatrix(
        clahe_gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )

    contrast = float(graycoprops(glcm, "contrast")[0, 0])
    homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
    energy = float(graycoprops(glcm, "energy")[0, 0])
    correlation = float(graycoprops(glcm, "correlation")[0, 0])

    lbp = local_binary_pattern(clahe_gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = (lbp_hist / (lbp_hist.sum() + 1e-6)).tolist()

    h, s, v = cv2.split(hsv)

    features = [
        float(np.mean(gray)),
        float(np.std(gray)),
        float(np.mean(blur)),
        float(np.std(blur)),
        crack_ratio,
        edge_density,
        color_defect_score,
        contrast,
        homogeneity,
        energy,
        correlation,
        float(np.mean(h)),
        float(np.std(h)),
        float(np.mean(s)),
        float(np.std(s)),
        float(np.mean(v)),
        float(np.std(v)),
        float(np.count_nonzero(color_mask) / color_mask.size),
    ]

    features.extend(_safe_hist(gray, bins=16))
    features.extend(lbp_hist)

    metrics = {
        "crack_ratio": crack_ratio,
        "edge_density": edge_density,
        "color_defect_score": color_defect_score,
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": correlation,
        "color_mask_ratio": float(np.count_nonzero(color_mask) / color_mask.size),
    }

    maps = {
        "original_bgr": original_bgr,
        "gray": gray,
        "clahe_gray": clahe_gray,
        "crack_mask": crack_mask,
        "color_mask": color_mask,
    }

    return np.array(features, dtype=np.float32), metrics, maps



def train_anomaly_model(images, model_type: str = "Isolation Forest", contamination: float = 0.15):
    if len(images) < 3:
        raise ValueError("Upload at least 3 reference good images to train the ML model.")

    feature_bank = []
    for img in images:
        feats, _, _ = extract_features(img)
        feature_bank.append(feats)

    X = np.vstack(feature_bank)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "One-Class SVM":
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=max(0.05, contamination))
    else:
        model = IsolationForest(
            n_estimators=250,
            contamination=contamination,
            random_state=42,
        )

    model.fit(X_scaled)

    return {
        "scaler": scaler,
        "model": model,
        "model_type": model_type,
        "feature_dim": X.shape[1],
        "training_samples": X.shape[0],
    }



def predict_anomaly(model_bundle, image: np.ndarray):
    feats, metrics, maps = extract_features(image)
    scaled = model_bundle["scaler"].transform(feats.reshape(1, -1))

    prediction = int(model_bundle["model"].predict(scaled)[0])

    # Normalize score to 0-1 where larger means more anomalous.
    if hasattr(model_bundle["model"], "score_samples"):
        raw = float(model_bundle["model"].score_samples(scaled)[0])
        anomaly_score = float(1 / (1 + np.exp(raw)))
    else:
        raw = float(model_bundle["model"].decision_function(scaled)[0])
        anomaly_score = float(1 / (1 + np.exp(raw)))

    label = "Anomalous" if prediction == -1 else "Normal"

    return {
        "features": feats,
        "metrics": metrics,
        "maps": maps,
        "prediction": label,
        "anomaly_score": anomaly_score,
        "raw_score": raw,
    }
