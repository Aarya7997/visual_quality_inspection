# AI Visual Quality Inspection Project

This project upgrades a threshold-based visual inspection dashboard into a more presentation-ready machine-learning quality inspection demo.

## What is included

- Streamlit dashboard with a cleaner UI
- Crack and color defect visualization
- ML anomaly detection using:
  - Isolation Forest
  - One-Class SVM
- Hybrid decision logic using model score + interpretable defect metrics

## Folder structure

- `app.py` — Streamlit user interface
- `ml_model.py` — feature extraction and anomaly-model pipeline
- `preprocess.py` — preprocessing utilities
- `crack_detector.py` — crack-like edge extraction
- `color_defect.py` — color inconsistency analysis
- `heatmap.py` — combined defect heatmap overlay
- `requirements.txt` — recommended dependency versions

## Recommended setup

Use a clean virtual environment. This avoids package conflicts between NumPy and compiled packages such as pandas, scikit-image, or scikit-learn.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### If you still see a NumPy binary incompatibility error

Remove old compiled packages and reinstall from the pinned requirements:

```bash
pip uninstall -y numpy pandas scikit-learn scikit-image opencv-python
pip cache purge
pip install -r requirements.txt
```

## How to use

1. Upload **3 or more good reference images** in the sidebar.
2. Choose the anomaly model.
3. Upload one test image.
4. Review:
   - AI anomaly score
   - crack mask
   - color anomaly mask
   - combined heatmap
   - final PASS / REVIEW / FAIL result

## How to explain the ML part to your professor

- The original project used fixed image-processing thresholds.
- This upgraded version extracts visual features from the product image.
- It trains an anomaly model on normal samples.
- During testing, it estimates whether a new sample deviates from the learned normal distribution.
- The final decision is still interpretable because crack and color metrics are shown explicitly.

## Future DL extension

If you want to convert this into a deep-learning version later, the next step is to replace handcrafted features with:

- CNN embeddings from a pretrained backbone such as MobileNet
- or an autoencoder for unsupervised anomaly detection
