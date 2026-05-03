import io
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

from heatmap import generate_heatmap
from ml_model import predict_anomaly, train_anomaly_model, extract_features

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Industrial Inspection System",
    page_icon="🏭",
    layout="wide"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.big-title {
    font-size: 34px;
    font-weight: 700;
}
.section-card {
    background: rgba(15, 23, 42, 0.85);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(148,163,184,0.15);
}
.status-pass {color:#22c55e; font-weight:bold;}
.status-review {color:#f59e0b; font-weight:bold;}
.status-fail {color:#ef4444; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<p class="big-title">🏭 AI Visual Inspection System</p>', unsafe_allow_html=True)
st.caption("Industrial-grade defect detection using Computer Vision + Machine Learning")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Configuration")

model_type = st.sidebar.selectbox("Model", ["Isolation Forest", "One-Class SVM"])
contamination = st.sidebar.slider("Anomaly Sensitivity", 0.05, 0.4, 0.15)

reference_files = st.sidebar.file_uploader(
    "Upload GOOD reference images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.info("Step 1: Upload good samples\nStep 2: Upload test image")

# ---------------- MODEL BUILD ----------------
@st.cache_resource
def build_model(files):
    imgs = []
    for f in files:
        img = Image.open(io.BytesIO(f)).convert("RGB")
        imgs.append(np.array(img))
    return train_anomaly_model(imgs, model_type, contamination)

model_bundle = None
if reference_files:
    model_bundle = build_model([f.getvalue() for f in reference_files])
    st.sidebar.success(f"Model trained on {model_bundle['training_samples']} images")

# ---------------- IMAGE INPUT ----------------
uploaded = st.file_uploader("📸 Upload product image", type=["jpg","png","jpeg"])

if not uploaded:
    st.warning("Upload an image to start inspection")
    st.stop()

image = np.array(Image.open(uploaded).convert("RGB"))

# ---------------- PROCESSING ----------------
with st.spinner("🔍 Running AI inspection..."):
    time.sleep(1)

features, metrics, maps = extract_features(image)
heatmap, combined = generate_heatmap(
    maps["original_bgr"],
    maps["crack_mask"],
    maps["color_mask"]
)

# ---------------- ML ----------------
if model_bundle:
    result = predict_anomaly(model_bundle, image)
    anomaly_score = result["anomaly_score"]
    model_prediction = result["prediction"]
else:
    anomaly_score = 0.5
    model_prediction = "No model"

# ---------------- FINAL DECISION ----------------
def normalize(v, lo, hi):
    return np.clip((v - lo)/(hi-lo), 0, 1)

fused = (
    0.55 * anomaly_score +
    0.25 * normalize(metrics["crack_ratio"], 0.005, 0.05) +
    0.20 * normalize(metrics["color_defect_score"], 0.12, 0.7)
)

if fused < 0.35:
    status = "PASS"
elif fused < 0.6:
    status = "REVIEW"
else:
    status = "FAIL"

# ---------------- STATUS HEADER ----------------
color_map = {
    "PASS":"status-pass",
    "REVIEW":"status-review",
    "FAIL":"status-fail"
}

st.markdown(f"""
### 🔎 Inspection Result: <span class="{color_map[status]}">{status}</span>
- **Confidence Score:** {fused:.3f}
- **Model Output:** {model_prediction}
""", unsafe_allow_html=True)

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns([1.5,1])

# -------- LEFT: IMAGE --------
with col1:
    st.markdown("### 🖼 Inspection View")

    tab1, tab2 = st.tabs(["Original", "AI Heatmap"])

    with tab1:
        st.image(image, use_container_width=True)

    with tab2:
        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_container_width=True)

# -------- RIGHT: GAUGE --------
with col2:
    st.markdown("### 📊 Defect Severity")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fused*100,
        title={'text': "Risk Score"},
        gauge={
            'axis': {'range':[0,100]},
            'bar': {'color':"red"},
            'steps':[
                {'range':[0,35],'color':'green'},
                {'range':[35,60],'color':'orange'},
                {'range':[60,100],'color':'red'}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Explanation
    if status == "PASS":
        st.success("✔ Product meets quality standards.")
    elif status == "REVIEW":
        st.warning("⚠ Moderate defect detected. Manual inspection recommended.")
    else:
        st.error("❌ Severe defect detected!")

# ---------------- METRICS ----------------
st.markdown("### 📌 Key Metrics")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Anomaly Score", f"{anomaly_score:.3f}")
m2.metric("Crack Ratio", f"{metrics['crack_ratio']:.4f}")
m3.metric("Edge Density", f"{metrics['edge_density']:.4f}")
m4.metric("Color Score", f"{metrics['color_defect_score']:.4f}")
m5.metric("Texture", f"{metrics['glcm_contrast']:.2f}")

# ---------------- CHART ----------------
st.markdown("### 📊 Risk Breakdown")

chart_data = {
    "ML": anomaly_score,
    "Crack": metrics["crack_ratio"],
    "Color": metrics["color_defect_score"]
}
st.bar_chart(chart_data)

# ---------------- DEFECT MAPS ----------------
st.markdown("### 🧠 Defect Analysis")

c1, c2, c3 = st.columns(3)

with c1:
    st.image(maps["crack_mask"], caption="Cracks", use_container_width=True)

with c2:
    st.image(maps["color_mask"], caption="Color Defects", use_container_width=True)

with c3:
    st.image(combined, caption="Combined Map", use_container_width=True)

# ---------------- MODEL INFO ----------------
st.markdown("### 🤖 Model Insights")

if model_bundle:
    st.write(f"""
    - Model: {model_bundle['model_type']}
    - Training Samples: {model_bundle['training_samples']}
    - Feature Dimension: {model_bundle['feature_dim']}
    """)
else:
    st.info("Upload reference images to enable ML model")

# ---------------- PIPELINE ----------------
st.markdown("### 🔄 Inspection Pipeline")

st.write("""
1. Image Preprocessing  
2. Crack Detection (Edges + Morphology)  
3. Color Defect Detection (HSV + Deviation)  
4. Feature Extraction (GLCM + LBP + Stats)  
5. ML Prediction (Isolation Forest / SVM)  
6. Decision Fusion  
""")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Computer Vision + Machine Learning | Final Year Project")