import os
import sys
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import DRTwoStagePipeline
from src.preprocessing import preprocess_fundus_image

# Set Streamlit page configuration
st.set_page_config(
    page_title="RetinaAI - Diabetic Retinopathy 2-Stage Screening System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern biomedical aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    .badge-healthy {
        background-color: #065f46;
        color: #34d399;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-diseased {
        background-color: #991b1b;
        color: #fca5a5;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-severity {
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        text-align: center;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Load pipeline with caching
@st.cache_resource
def load_pipeline():
    stage1_path = os.path.join("models", "stage1_binary.pth")
    stage2_path = os.path.join("models", "stage2_severity.pth")
    return DRTwoStagePipeline(stage1_weights_path=stage1_path, stage2_weights_path=stage2_path)

pipeline = load_pipeline()

# Header Section
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("👁️ Retinal AI Diagnostic System")
    st.markdown("##### *Two-Stage Deep Learning Screening & Severity Grading for Diabetic Retinopathy*")

with col_status:
    s1_status = "🟢 Model Ready" if pipeline.stage1_loaded else "🟡 Pre-trained Backbone"
    s2_status = "🟢 Model Ready" if pipeline.stage2_loaded else "🟡 Pre-trained Backbone"
    st.markdown(f"""
    <div class="metric-card" style="padding: 10px; font-size: 0.85rem;">
        <b>Stage 1 (Binary):</b> {s1_status}<br>
        <b>Stage 2 (Severity):</b> {s2_status}
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Sidebar Setup
st.sidebar.header("🕹️ Control Panel")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Retinal Fundus Image", 
    type=["png", "jpg", "jpeg"],
    help="Upload a standard high-resolution retinal fundus photograph."
)

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Project Architecture")
st.sidebar.info("""
**Stage 1**: Binary Classifier (ResNet18)
*Healthy vs. DR Present*

**Stage 2**: Severity Classifier (EfficientNetB0)
*Mild, Moderate, Severe, Proliferative DR*

**Preprocessing**: OpenCV CLAHE Contrast Enhancement
""")

# Main Content Body
if uploaded_file is not None:
    # Read Image
    raw_image = Image.open(uploaded_file).convert("RGB")

    # Run Prediction Pipeline
    with st.spinner("Processing image (Applying CLAHE & running 2-Stage Neural Networks)..."):
        results = pipeline.predict(raw_image)

    # 1. Preprocessing Comparison Section
    st.subheader("🖼️ Image Preprocessing & Contrast Enhancement (CLAHE)")
    col_orig, col_clahe = st.columns(2)
    
    with col_orig:
        st.markdown("**Original Fundus Photograph**")
        st.image(raw_image, use_container_width=True)

    with col_clahe:
        st.markdown("**CLAHE Enhanced (Green/L-Channel Contrast)**")
        st.image(results["clahe_image_np"], use_container_width=True)

    st.divider()

    # 2. Diagnosis Results Section
    st.subheader("📊 Two-Stage AI Diagnostic Analysis")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Stage 1: Initial DR Screening")
        
        s1_label = results["stage1"]["label"]
        s1_conf = results["stage1"]["confidence"] * 100
        
        if results["stage1"]["class_id"] == 0:
            st.markdown(f'**Result**: <span class="badge-healthy">{s1_label}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'**Result**: <span class="badge-diseased">{s1_label}</span>', unsafe_allow_html=True)

        st.markdown(f"**Confidence**: `{s1_conf:.2f}%`")
        
        st.progress(results["stage1"]["confidence"])
        st.markdown('</div>', unsafe_allow_html=True)

    with res_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Stage 2: Severity Grade")

        if results["stage2"] is None:
            st.markdown('**Severity Level**: <span class="badge-healthy">N/A - Healthy Retina</span>', unsafe_allow_html=True)
            st.markdown("No secondary severity grading required.")
        else:
            s2_label = results["stage2"]["label"]
            s2_conf = results["stage2"]["confidence"] * 100
            badge_color = results["severity_badge_color"]
            
            st.markdown(f'**Severity**: <div class="badge-severity" style="background-color: {badge_color};">{s2_label}</div>', unsafe_allow_html=True)
            st.markdown(f"**Confidence**: `{s2_conf:.2f}%`")
            st.progress(results["stage2"]["confidence"])

        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Probability Distribution & Clinical Guidance
    if results["stage2"] is not None:
        st.subheader("📈 Severity Probability Distribution")
        probs_dict = results["stage2"]["probabilities"]
        
        prob_df = pd.DataFrame({
            "Severity Grade": list(probs_dict.keys()),
            "Probability (%)": [val * 100 for val in probs_dict.values()]
        })

        st.bar_chart(prob_df.set_index("Severity Grade"))

    st.subheader("🩺 Clinical Action Guidance & Explanations")
    st.info(f"**Final Diagnostic Verdict**: {results['final_diagnosis']}\n\n**Recommendation**: {results['clinical_guidance']}")

else:
    st.info("👈 Please upload a retinal fundus image from the sidebar to begin AI analysis.")
    
    # Showcase Sample Demo Card
    st.markdown("### How the 2-Stage Pipeline Works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        #### 1. CLAHE Contrast
        Automatically enhances retinal vascular features, microaneurysms, and cotton wool spots.
        """)
    with c2:
        st.markdown("""
        #### 2. Stage 1 Binary Check
        Determines whether the fundus shows healthy retina or presence of Diabetic Retinopathy.
        """)
    with c3:
        st.markdown("""
        #### 3. Stage 2 Severity
        If DR is present, classifies severity into **Mild, Moderate, Severe, or Proliferative DR**.
        """)
