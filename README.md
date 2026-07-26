# 👁️ Diabetic Retinopathy 2-Stage AI Screening System

An AI-based diagnostic tool that automatically analyzes retinal fundus images to detect **Diabetic Retinopathy (DR)** and grade its severity (**No DR, Mild, Moderate, Severe, Proliferative DR**) using a two-stage transfer learning pipeline inspired by *Zafar et al. (2025)*.

---

## 🌟 Features
* **Two-Stage Deep Learning Pipeline**:
  * **Stage 1 (Binary Screening)**: Distinguishes between Healthy (No DR) and Diseased (DR Present) using a pre-trained `ResNet18` backbone.
  * **Stage 2 (Severity Grading)**: Grades diseased fundus images into **Mild, Moderate, Severe, or Proliferative DR** using a pre-trained `EfficientNetB0` backbone with class-weighted loss handling.
* **Image Preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization) contrast enhancement to highlight subtle microaneurysms and vascular features.
* **Streamlit Web Application**: Clean, responsive UI for image uploading, side-by-side CLAHE preview, confidence scores, and clinical recommendations.
* **Colab GPU Training Notebook**: Ready-to-run interactive Google Colab notebook for free training on Kaggle's APTOS 2019 dataset.

---

## 📁 Repository Structure
```
diabetic_retinopathy_app/
├── app.py                      # Streamlit Web Application UI
├── requirements.txt            # Python dependencies
├── README.md                   # Project setup guide
├── src/
│   ├── __init__.py             # Package initializer
│   ├── preprocessing.py        # CLAHE contrast enhancement & PyTorch transforms
│   ├── dataset.py              # Custom PyTorch DataLoader for Stage 1 & Stage 2
│   ├── models.py               # Transfer learning model definitions (ResNet / EfficientNet)
│   ├── train.py                # Model training loop & evaluation metrics
│   └── pipeline.py             # 2-Stage Inference Engine
├── notebooks/
│   └── train_colab.ipynb       # Google Colab training notebook
└── models/                     # Save trained .pth weight files here
    ├── stage1_binary.pth
    └── stage2_severity.pth
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone or navigate to the project directory and install requirements:
```bash
pip install -r requirements.txt
```

### 2. Run Web Application
Launch the Streamlit app:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Model Training on Google Colab (Free T4 GPU)
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `notebooks/train_colab.ipynb`.
3. Select **Runtime -> Change runtime type -> T4 GPU**.
4. Download your `kaggle.json` API token from Kaggle Account Settings and upload it when prompted in the notebook.
5. Run all cells to train both Stage 1 and Stage 2 models on the APTOS 2019 dataset.
6. The notebook will automatically download `stage1_binary.pth` and `stage2_severity.pth`. Place these weight files into the `models/` directory of this repo!

---

## 👥 Student Task Distribution (2-Person Engineering Team)

| Phase | Student A Tasks | Student B Tasks |
| :--- | :--- | :--- |
| **Weeks 1–3** | Image Preprocessing & CLAHE implementation (`src/preprocessing.py`). | Dataset EDA & PyTorch DataLoader setup (`src/dataset.py`). |
| **Weeks 4–6** | Stage 1 Model architecture (`src/models.py`) & Training setup. | Stage 1 Evaluation metrics & Confusion Matrix plotting (`src/train.py`). |
| **Weeks 7–10** | Stage 2 Model architecture & Class-weight optimization. | 2-Stage Inference Engine (`src/pipeline.py`) & End-to-end integration. |
| **Weeks 11–14** | Streamlit Web App development (`app.py`). | Project Report writing, Slides & Demo preparation. |
