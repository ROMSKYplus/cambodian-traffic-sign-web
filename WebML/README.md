# 🚦 TrafficSign AI: Cambodian Traffic Sign Recognition

An intelligent, interactive web application built with **Streamlit**, **PyTorch**, and **Scikit-learn** for classifying Cambodian traffic signs. This project combines unsupervised dimensionality reduction (**Principal Component Analysis - PCA**) with supervised classification (**Artificial Neural Network - ANN**).

---

## 📌 1. Project Overview

- **Academic Context:** University Machine Learning Final Project.
- **Problem Statement:** Classify tightly cropped road traffic sign images into one of 10 supported Cambodian traffic categories.
- **Core ML Strategy:**
  1. **Unsupervised Preprocessing (PCA):** Compresses high-dimensional 32×32 grayscale images ($1,024$ pixels) down to $50$ principal components, reducing input dimensions by **95.12%** while removing spatial redundancy and camera noise.
  2. **Supervised Classification (ANN):** A 3-layer fully connected Multilayer Perceptron ($50 \rightarrow 100 \rightarrow 50 \rightarrow 10$) trained with Cross-Entropy Loss to predict class logits.
- **Held-Out Test Accuracy:** **96.63%** on $1,127$ unseen test crops.

---

## 🛠️ 2. Technology Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Deep Learning Framework:** [PyTorch](https://pytorch.org/) (CPU-compatible inference)
- **Dimensionality Reduction:** [Scikit-learn](https://scikit-learn.org/) (PCA)
- **Computer Vision:** [OpenCV](https://opencv.org/) (`opencv-python-headless`) & [Pillow](https://pillow.readthedocs.io/)
- **Data & Numeric Computing:** [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Joblib](https://joblib.readthedocs.io/)
- **Testing:** [Pytest](https://pytest.org/)
- **Custom UI Styling:** Minimalist CSS design system (`assets/styles.css`)

---

## 📂 3. Repository Structure

```
WebML/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python package dependencies
├── README.md                   # Complete documentation & instructions
├── .gitignore                  # Git ignore rules
├── models/
│   ├── best_ann.pth            # Trained PyTorch ANN model state dictionary
│   └── pca_ann.pkl             # Fitted Scikit-learn PCA transformer (1024 -> 50)
├── config/
│   └── class_names.json        # Class metadata, IDs, categories, and descriptions
├── src/
│   ├── __init__.py             # Package initialization
│   ├── model.py                # PyTorch TrafficSignANN architecture and loader
│   ├── preprocessing.py        # 8-stage image preprocessing pipeline
│   └── prediction.py           # Inference engine and top-K prediction logic
├── assets/
│   ├── styles.css              # Custom minimalist CSS dashboard stylesheet
│   └── sample_images/          # Benchmark sign icons and dataset test crops
└── tests/
    ├── conftest.py             # Pytest environment configuration
    ├── test_preprocessing.py   # Unit tests for image processing & shapes
    └── test_prediction.py      # Unit tests for model loading & inference
```

---

## 🧠 4. Machine Learning Pipeline & Architecture

### Image Preprocessing Stages
When a user uploads or selects an image:
1. **Read Image:** Decodes image bytes into standard RGB/BGR format via OpenCV.
2. **Resize:** Standardizes dimensions to $32 \times 32$ pixels.
3. **Grayscale:** Converts RGB channels to a single intensity channel.
4. **Normalize:** Scales pixel intensities from $[0, 255]$ to $[0.0, 1.0]$.
5. **Flatten:** Reshapes $32 \times 32$ matrix into a $1,024$-element float32 1D vector.
6. **PCA Projection:** Applies the saved `PCA(n_components=50)` transformer without re-fitting.
7. **Tensor Conversion:** Converts $50$-dimensional array to a PyTorch float32 tensor.
8. **ANN Inference:** Executes forward pass under `torch.inference_mode()`.
9. **Softmax & Argmax:** Computes normalized confidence scores and ranks top predictions.

### Neural Network Architecture ($50 \rightarrow 100 \rightarrow 50 \rightarrow 10$)
```
Input (50 PCA features)
   │
   ▼
Linear(50, 100) + ReLU
   │
   ▼
Linear(100, 50) + ReLU
   │
   ▼
Linear(50, 10)
   │
   ▼
Output Logits (10 classes) ──► Softmax ──► Predicted Traffic Sign
```

---

## 🚦 5. Supported Traffic Sign Classes

The model was trained on 10 specific categories mapped as follows:

| Model Index | Original Class ID | Traffic Sign Name | Category |
|:---:|:---:|:---|:---|
| `0` | `0` | Pedestrian Crossing | Warning / Information |
| `1` | `1` | No Left Turn | Prohibitory |
| `2` | `2` | No Right Turn | Prohibitory |
| `3` | `4` | No Entry | Prohibitory |
| `4` | `5` | Speed Limit (20 km/h) | Regulatory / Speed |
| `5` | `7` | No Parking | Prohibitory |
| `6` | `8` | No Stopping | Prohibitory |
| `7` | `9` | No Overtaking | Prohibitory |
| `8` | `22` | Stop | Priority / Mandatory |
| `9` | `33` | Children Crossing | Warning |

---

## 📊 6. Experimental Performance Comparison

| Model Architecture | Input Features | Validation Accuracy | Status |
|:---|:---:|:---:|:---|
| **ANN Model A** | 1,024 (Raw Pixels) | 93.77% | Baseline |
| **ANN Model B** | 1,024 (Raw Pixels) | 92.25% | Baseline |
| **PCA + ANN (Final)** | **50 (PCA Components)** | **96.44%** | **Champion (Deployed)** |

**Held-out Test Evaluation:** **96.63% Accuracy** ($1,089 / 1,127$ correct crops).

---

## 🚀 7. Local Installation & Setup

### Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, or 3.13
- Git

### Step-by-Step Setup

1. **Clone or Navigate to the Repository:**
   ```bash
   cd /path/to/WebML
   ```

2. **Create a Virtual Environment:**
   - **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows:**
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Trained Model Files:**
   Ensure the following files are in the `models/` directory:
   - `models/best_ann.pth`
   - `models/pca_ann.pkl`

5. **Run the Test Suite:**
   ```bash
   pytest tests
   ```
   All 13 unit tests should pass.

6. **Launch the Streamlit Web Application:**
   ```bash
   streamlit run app.py
   ```
   The browser will automatically open at `http://localhost:8501`.

---

## 🌐 8. Deployment Instructions

### Deploying to Streamlit Community Cloud (Recommended & Free)
1. Push your repository to GitHub (include `models/best_ann.pth` and `models/pca_ann.pkl`).
   *(Both files are under 1 MB, so Git LFS is not required).*
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **"Create app"** and select:
   - **Repository:** `your-username/traffic-sign-web`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy!"**. Streamlit Cloud will install dependencies from `requirements.txt` and launch the web app.

---

## ⚠️ 9. Limitations & Educational Disclaimer

1. **Classification vs. Object Detection:** The model is an image classifier trained on **tightly cropped signs**. It does not perform multi-object detection or bounding box extraction on raw panoramic dashcam images.
2. **Crop Sensitivity:** Users should upload closely cropped signs. Background road clutter may lead to incorrect predictions.
3. **Out-of-Distribution Signs:** Any unseen sign (outside the 10 trained classes) will still be mapped to one of the 10 known classes.
4. **Safety Notice:** This system is created solely for university coursework and educational demonstration. It must not be deployed in real vehicular navigation or safety-critical road safety applications.

