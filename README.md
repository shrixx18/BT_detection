# 🧠 Brain Tumor Detection from MRI

A machine learning project that detects brain tumors from MRI images using classical ML (SVM), a custom CNN, and transfer learning (MobileNetV2). Includes a Streamlit web app for real-time inference.

---

## 📌 Project Overview

This project classifies MRI brain scans into two categories:
- **No Tumor** (`notumor`)
- **Pituitary Tumor** (`pituitary`)

Three modeling approaches are compared:

| Model | Approach |
|---|---|
| SVM | Classical ML with hand-crafted features |
| Custom CNN | Deep learning from scratch |
| MobileNetV2 | Transfer learning (fine-tuned on ImageNet weights) |

---

## 🗂️ Dataset

- **Source**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle)
- **Classes**: `notumor`, `pituitary`
- **Image size**: Resized to **224 × 224** pixels (RGB)
- Training and testing folders are combined and re-split (80/20, stratified) to ensure balanced class representation.

---

## 🔬 Pipeline

### 1. Preprocessing
- **Gaussian Blur** (`3×3` kernel) — reduces noise in MRI scans
- **MobileNetV2 normalization** — scales pixel values to `[-1, 1]`

### 2. Image Augmentation
Applied only to training data to improve generalization:
- **Rotation** (±15°) — handles tumors at different orientations
- **Horizontal Flip** — improves spatial robustness
- **Zoom** (±10%) — simulates varying scan distances

### 3. Feature Extraction (for SVM)
Two complementary features are extracted per image:
- **Canny Edge Map** (resized to 32×32, flattened) — captures structural boundaries
- **Intensity Histogram** (64 bins) — captures texture information

These are concatenated into a single feature vector fed to the SVM.

---

## 🤖 Models

### Classical ML — SVM
- Kernel: **RBF**
- Input: hand-crafted feature vectors (edges + texture)
- Evaluated with accuracy score, classification report, and confusion matrix

### Custom CNN
```
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dropout(0.5) → Softmax(2)
```
- Optimizer: Adam | Loss: Categorical Crossentropy
- Trained for **5 epochs** with augmented batches

### Transfer Learning — MobileNetV2
- Pre-trained on ImageNet; last 30 layers fine-tuned
- Custom head: `Flatten → Dense(128, ReLU) → Dropout(0.5) → Softmax(2)`
- Optimizer: Adam (lr=0.0001) | Trained for **5 epochs**
- **This model is saved as `brain_tumor_model.h5` for deployment**

---

## 📊 Model Comparison

After training, the accuracy of all three models is printed side by side:

```
SVM:        xx.xx%
CNN:        xx.xx%
MobileNet:  xx.xx%
```

---

## 🔍 Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions of the MRI image the MobileNetV2 model focused on when making its prediction. This provides interpretability and helps verify the model is attending to medically relevant areas.

---

## 🌐 Streamlit Web App

The `app.py` file provides a browser-based interface for real-time tumor detection.

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Usage
1. Open the app in your browser (usually `http://localhost:8501`)
2. Upload a brain MRI image (`.jpg`, `.jpeg`, or `.png`)
3. Click **🔍 Predict**
4. View the result:
   - 🟩 **No Tumor Detected**
   - 🟥 **Tumor Detected**
5. Confidence score is displayed alongside the prediction

> The app loads `brain_tumor_model.h5` (the saved MobileNetV2 model) for inference.

---

## 📁 Project Structure

```
BT_detection/
├── HealthCare Assignment 2.ipynb   # Full ML pipeline (training, evaluation, Grad-CAM)
├── app.py                          # Streamlit web application
├── brain_tumor_model.h5            # Saved MobileNetV2 model weights
├── requirements.txt                # Python dependencies
└── runtime.txt                     # Python runtime version
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | CNN and MobileNetV2 training |
| scikit-learn | SVM, train-test split, metrics |
| OpenCV | Image loading, resizing, edge detection |
| Streamlit | Web app interface |
| NumPy / Matplotlib / Seaborn | Numerical ops and visualizations |

---

## ⚙️ Installation

```bash
git clone https://github.com/shrixx18/BT_detection.git
cd BT_detection
pip install -r requirements.txt
```

To reproduce training, open and run **`HealthCare Assignment 2.ipynb`** on [Kaggle](https://www.kaggle.com/) with the Brain Tumor MRI Dataset.

---

## 📄 License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.
