import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("Brain Tumor Detection from MRI")
st.write("Upload an MRI image to predict whether a **tumor is present or not**.")

# -------------------------------
# Load Model (cached)
# -------------------------------


@st.cache_resource
def load_trained_model():
    model = load_model("brain_tumor_model.h5")
    return model


model = load_trained_model()

# -------------------------------
# Image Preprocessing
# -------------------------------


def preprocess_image(image, target_size=(224, 224)):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, target_size)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing MRI..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)

            # confidence = float(prediction[0][0])

            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            class_labels = ["No Tumor", "Tumor"]

            # Binary classification threshold
            # if confidence > 0.5:
            #     result = "🟩 No Tumor Detected"
            #     score = confidence
            # else:
            #     result = "🟥 Pitutory Tumor Detected"
            #     score = 1 - confidence

        st.subheader("Prediction Result")

        if predicted_class == 1:
            st.error("🟥 Tumor Detected")
        else:
            st.success("🟩 No Tumor Detected")

        st.write(f"**Confidence:** {confidence * 100:.2f}%")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit & TensorFlow")
