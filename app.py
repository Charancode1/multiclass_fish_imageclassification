import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# ============================
# CONFIG
# ============================
MODEL_FILE = "best_model.h5"
GDRIVE_FILE_ID = "1wziwfQbD9frNreqAej-LQl0gtKPHl62z"  # Your file ID from Google Drive

# ============================
# DOWNLOAD MODEL FROM DRIVE
# ============================
if not os.path.exists(MODEL_FILE):
    with st.spinner("📥 Downloading model from Google Drive..."):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_FILE, quiet=False)

# ============================
# LOAD MODEL
# ============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_FILE)
    return model

model = load_model()

# ============================
# STREAMLIT UI
# ============================
st.title("🐟 Fish Species Classifier")
st.write("Upload an image of a fish to identify its species using our trained deep learning model.")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = image.resize((299, 299))  # Match model input size (e.g., InceptionV3)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_names = ["Class1", "Class2", "Class3", "Class4", "Class5"]  # <-- Replace with actual fish class names
    predicted_class = class_names[np.argmax(predictions)]

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {np.max(predictions) * 100:.2f}%")
