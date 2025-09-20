import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json, os

MODEL_PATH = r"D:\groundnut_leaf_disease_detection\models\best_model.h5"
CLASS_INDICES_PATH = r"D:\groundnut_leaf_disease_detection\models\class_indices.json"

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Load class labels dynamically
with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}  # reverse mapping

st.title("🌱 Groundnut Leaf Disease Detection")
st.write("Upload a groundnut leaf image and let AI predict the disease.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224,224))
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(img_array)

    pred_idx = np.argmax(prediction)
    pred_class = class_labels[pred_idx]
    confidence = float(np.max(prediction) * 100)

    st.success(f"✅ Predicted Class: **{pred_class}** ({confidence:.2f}%)")
