import streamlit as st
import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow.keras import layers, models

MODEL_PATH = "models/best_model.h5"
CLASS_MAP_PATH = "models/best_model_class_map.json"
IMG_SIZE = (224, 224)

# Load class names
with open(CLASS_MAP_PATH, "r") as f:
    class_names = json.load(f)["class_names"]

def build_model(num_classes):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=(224,224,3)
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(base.input, output)
    model.load_weights(MODEL_PATH)
    return model

model = build_model(len(class_names))

st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image to classify the tumor type")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, width=700)


    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    if st.button("Predict"):
        preds = model.predict(img)[0]
        idx = np.argmax(preds)
        st.success(f"🧬 Predicted Tumor Type: **{class_names[idx]}**")
        st.info(f"🔢 Confidence: **{preds[idx]*100:.2f}%**")
