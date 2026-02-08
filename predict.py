# predict.py (robust loader)
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2

MODEL_PATH = "models/best_model.h5"
CLASS_MAP_PATH = "models/best_model_class_map.json"
INPUT_SHAPE = (224, 224, 3)

# load class names
if not os.path.exists(CLASS_MAP_PATH):
    raise FileNotFoundError("Class map not found: " + CLASS_MAP_PATH)
with open(CLASS_MAP_PATH, "r") as f:
    data = json.load(f)
    class_names = data.get("class_names") if isinstance(data, dict) else data
if class_names is None:
    raise ValueError("Could not read class names from class map.")

def build_transfer_backbone(input_shape=INPUT_SHAPE, num_classes=None, backbone_name="EfficientNetB0"):
    """Build a transfer model architecture without pretrained weights (weights=None)."""
    if num_classes is None:
        raise ValueError("num_classes required")
    if backbone_name == "EfficientNetB0":
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=input_shape)
    elif backbone_name == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=input_shape)
    else:
        # fallback to a small custom cnn if unknown backbone
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        return models.Model(inputs, outputs)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(base.input, outputs)

def try_load_model(path):
    """Try normal tf.keras.load_model first, otherwise rebuild architecture and load weights."""
    try:
        print("Trying tf.keras.models.load_model(...)")
        m = tf.keras.models.load_model(path)
        print("Loaded model with tf.keras.models.load_model")
        return m
    except Exception as e:
        print("load_model failed:", e)
        print("Falling back to rebuilding architecture & loading weights.")
    # Fallback: rebuild model architecture using class count and attempt to load weights
    num_classes = len(class_names)
    # Try EfficientNetB0 architecture first (common in this project)
    try:
        m = build_transfer_backbone(input_shape=INPUT_SHAPE, num_classes=num_classes, backbone_name="EfficientNetB0")
        # load weights (h5 may contain entire model; loading weights from it often works)
        m.load_weights(path)
        print("Loaded weights into rebuilt EfficientNetB0 model.")
        return m
    except Exception as e2:
        print("Failed to load weights into EfficientNet build:", e2)
    # Try custom small CNN architecture
    try:
        print("Trying small custom CNN architecture as fallback.")
        inputs = layers.Input(shape=INPUT_SHAPE)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        small = models.Model(inputs, outputs)
        small.load_weights(path)
        print("Loaded weights into small custom CNN model.")
        return small
    except Exception as e3:
        print("Final fallback failed:", e3)
    raise RuntimeError("Could not load model or weights from: " + path)

# MAIN
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found: " + MODEL_PATH)
    if not os.path.exists(image_path):
        raise FileNotFoundError("Image file not found: " + image_path)

    model = try_load_model(MODEL_PATH)
    # Preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Unable to read image: " + image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
    x = img.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    label = class_names[idx] if isinstance(class_names, (list,tuple)) else str(idx)
    print(f"\nPredicted class: {label}")
    print(f"Confidence: {preds[idx]:.4f}")
    print("Full distribution:", preds)
