# eval_fallback.py
import os
import json
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = "models/best_model.h5"
CLASS_MAP_PATH = "models/best_model_class_map.json"
IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

def load_class_names(class_map_path):
    with open(class_map_path, "r") as f:
        data = json.load(f)
    # support either {"class_names": [...]} or list
    if isinstance(data, dict):
        return data.get("class_names")
    return data

def build_transfer_backbone(input_shape=(224,224,3), num_classes=4, backbone_name="EfficientNetB0"):
    if backbone_name == "EfficientNetB0":
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=input_shape)
    elif backbone_name == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=input_shape)
    else:
        # small CNN fallback
        inp = layers.Input(shape=input_shape)
        x = layers.Conv2D(32,3,activation="relu",padding="same")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64,3,activation="relu",padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        out = layers.Dense(num_classes, activation="softmax")(x)
        return models.Model(inp, out)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(base.input, out)

def try_load_model(path, class_names, backbone_name="EfficientNetB0"):
    # try load_model first
    try:
        print("Trying tf.keras.models.load_model(...)")
        m = tf.keras.models.load_model(path)
        print("Loaded with load_model()")
        return m
    except Exception as e:
        print("load_model failed:", e)
        print("Falling back to rebuild+load_weights.")
    # rebuild and load weights
    num_classes = len(class_names)
    # try EfficientNet style
    try:
        m = build_transfer_backbone(input_shape=(IMG_SIZE[0],IMG_SIZE[1],3),
                                    num_classes=num_classes,
                                    backbone_name=backbone_name)
        m.load_weights(path)
        print("Loaded weights into rebuilt transfer model.")
        return m
    except Exception as e2:
        print("Failed to load weights into transfer model:", e2)
    # try small CNN
    try:
        m = build_transfer_backbone(input_shape=(IMG_SIZE[0],IMG_SIZE[1],3),
                                    num_classes=num_classes,
                                    backbone_name="__custom__")
        m.load_weights(path)
        print("Loaded weights into small CNN fallback.")
        return m
    except Exception as e3:
        print("Final fallback failed:", e3)
    raise RuntimeError("Unable to load model or weights from: " + path)

def make_dataset(data_dir, batch_size=16):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=False
    )
    norm = tf.keras.layers.Rescaling(1./255)
    ds = ds.map(lambda x,y: (norm(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def evaluate(model, ds, class_names):
    y_true = []
    y_pred = []
    for x,y in ds:
        preds = model.predict(x)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--class_map", default=CLASS_MAP_PATH)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--backbone", type=str, default="EfficientNetB0")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError("Model not found: " + args.model)
    if not os.path.exists(args.class_map):
        raise FileNotFoundError("Class map not found: " + args.class_map)
    class_names = load_class_names(args.class_map)
    print("Classes:", class_names)
    model = try_load_model(args.model, class_names, backbone_name=args.backbone)
    ds = make_dataset(args.data_dir, batch_size=args.batch_size)
    evaluate(model, ds, class_names)
