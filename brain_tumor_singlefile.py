import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import datetime
import json

# tensorflow and sklearn imports
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
except Exception as e:
    print("ERROR: TensorFlow import failed. Please install tensorflow (e.g. pip install tensorflow).")
    raise e

try:
    from sklearn.metrics import classification_report, confusion_matrix
except Exception:
    print("Please install scikit-learn: pip install scikit-learn")
    raise

# -------------------------
# Small utilities
# -------------------------
def now():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def ensure_dir(path):
    if not path:
        return
    Path(path).mkdir(parents=True, exist_ok=True)

# -------------------------
# Data loader (tf.data)
# -------------------------
AUTOTUNE = tf.data.AUTOTUNE

def get_datasets(data_dir, img_size=(224,224), batch_size=16, val_split=0.2, seed=42):
    """
    Expects `data_dir` containing subfolders per class:
      data_dir/class_a/*.jpg
      data_dir/class_b/*.jpg
    Returns: train_ds, val_ds, class_names
    """
    data_dir = str(data_dir)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # training dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset='training'
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset='validation'
    )

    class_names = train_ds.class_names

    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x,y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x,y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

# -------------------------
# Model builders
# -------------------------
def build_custom_cnn(input_shape=(224,224,3), num_classes=2):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='custom_cnn')
    return model

def build_transfer_model(backbone_name='EfficientNetB0', input_shape=(224,224,3), num_classes=2, freeze_backbone=True):
    try:
        if backbone_name == 'EfficientNetB0':
            base = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
        elif backbone_name == 'MobileNetV2':
            base = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
        else:
            raise ValueError("Unsupported backbone")
    except Exception as e:
        # fallback to custom small CNN if weights can't be loaded
        print("Could not load pretrained backbone (internet/weights issue). Falling back to custom CNN. Error:", e)
        return build_custom_cnn(input_shape, num_classes)

    base.trainable = not freeze_backbone  # allow training if needed (default False)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(base.input, outputs, name=f"{backbone_name}_transfer")
    return model

# -------------------------
# Training routine
# -------------------------
def train(data_dir, out_model='models/best_model.h5', epochs=15, batch_size=16, lr=1e-4, backbone='EfficientNetB0'):
    ensure_dir(os.path.dirname(out_model) or '.')

    print(f"[{now()}] Preparing datasets from: {data_dir}")
    train_ds, val_ds, class_names = get_datasets(data_dir, img_size=(224,224), batch_size=batch_size)
    num_classes = len(class_names)
    print(f"[{now()}] Found classes: {class_names} (num_classes={num_classes})")

    # build model
    print(f"[{now()}] Building model (backbone={backbone}) ...")
    model = build_transfer_model(backbone_name=backbone, input_shape=(224,224,3), num_classes=num_classes, freeze_backbone=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    ckpt = ModelCheckpoint(out_model, monitor='val_loss', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    print(f"[{now()}] Starting training for {epochs} epochs ...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[ckpt, es, rl])
    # save final model as timestamped file too
    final_path = os.path.splitext(out_model)[0] + f"_{now()}.h5"
    model.save(final_path)
    print(f"[{now()}] Training complete. Best model saved to: {out_model} and final model saved to: {final_path}")

    # save class mapping
    mapping_path = os.path.splitext(out_model)[0] + "_class_map.json"
    with open(mapping_path, 'w') as f:
        json.dump({'class_names': class_names}, f)
    print(f"[{now()}] Class mapping saved to: {mapping_path}")

    return history, class_names

# -------------------------
# Evaluation routine
# -------------------------
def evaluate(model_path, data_dir, batch_size=16):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"[{now()}] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    train_ds, val_ds, class_names = get_datasets(data_dir, img_size=(224,224), batch_size=batch_size)
    y_true = []
    y_pred = []
    for x, y in val_ds:
        preds = model.predict(x)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    print(f"[{now()}] Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print(f"[{now()}] Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    return y_true, y_pred, class_names

# -------------------------
# Single image prediction
# -------------------------
def predict_image(model_path, image_path, class_map_path=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = tf.keras.models.load_model(model_path)

    # load class names if saved
    if class_map_path and os.path.exists(class_map_path):
        with open(class_map_path, 'r') as f:
            data = json.load(f)
            class_names = data.get('class_names', None)
    else:
        class_names = None

    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224,224))
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    label = class_names[idx] if class_names else str(idx)
    print(f"[{now()}] Predicted: {label} (index={idx}) with confidence {conf:.4f}")
    # print full distribution
    print("Full confidence distribution:")
    for i,p in enumerate(preds):
        name = class_names[i] if class_names else str(i)
        print(f"  {i:02d} - {name}: {p:.4f}")
    return idx, conf, preds

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Single-file Brain Tumor MRI classifier (train/eval/predict).")
    p.add_argument('--mode', type=str, required=True, choices=['train','eval','predict'], help='Action to perform')
    p.add_argument('--data_dir', type=str, default='./data', help='Path to data directory (for train/eval)')
    p.add_argument('--out_model', type=str, default='models/best_model.h5', help='Path to save trained model')
    p.add_argument('--model_path', type=str, default='models/best_model.h5', help='Path to model to load for eval/predict')
    p.add_argument('--class_map', type=str, default=None, help='Path to class mapping JSON (optional)')
    p.add_argument('--image_path', type=str, default=None, help='Image file for predict mode')
    p.add_argument('--epochs', type=int, default=12, help='Epochs for training')
    p.add_argument('--batch_size', type=int, default=16, help='Batch size')
    p.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    p.add_argument('--backbone', type=str, default='EfficientNetB0', help='Backbone: EfficientNetB0 or MobileNetV2 (or custom)')
    return p.parse_args()

# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    args = parse_args()

    # basic GPU info
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[{now()}] GPUs available: {len(physical_devices)}")
        except Exception as e:
            print("Could not set memory growth:", e)
    else:
        print(f"[{now()}] No GPU detected or TensorFlow can't see it. Running on CPU.")

    if args.mode == 'train':
        print(f"[{now()}] Mode: TRAIN")
        ensure_dir(os.path.dirname(args.out_model) or '.')
        train(args.data_dir, out_model=args.out_model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, backbone=args.backbone)

    elif args.mode == 'eval':
        print(f"[{now()}] Mode: EVALUATE")
        evaluate(args.model_path, args.data_dir, batch_size=args.batch_size)

    elif args.mode == 'predict':
        if not args.image_path:
            print("ERROR: --image_path is required in predict mode.")
            sys.exit(1)
        print(f"[{now()}] Mode: PREDICT")
        predict_image(args.model_path, args.image_path, class_map_path=args.class_map)

    else:
        print("Unknown mode. Choose train / eval / predict.")
