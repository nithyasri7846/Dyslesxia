import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import to_categorical
import efficientnet.tfkeras as efn

# ==========================
# CONFIG
# ==========================

IMG_SIZE = 224

# Your dataset must look like:
# dataset/no_dyslexia/
# dataset/mild_dyslexia/
# dataset/high_dyslexia/
DATASET_DIR = "dataset"

CLASSES = ["no_dyslexia", "mild_dyslexia", "high_dyslexia"]
NUM_CLASSES = len(CLASSES)

MODEL_SAVE_PATH = "models_store/hybrid_model.h5"

# ==========================
# HANDCRAFTED FEATURES
# ==========================

def extract_handcrafted_features(path):
    img = cv2.imread(path, 0)

    # 1. Stroke width via edge intensity
    edges = cv2.Canny(img, 30, 150)
    stroke_width = edges.mean()

    # 2. Skew angle detection
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1] if coords.shape[0] > 0 else 0

    # 3. Writing pressure
    intensity_var = img.var()

    return np.array([stroke_width, angle, intensity_var], dtype=np.float32)

# ==========================
# IMAGE PREPROCESSING
# ==========================

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

# ==========================
# LOAD DATASET
# ==========================

def load_dataset():
    images = []
    features = []
    labels = []

    for label_idx, class_name in enumerate(CLASSES):
        folder = os.path.join(DATASET_DIR, class_name)

        print(f"Loading: {folder}")

        for file in glob(folder + "/*"):
            try:
                img = preprocess_image(file)
                feat = extract_handcrafted_features(file)

                images.append(img)
                features.append(feat)
                labels.append(label_idx)
            except Exception as e:
                print("Error loading:", file, e)

    images = np.array(images)
    features = np.array(features)
    labels = to_categorical(labels, NUM_CLASSES)

    return images, features, labels

# ==========================
# BUILD HYBRID MODEL
# ==========================

def build_hybrid_model():
    # EfficientNetB0 backbone
    effnet = efn.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    effnet.trainable = False  # Freeze layers

    # Inputs
    img_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    feat_input = layers.Input(shape=(3,))

    # Deep features
    deep_features = effnet(img_input)

    # Fusion
    x = layers.Concatenate()([deep_features, feat_input])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=[img_input, feat_input], outputs=output)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ==========================
# TRAINING
# ==========================

print("\n======================")
print("Loading Dataset...")
print("======================\n")

X_img, X_feat, y = load_dataset()

print("\nDataset Loaded!")
print("Images Shape:", X_img.shape)
print("Handcrafted Shape:", X_feat.shape)
print("Labels Shape:", y.shape)

# Split dataset
X_img_train, X_img_val, X_feat_train, X_feat_val, y_train, y_val = train_test_split(
    X_img, X_feat, y, test_size=0.2, random_state=42
)

# Build model
model = build_hybrid_model()
model.summary()

print("\n======================")
print("Training Model...")
print("======================\n")

history = model.fit(
    [X_img_train, X_feat_train],
    y_train,
    validation_data=([X_img_val, X_feat_val], y_val),
    epochs=20,
    batch_size=16
)

# Save model
os.makedirs("models_store", exist_ok=True)
model.save(MODEL_SAVE_PATH)

print("\n=====================================")
print("TRAINING COMPLETE!")
print("Model saved to:", MODEL_SAVE_PATH)
print("=====================================\n")
