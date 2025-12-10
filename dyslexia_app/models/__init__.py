import tensorflow as tf
from tensorflow.keras.models import load_model

LABELS = ["No Dyslexia", "Mild Dyslexia", "High Dyslexia"]

def load_hybrid_model():
    model = load_model("models_store/hybrid_model.h5", compile=False)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def severity_from_probs(probs):
    weights = [0, 1, 2]
    return float((probs * weights).sum())
