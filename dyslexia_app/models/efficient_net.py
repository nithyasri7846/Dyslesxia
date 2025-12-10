import efficientnet.tfkeras as efn
from tensorflow.keras import layers, Model

def get_efficientnet_base():
    base = efn.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(224, 224, 3)
    )
    base.trainable = False
    return base
