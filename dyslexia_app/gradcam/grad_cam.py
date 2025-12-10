import tensorflow as tf
import numpy as np
import cv2
from config import Config

def generate_gradcam(model, img_array, layer_name="top_conv"):
    """
    Generate a Grad-CAM heatmap for EfficientNet.
    img_array must be shape (1, 224, 224, 3)
    """

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs[0]

    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i, w in enumerate(pooled_grads):
        cam += w * conv_outputs[:, :, i]

    cam = cv2.resize(cam, (Config.IMG_SIZE, Config.IMG_SIZE))
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    return cam


def overlay_gradcam(rgb_img, cam, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the RGB handwriting image.
    rgb_img must be uint8 of shape (224,224,3)
    """

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    over = heatmap * alpha + rgb_img * (1 - alpha)

    return np.uint8(over)
