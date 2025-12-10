import cv2
import numpy as np

def extract_handcrafted_features(path):
    """
    Extract handcrafted handwriting features such as:
    - Stroke width (edge density)
    - Skew angle of writing
    - Intensity variance (pressure-related)
    """

    # read grayscale
    img = cv2.imread(path, 0)

    # 1. Stroke Width Approximation (Canny edges mean)
    edges = cv2.Canny(img, 30, 150)
    stroke_width = edges.mean()

    # 2. Skew Angle (orientation of text)
    coords = np.column_stack(np.where(img > 0))
    if coords.shape[0] > 0:
        angle = cv2.minAreaRect(coords)[-1]
    else:
        angle = 0.0

    # 3. Intensity Variance (pressure)
    intensity_var = float(img.var())

    return np.array([stroke_width, angle, intensity_var], dtype="float32")
