import os
import cv2
from config import Config

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def preprocess_image(path, img_size=224):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    return img

def save_image(file, folder):
    filename = file.filename.replace(" ", "_")
    filepath = os.path.join(folder, filename)
    file.save(filepath)
    return filepath
