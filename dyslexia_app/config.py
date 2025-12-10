import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = "secret-key"
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "instance", "dyslexia.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    IMG_SIZE = 224

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    GRADCAM_FOLDER = os.path.join(BASE_DIR, "static", "gradcam")

    NUM_CLASSES = 3
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
