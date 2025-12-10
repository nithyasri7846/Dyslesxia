import os
import numpy as np
import cv2
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy

from config import Config
from utils import allowed_file, preprocess_image, save_image
from features.extract_features import extract_handcrafted_features
from gradcam.grad_cam import generate_gradcam, overlay_gradcam
from models import load_hybrid_model, LABELS, severity_from_probs

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

# ---------------- DB MODELS ----------------

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True)

    results = db.relationship("TestResult", backref="user")

class TestResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))

    image_path = db.Column(db.String(255))
    gradcam_path = db.Column(db.String(255))

    predicted_label = db.Column(db.String(50))
    prob_no = db.Column(db.Float)
    prob_mild = db.Column(db.Float)
    prob_high = db.Column(db.Float)

    severity_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ---------------- LOAD MODEL ----------------

model = load_hybrid_model()

# ---------------- ROUTES ---------------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/test", methods=["GET", "POST"])
def test_page():
    if request.method == "POST":
        username = request.form.get("username")
        file = request.files.get("image")

        if not username:
            flash("Enter username")
            return redirect(request.url)

        if not file or file.filename == "":
            flash("Upload an image")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Invalid file")
            return redirect(request.url)

        # USER FETCH OR CREATE
        user = User.query.filter_by(name=username).first()
        if user is None:
            user = User(name=username)
            db.session.add(user)
            db.session.commit()

        path = save_image(file, Config.UPLOAD_FOLDER)

        img_arr = preprocess_image(path, Config.IMG_SIZE)
        X_img = np.expand_dims(img_arr, axis=0)

        feats = extract_handcrafted_features(path)
        X_feat = np.expand_dims(feats, axis=0)

        probs = model.predict([X_img, X_feat])[0]
        label_idx = np.argmax(probs)
        predicted_label = LABELS[label_idx]

        sever = severity_from_probs(probs)

        cam = generate_gradcam(model, X_img)
        overlay = overlay_gradcam((img_arr * 255).astype("uint8"), cam)

        gradcam_name = "gradcam_" + os.path.basename(path)
        gradcam_path = os.path.join(Config.GRADCAM_FOLDER, gradcam_name)

        cv2.imwrite(gradcam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        result = TestResult(
            user_id=user.id,
            image_path="uploads/" + os.path.basename(path),
            gradcam_path="gradcam/" + gradcam_name,
            predicted_label=predicted_label,
            prob_no=float(probs[0]),
            prob_mild=float(probs[1]),
            prob_high=float(probs[2]),
            severity_score=float(sever)
        )
        db.session.add(result)
        db.session.commit()

        return redirect(url_for("score_page", result_id=result.id))

    return render_template("test.html")

@app.route("/score/<int:result_id>")
def score_page(result_id):
    result = TestResult.query.get_or_404(result_id)
    return render_template("score.html", result=result)

@app.route("/user/<string:username>")
def user_data(username):
    user = User.query.filter_by(name=username).first_or_404()
    results = TestResult.query.filter_by(user_id=user.id).all()

    if results:
        avg = sum(r.severity_score for r in results) / len(results)
        level = "No Dyslexia" if avg < 0.5 else ("Mild Dyslexia" if avg < 1.5 else "High Dyslexia")
    else:
        avg = None
        level = "No data"

    return render_template("user_data.html", user=user, results=results, avg=avg, level=level)

@app.cli.command("init-db")
def init_db():
    os.makedirs("instance", exist_ok=True)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.GRADCAM_FOLDER, exist_ok=True)
    db.create_all()
    print("DB Ready!")

if __name__ == "__main__":
    app.run(debug=True)
