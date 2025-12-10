import os
import numpy as np
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for, flash
)
from flask_sqlalchemy import SQLAlchemy

from config import Config
from utils import allowed_file, preprocess_image, save_image
from features.extract_features import extract_handcrafted_features
from gradcam.grad_cam import generate_gradcam, overlay_gradcam
from models import load_hybrid_model, LABELS, severity_from_probs

import cv2

# ----------------- Flask & DB setup -----------------

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

# --------------- Database tables --------------------

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)

    results = db.relationship("TestResult", backref="user", lazy=True)

class TestResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    image_path = db.Column(db.String(255), nullable=False)
    gradcam_path = db.Column(db.String(255), nullable=True)

    predicted_label = db.Column(db.String(50), nullable=False)
    prob_no = db.Column(db.Float, nullable=False)
    prob_mild = db.Column(db.Float, nullable=False)
    prob_high = db.Column(db.Float, nullable=False)

    severity_score = db.Column(db.Float, nullable=False)  # 0-2 based on probs
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# --------------- Load model once --------------------

model = load_hybrid_model()

# --------------- Routes -----------------------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/test", methods=["GET", "POST"])
def test_page():
    if request.method == "POST":
        username = request.form.get("username")
        file = request.files.get("image")

        if not username:
            flash("Please enter a username.")
            return redirect(request.url)

        if not file or file.filename == "":
            flash("Please upload a handwriting image.")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Invalid file type.")
            return redirect(request.url)

        # get or create user
        user = User.query.filter_by(name=username).first()
        if user is None:
            user = User(name=username)
            db.session.add(user)
            db.session.commit()

        # save image
        img_path = save_image(file, Config.UPLOAD_FOLDER)

        # preprocessing
        img_arr = preprocess_image(img_path, Config.IMG_SIZE)
        X_img = np.expand_dims(img_arr, axis=0)

        # handcrafted features
        feats = extract_handcrafted_features(img_path)
        X_feat = np.expand_dims(feats, axis=0)

        # model prediction
        probs = model.predict([X_img, X_feat])[0]
        probs = probs.astype("float32")
        label_idx = int(np.argmax(probs))
        predicted_label = LABELS[label_idx]

        severity_score = severity_from_probs(probs)
        prob_no, prob_mild, prob_high = map(float, probs)

        # Grad-CAM
        cam = generate_gradcam(model, X_img, layer_name="top_conv")
        overlay = overlay_gradcam((img_arr * 255).astype("uint8"), cam)

        gradcam_filename = f"gradcam_{os.path.basename(img_path)}"
        gradcam_path = os.path.join(Config.GRADCAM_FOLDER, gradcam_filename)
        cv2.imwrite(gradcam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # Save result in DB
        result = TestResult(
            user_id=user.id,
            image_path=os.path.relpath(img_path, start="static"),
            gradcam_path=os.path.relpath(gradcam_path, start="static"),
            predicted_label=predicted_label,
            prob_no=prob_no,
            prob_mild=prob_mild,
            prob_high=prob_high,
            severity_score=severity_score,
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
    results = TestResult.query.filter_by(user_id=user.id).order_by(TestResult.created_at.desc()).all()

    if not results:
        avg_severity = None
        level = "No test data yet"
    else:
        avg_severity = sum(r.severity_score for r in results) / len(results)
        if avg_severity < 0.5:
            level = "No Dyslexia"
        elif avg_severity < 1.5:
            level = "Mild Dyslexia"
        else:
            level = "High Dyslexia"

    # Data for graph (dates vs severity)
    dates = [r.created_at.strftime("%Y-%m-%d") for r in results]
    scores = [r.severity_score for r in results]

    return render_template(
        "user_data.html",
        user=user,
        results=results,
        avg_severity=avg_severity,
        level=level,
        dates=dates,
        scores=scores,
    )

# --------------- CLI helper ------------------------

@app.cli.command("init-db")
def init_db():
    """flask init-db"""
    os.makedirs("instance", exist_ok=True)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.GRADCAM_FOLDER, exist_ok=True)
    db.create_all()
    print("Database initialized.")

if __name__ == "__main__":
    app.run(debug=True)
