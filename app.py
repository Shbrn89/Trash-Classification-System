import os
import uuid
import cv2
import joblib
import numpy as np

from flask import Flask, render_template, request

from feature_extraction import extract_features_from_image, extract_object_cues


app = Flask(__name__)

MODEL_PATH = "model/trash_xgboost_model.pkl"
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model belum ditemukan di {MODEL_PATH}. "
        "Jalankan dulu: python train_model.py"
    )

model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
classes = model_package["classes"]


def allowed_file(filename):
    allowed_extensions = {"jpg", "jpeg", "png", "bmp", "webp"}

    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def make_prediction_variants(image):
    variants = []

    variants.append(image)
    variants.append(image)
    variants.append(cv2.convertScaleAbs(image, alpha=1.06, beta=8))
    variants.append(cv2.convertScaleAbs(image, alpha=0.94, beta=-8))
    variants.append(cv2.convertScaleAbs(image, alpha=1.12, beta=0))
    variants.append(cv2.convertScaleAbs(image, alpha=0.88, beta=0))
    variants.append(cv2.flip(image, 1))

    h, w = image.shape[:2]

    for angle in [-10, -5, 5, 10]:
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

        rotated = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            borderMode=cv2.BORDER_REFLECT
        )

        variants.append(rotated)

    for crop_ratio in [0.92, 0.86]:
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        y1 = max(0, (h - crop_h) // 2)
        x1 = max(0, (w - crop_w) // 2)
        cropped = image[y1:y1 + crop_h, x1:x1 + crop_w]

        if cropped.size > 0:
            variants.append(cv2.resize(cropped, (w, h)))

    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    sharpened = cv2.addWeighted(image, 1.35, blurred, -0.35, 0)
    variants.append(sharpened)

    return variants


def predict_with_confidence(image):
    variants = make_prediction_variants(image)

    feature_list = []

    for variant in variants:
        features = extract_features_from_image(variant)
        feature_list.append(features)

    feature_matrix = np.array(feature_list, dtype=np.float32)

    probabilities = model.predict_proba(feature_matrix)
    mean_probabilities = np.mean(probabilities, axis=0)

    sorted_indices = np.argsort(mean_probabilities)[::-1]

    pred_id = int(sorted_indices[0])
    prediction = classes[pred_id]
    confidence = float(mean_probabilities[pred_id]) * 100

    top_predictions = []

    for index in sorted_indices[:3]:
        top_predictions.append({
            "class_name": classes[index],
            "probability": round(float(mean_probabilities[index]) * 100, 2)
        })

    return prediction, confidence, top_predictions, mean_probabilities


def apply_domain_correction(image, prediction, confidence, top_predictions, probabilities):
    note = None

    if "plastic" not in classes:
        return prediction, confidence, top_predictions, note

    plastic_id = classes.index("plastic")
    plastic_probability = float(probabilities[plastic_id]) * 100

    cues = extract_object_cues(image)

    looks_vertical = (
        cues["aspect_ratio"] < 0.85 and
        cues["height_ratio"] > 0.50
    )

    looks_transparent = cues["transparent_ratio"] > 0.30
    has_blue_part = cues["blue_ratio"] > 0.012

    looks_like_plastic_bottle = looks_vertical and (looks_transparent or has_blue_part)

    wrong_material_prediction = prediction in ["paper", "cardboard", "metal"]
    uncertain_glass_prediction = (
        prediction == "glass" and
        plastic_probability >= 30 and
        confidence - plastic_probability <= 18
    )

    if (wrong_material_prediction or uncertain_glass_prediction) and looks_like_plastic_bottle and plastic_probability >= 10:
        prediction = "plastic"
        confidence = max(confidence, plastic_probability)

        note = (
            "Domain correction applied: the object has characteristics similar to a plastic bottle "
            "such as vertical shape, transparent body, or blue cap/label."
        )

        top_predictions = [
            {
                "class_name": "plastic",
                "probability": round(float(confidence), 2)
            }
        ] + [
            item for item in top_predictions
            if item["class_name"] != "plastic"
        ]

        top_predictions = top_predictions[:3]

    return prediction, confidence, top_predictions, note


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    top_predictions = []
    image_path = None
    error = None
    warning = None
    note = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "File gambar tidak ditemukan."

            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                top_predictions=top_predictions,
                image_path=image_path,
                error=error,
                warning=warning,
                note=note
            )

        file = request.files["image"]

        if file.filename == "":
            error = "Silakan pilih gambar terlebih dahulu."

            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                top_predictions=top_predictions,
                image_path=image_path,
                error=error,
                warning=warning,
                note=note
            )

        if not allowed_file(file.filename):
            error = "Format file tidak didukung. Gunakan JPG, JPEG, PNG, BMP, atau WEBP."

            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                top_predictions=top_predictions,
                image_path=image_path,
                error=error,
                warning=warning,
                note=note
            )

        extension = file.filename.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{extension}"
        saved_path = os.path.join(UPLOAD_FOLDER, filename)

        file.save(saved_path)

        image = cv2.imread(saved_path)

        if image is None:
            error = "Gambar tidak valid."

            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                top_predictions=top_predictions,
                image_path=image_path,
                error=error,
                warning=warning,
                note=note
            )

        prediction, confidence, top_predictions, probabilities = predict_with_confidence(image)

        prediction, confidence, top_predictions, note = apply_domain_correction(
            image,
            prediction,
            confidence,
            top_predictions,
            probabilities
        )

        if confidence < 45:
            warning = (
                "Confidence masih rendah. Hasil bisa kurang akurat. "
                "Gunakan gambar dengan objek jelas dan background sederhana."
            )

        image_path = saved_path.replace("\\", "/")

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        top_predictions=top_predictions,
        image_path=image_path,
        error=error,
        warning=warning,
        note=note
    )


if __name__ == "__main__":
    app.run(debug=True)
