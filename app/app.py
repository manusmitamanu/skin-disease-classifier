import os, sys, json, io, base64
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

model = keras.models.load_model(os.path.join(PROJECT_ROOT, "models", "final_efficientnetb0.keras"))
config = json.load(open(os.path.join(PROJECT_ROOT, "models", "final_config.json")))
CLASSES = config["classes"]
MEL_THRESHOLD = config["mel_threshold"]
MEL_IDX = CLASSES.index("mel")

# OOD gate: load the training feature distribution
stats = np.load(os.path.join(PROJECT_ROOT, "models", "ood_stats.npz"))
MU, P = stats["mu"], stats["P"]
OOD_THR = config.get("ood_threshold")


def ood_distance(f):
    d = f - MU
    return float(np.sqrt(d @ P @ d))


FULL_NAMES = {
    "akiec": "Actinic keratoses", "bcc": "Basal cell carcinoma", "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma", "mel": "Melanoma", "nv": "Melanocytic nevus", "vasc": "Vascular lesion",
}

# Grad-CAM pieces (layer names verified from the saved model)
base = model.get_layer("efficientnetb0")
feat_model = keras.Model(base.input, base.get_layer("top_activation").output)
gap, drop, dense = (model.get_layer(n) for n in ["global_average_pooling2d", "dropout", "dense"])


def preprocess(data):
    img = Image.open(io.BytesIO(data)).convert("RGB")          # handles JPG, PNG, WEBP
    arr = tf.image.resize(np.array(img), (224, 224)).numpy()   # raw 0-255, as in training
    return arr.astype("float32")


def run(img):
    """Returns (probabilities, Grad-CAM map, pooled features)."""
    x = tf.convert_to_tensor(img[None])
    with tf.GradientTape() as tape:
        fmap = feat_model(x, training=False)
        tape.watch(fmap)
        pooled = gap(fmap)
        probs = dense(drop(pooled, training=False))
        score = probs[0, int(tf.argmax(probs[0]))]
    grads = tape.gradient(score, fmap)
    w = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(fmap * w[:, None, None, :], axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    return probs[0].numpy(), cv2.resize(cam, (224, 224)), pooled[0].numpy()


def overlay_b64(img, cam):
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    bgr = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2BGR)
    out = cv2.addWeighted(bgr, 0.6, heat, 0.4, 0)
    out = cv2.resize(out, (672, 672), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", out)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def risk_level(probs):
    # Our own grouping rule, not a model output. "Lower" does not mean safe.
    if probs[MEL_IDX] >= MEL_THRESHOLD:
        return "Elevated"
    if CLASSES[int(np.argmax(probs))] in ("bcc", "akiec"):
        return "Moderate"
    return "Lower"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("file")
    if not f or f.filename == "":
        return jsonify(error="No file uploaded."), 400
    try:
        img = preprocess(f.read())
    except Exception:
        return jsonify(error="Could not read that image."), 400

    # Run the model once: probabilities, Grad-CAM map and pooled features
    probs, cam, feat = run(img)

    # OOD gate: refuse images far from the HAM10000 feature distribution.
    # Imperfect in both directions (see 07_ood_gate.ipynb).
    dist = ood_distance(feat)
    if OOD_THR is not None and dist > OOD_THR:
        return jsonify(unsupported=True, distance=dist)

    top = int(np.argmax(probs))
    name = FULL_NAMES[CLASSES[top]]

    return jsonify(
        prediction=name,
        code=CLASSES[top],
        confidence=float(probs[top]),
        risk_level=risk_level(probs),
        description=(f"The model's highest-probability class is {name.lower()}. This is a statistical "
                     "pattern match from a model trained on dermatoscopic images, not a diagnosis."),
        mel_probability=float(probs[MEL_IDX]),
        mel_flag=bool(probs[MEL_IDX] >= MEL_THRESHOLD),
        threshold=MEL_THRESHOLD,
        low_confidence=bool(probs[top] < 0.5),
        distance=dist,
        probabilities=sorted(
            [{"code": c, "name": FULL_NAMES[c], "p": float(p)} for c, p in zip(CLASSES, probs)],
            key=lambda d: -d["p"]),
        gradcam=overlay_b64(img, cam),
    )


if __name__ == "__main__":
    app.run(debug=False)