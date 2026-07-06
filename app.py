"""
BIHOS — Bangladesh Intelligent Healthcare Operating System
Flask Backend · app.py
Endpoints:
  GET  /status          — health check (model_loaded, model_info)
  POST /predict/xray    — Chest X-Ray RetinaNet detection
  POST /predict/skin    — Skin Disease 20-class Keras classification
"""

import os, io, base64, traceback, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"       # suppress TF info/warning logs
import tensorflow as tf

# ── App init ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# ── Skin 20-class label list (must match training order exactly) ───────────────
SKIN_CLASSES = [
    "Acne",
    "Actinic Carcinoma",
    "Atopic Dermatitis",
    "Bullous Disease",
    "Cellulitis",
    "Eczema",
    "Drug Eruptions",
    "Herpes HPV",
    "Light Diseases",
    "Lupus",
    "Melanoma",
    "Poison IVY",
    "Psoriasis",
    "Benign Tumors",
    "Systemic Disease",
    "Ringworm",
    "Urticarial Hives",
    "Vascular Tumors",
    "Vasculitis",
    "Viral Infections",
]

# ── Dental 3-class label list (must match training order exactly) ───────────────
DENTAL_CLASSES = [
    "Dental Caries / Cavities",
    "Missing Teeth",
    "Crowned Teeth / Dental Crowns",
]

# ── Nail 6-class label list (must match training order exactly) ────────────────
NAIL_CLASSES = [
    "Onychomycosis (Fungal Infection)",
    "Nail Psoriasis",
    "Paronychia (Bacterial/Fungal Inflammation)",
    "Melanonychia (Pigmentation/Stripes)",
    "Onycholysis (Nail Separation)",
    "Normal / Healthy Nail",
]

# ── Eye 4-class label list (must match training order exactly) ────────────────
EYE_CLASSES = [
    "Cataract",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Normal / Healthy Eye",
]

# ── Oral 6-class label list (must match training order exactly) ───────────────
ORAL_CLASSES = [
    "Oral Squamous Cell Carcinoma",
    "Leukoplakia",
    "Oral Lichen Planus",
    "Aphthous Ulcer (Canker Sore)",
    "Oral Candidiasis (Thrush)",
    "Normal / Healthy Oral Cavity",
]

# Severity colour per class (matches frontend JS)
CLASS_COLORS = {
    "Acne":              "#f59e0b",
    "Actinic Carcinoma": "#ef4444",
    "Atopic Dermatitis": "#f97316",
    "Bullous Disease":   "#8b5cf6",
    "Cellulitis":        "#ef4444",
    "Eczema":            "#f97316",
    "Drug Eruptions":    "#ec4899",
    "Herpes HPV":        "#6366f1",
    "Light Diseases":    "#06b6d4",
    "Lupus":             "#ef4444",
    "Melanoma":          "#dc2626",
    "Poison IVY":        "#22c55e",
    "Psoriasis":         "#f59e0b",
    "Benign Tumors":     "#10b981",
    "Systemic Disease":  "#6366f1",
    "Ringworm":          "#f59e0b",
    "Urticarial Hives":  "#f97316",
    "Vascular Tumors":   "#8b5cf6",
    "Vasculitis":        "#ef4444",
    "Viral Infections":  "#06b6d4",
}

# Dental class colours (matches frontend JS)
DENTAL_CLASS_COLORS = {
    "Dental Caries / Cavities":       "#0d9488",   # teal
    "Missing Teeth":                  "#6366f1",   # indigo
    "Crowned Teeth / Dental Crowns":  "#f59e0b",   # amber
}

# Nail class colours (matches frontend JS)
NAIL_CLASS_COLORS = {
    "Onychomycosis (Fungal Infection)":           "#f43f5e",   # rose
    "Nail Psoriasis":                             "#f59e0b",   # amber
    "Paronychia (Bacterial/Fungal Inflammation)": "#ef4444",   # red
    "Melanonychia (Pigmentation/Stripes)":        "#8b5cf6",   # violet
    "Onycholysis (Nail Separation)":              "#6366f1",   # indigo
    "Normal / Healthy Nail":                      "#10b981",   # emerald
}

# Eye class colours (matches frontend JS)
EYE_CLASS_COLORS = {
    "Cataract":             "#f59e0b",   # amber
    "Diabetic Retinopathy": "#ef4444",   # red
    "Glaucoma":             "#8b5cf6",   # violet
    "Normal / Healthy Eye": "#10b981",   # emerald
}

# Oral class colours (matches frontend JS)
ORAL_CLASS_COLORS = {
    "Oral Squamous Cell Carcinoma": "#dc2626",   # red-600 (critical)
    "Leukoplakia":                  "#f97316",   # orange-500
    "Oral Lichen Planus":           "#8b5cf6",   # violet-500
    "Aphthous Ulcer (Canker Sore)": "#eab308",   # yellow-500
    "Oral Candidiasis (Thrush)":    "#f43f5e",   # rose-500
    "Normal / Healthy Oral Cavity": "#10b981",   # emerald-500
}

# ── Model registry ────────────────────────────────────────────────────────────
models = {
    "skin":   {"model": None, "loaded": False, "error": None},
    "xray":   {"model": None, "loaded": False, "error": None},
    "dental": {"model": None, "loaded": False, "error": None},
    "nail":   {"model": None, "loaded": False, "error": None},
    "eye":    {"model": None, "loaded": False, "error": None},
    "oral":   {"model": None, "loaded": False, "error": None},
}

# ── Load skin model ───────────────────────────────────────────────────────────
def load_skin_model():
    skin_path = MODELS_DIR / "best_model.keras"
    if not skin_path.exists():
        models["skin"]["error"] = f"Model file not found: {skin_path}"
        print(f"[SKIN] ❌  {models['skin']['error']}")
        return
    try:
        print(f"[SKIN] Loading {skin_path} …")
        models["skin"]["model"] = tf.keras.models.load_model(str(skin_path))
        models["skin"]["loaded"] = True
        inp = models["skin"]["model"].input_shape
        print(f"[SKIN] [OK] Loaded. Input shape: {inp}")
    except Exception as e:
        models["skin"]["error"] = str(e)
        print(f"[SKIN] [FAIL] Load failed: {e}")
        traceback.print_exc()


# ── Load dental model ──────────────────────────────────────────────────────────
def load_dental_model():
    dental_path = MODELS_DIR / "best_dental_model.keras"
    if not dental_path.exists():
        models["dental"]["error"] = f"Model file not found: {dental_path}"
        print(f"[DENTAL] [FAIL]  {models['dental']['error']}")
        return
    try:
        print(f"[DENTAL] Loading {dental_path} …")
        models["dental"]["model"] = tf.keras.models.load_model(str(dental_path))
        models["dental"]["loaded"] = True
        inp = models["dental"]["model"].input_shape
        print(f"[DENTAL] [OK] Loaded. Input shape: {inp}")
    except Exception as e:
        models["dental"]["error"] = str(e)
        print(f"[DENTAL] [FAIL] Load failed: {e}")
        traceback.print_exc()


# ── Load nail model ────────────────────────────────────────────────────────────
def load_nail_model():
    nail_path = MODELS_DIR / "best_nail_model.keras"
    if not nail_path.exists():
        models["nail"]["error"] = f"Model file not found: {nail_path}"
        print(f"[NAIL] [FAIL]  {models['nail']['error']}")
        return
    try:
        print(f"[NAIL] Loading {nail_path} …")
        models["nail"]["model"] = tf.keras.models.load_model(str(nail_path))
        models["nail"]["loaded"] = True
        inp = models["nail"]["model"].input_shape
        print(f"[NAIL] [OK] Loaded. Input shape: {inp}")
    except Exception as e:
        models["nail"]["error"] = str(e)
        print(f"[NAIL] [FAIL] Load failed: {e}")
        traceback.print_exc()


# ── Load eye model ─────────────────────────────────────────────────────────────
def load_eye_model():
    eye_path = MODELS_DIR / "best_eye_model.keras"
    if not eye_path.exists():
        models["eye"]["error"] = f"Model file not found: {eye_path}"
        print(f"[EYE] [FAIL]  {models['eye']['error']}")
        return
    try:
        print(f"[EYE] Loading {eye_path} …")
        models["eye"]["model"] = tf.keras.models.load_model(str(eye_path))
        models["eye"]["loaded"] = True
        inp = models["eye"]["model"].input_shape
        print(f"[EYE] [OK] Loaded. Input shape: {inp}")
    except Exception as e:
        models["eye"]["error"] = str(e)
        print(f"[EYE] [FAIL] Load failed: {e}")
        traceback.print_exc()


# ── Load oral model ────────────────────────────────────────────────────────────
def load_oral_model():
    oral_path = MODELS_DIR / "best_oral_model.keras"
    if not oral_path.exists():
        models["oral"]["error"] = f"Model file not found: {oral_path}"
        print(f"[ORAL] [FAIL]  {models['oral']['error']}")
        return
    try:
        print(f"[ORAL] Loading {oral_path} …")
        models["oral"]["model"] = tf.keras.models.load_model(str(oral_path))
        models["oral"]["loaded"] = True
        inp = models["oral"]["model"].input_shape
        print(f"[ORAL] [OK] Loaded. Input shape: {inp}")
    except Exception as e:
        models["oral"]["error"] = str(e)
        print(f"[ORAL] [FAIL] Load failed: {e}")
        traceback.print_exc()


# ── Helper: preprocess image for skin model ───────────────────────────────────
def preprocess_skin(img: Image.Image) -> np.ndarray:
    """Convert PIL image → model input tensor."""
    m = models["skin"]["model"]
    # Auto-detect expected spatial size from model input_shape
    inp_shape = m.input_shape                  # e.g. (None, 224, 224, 3)
    h = inp_shape[1] if inp_shape[1] else 224
    w = inp_shape[2] if inp_shape[2] else 224
    img_rgb  = img.convert("RGB").resize((w, h), Image.LANCZOS)
    arr      = np.array(img_rgb, dtype=np.float32) / 255.0   # [0, 1]
    return np.expand_dims(arr, axis=0)                        # (1, h, w, 3)


# ── Helper: preprocess image for dental model ──────────────────────────────────
def preprocess_dental(img: Image.Image) -> np.ndarray:
    """Convert PIL image → dental model input tensor."""
    m = models["dental"]["model"]
    inp_shape = m.input_shape
    h = inp_shape[1] if inp_shape[1] else 224
    w = inp_shape[2] if inp_shape[2] else 224
    img_rgb  = img.convert("RGB").resize((w, h), Image.LANCZOS)
    arr      = np.array(img_rgb, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Helper: preprocess image for nail model ────────────────────────────────────
def preprocess_nail(img: Image.Image) -> np.ndarray:
    """Convert PIL image → nail model input tensor."""
    m = models["nail"]["model"]
    inp_shape = m.input_shape
    h = inp_shape[1] if inp_shape[1] else 224
    w = inp_shape[2] if inp_shape[2] else 224
    img_rgb  = img.convert("RGB").resize((w, h), Image.LANCZOS)
    arr      = np.array(img_rgb, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Helper: preprocess image for eye model ─────────────────────────────────────
def preprocess_eye(img: Image.Image) -> np.ndarray:
    """Convert PIL image → eye model input tensor."""
    m = models["eye"]["model"]
    inp_shape = m.input_shape
    h = inp_shape[1] if inp_shape[1] else 224
    w = inp_shape[2] if inp_shape[2] else 224
    img_rgb  = img.convert("RGB").resize((w, h), Image.LANCZOS)
    arr      = np.array(img_rgb, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Helper: preprocess image for oral model ────────────────────────────────────
def preprocess_oral(img: Image.Image) -> np.ndarray:
    """Convert PIL image → oral model input tensor."""
    m = models["oral"]["model"]
    inp_shape = m.input_shape
    h = inp_shape[1] if inp_shape[1] else 224
    w = inp_shape[2] if inp_shape[2] else 224
    img_rgb  = img.convert("RGB").resize((w, h), Image.LANCZOS)
    arr      = np.array(img_rgb, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Helper: image to base64 ───────────────────────────────────────────────────
def img_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    """Serve the frontend HTML."""
    return send_file(BASE_DIR / "index.html")


@app.route("/status", methods=["GET"])
def status():
    """Health check — used by frontend server-status indicator."""
    skin_m   = models["skin"]
    xray_m   = models["xray"]
    dental_m = models["dental"]
    nail_m   = models["nail"]
    eye_m    = models["eye"]
    oral_m   = models["oral"]

    skin_shape   = str(skin_m["model"].input_shape)   if skin_m["loaded"]   and skin_m["model"]   else None
    dental_shape = str(dental_m["model"].input_shape) if dental_m["loaded"] and dental_m["model"] else None
    nail_shape   = str(nail_m["model"].input_shape)   if nail_m["loaded"]   and nail_m["model"]   else None
    eye_shape    = str(eye_m["model"].input_shape)    if eye_m["loaded"]    and eye_m["model"]    else None
    oral_shape   = str(oral_m["model"].input_shape)   if oral_m["loaded"]   and oral_m["model"]   else None

    return jsonify({
        "status":       "ok",
        "model_loaded": skin_m["loaded"],          # used by xray page legacy check
        "skin": {
            "loaded":      skin_m["loaded"],
            "error":       skin_m["error"],
            "classes":     len(SKIN_CLASSES),
            "input_shape": skin_shape,
        },
        "xray": {
            "loaded":    xray_m["loaded"],
            "error":     xray_m["error"],
        },
        "dental": {
            "loaded":      dental_m["loaded"],
            "error":       dental_m["error"],
            "classes":     len(DENTAL_CLASSES),
            "input_shape": dental_shape,
        },
        "nail": {
            "loaded":      nail_m["loaded"],
            "error":       nail_m["error"],
            "classes":     len(NAIL_CLASSES),
            "input_shape": nail_shape,
        },
        "eye": {
            "loaded":      eye_m["loaded"],
            "error":       eye_m["error"],
            "classes":     len(EYE_CLASSES),
            "input_shape": eye_shape,
        },
        "oral": {
            "loaded":      oral_m["loaded"],
            "error":       oral_m["error"],
            "classes":     len(ORAL_CLASSES),
            "input_shape": oral_shape,
        },
    })


@app.route("/predict/skin", methods=["POST"])
def predict_skin():
    """
    POST multipart/form-data:
        image               — image file (jpg/png/webp)
        top_k               — (int, default 5) return top-K predictions
        confidence_threshold — (float, default 0.0)
    Returns JSON:
        {
          predictions: [{label, confidence, color, rank}],
          top_label:   str,
          top_confidence: float,
          model_info:  {name, classes, input_shape},
          annotated_image: base64_png   (thumbnail with top label overlay)
        }
    """
    # ── 1. Validate model loaded ──
    if not models["skin"]["loaded"]:
        return jsonify({"error": "Skin model not loaded. " + (models["skin"]["error"] or "Unknown error")}), 503

    # ── 2. Get image ──
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Key must be 'image'."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    top_k  = int(request.form.get("top_k", 5))
    top_k  = max(1, min(top_k, len(SKIN_CLASSES)))
    thresh = float(request.form.get("confidence_threshold", 0.0))

    try:
        img_bytes = file.read()
        img       = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    # ── 3. Inference ──
    try:
        inp   = preprocess_skin(img)
        preds = models["skin"]["model"].predict(inp, verbose=0)[0]   # shape (20,)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {e}"}), 500

    # ── 4. Build ranked predictions ──
    ranked = sorted(enumerate(preds.tolist()), key=lambda x: x[1], reverse=True)
    top_results = []
    for rank, (class_idx, conf) in enumerate(ranked[:top_k]):
        if conf < thresh:
            continue
        label = SKIN_CLASSES[class_idx] if class_idx < len(SKIN_CLASSES) else f"Class_{class_idx}"
        top_results.append({
            "rank":       rank + 1,
            "label":      label,
            "confidence": round(float(conf), 4),
            "color":      CLASS_COLORS.get(label, "#10b981"),
        })

    top_label = top_results[0]["label"] if top_results else "Unknown"
    top_conf  = top_results[0]["confidence"] if top_results else 0.0

    # ── 5. Create annotated thumbnail ──
    thumb   = img.convert("RGB")
    tw, th  = thumb.size
    scale   = 400 / max(tw, th)
    thumb   = thumb.resize((int(tw * scale), int(th * scale)), Image.LANCZOS)
    draw    = ImageDraw.Draw(thumb)

    # Draw result overlay
    top_color_hex = top_results[0]["color"] if top_results else "#10b981"
    # Convert hex to RGB
    hex_c = top_color_hex.lstrip("#")
    overlay_color = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))

    tw2, th2 = thumb.size
    # Banner at top
    draw.rectangle([0, 0, tw2, 52], fill=(0, 0, 0, 180))
    draw.rectangle([0, 0, tw2, 6], fill=overlay_color)

    # Label text (use default font since custom fonts may not be available)
    conf_text = f"{top_label}  {top_conf*100:.1f}%"
    try:
        font_title = ImageFont.truetype("arial.ttf", 18)
        font_sub   = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub   = font_title

    draw.text((12, 12), conf_text, fill=overlay_color, font=font_title)
    draw.text((12, 34), "BIHOS · Skin AI  (Research use only)", fill=(200, 200, 200), font=font_sub)

    annotated_b64 = img_to_b64(thumb, "PNG")

    m = models["skin"]["model"]
    inp_shape_str = str(m.input_shape)

    return jsonify({
        "predictions":      top_results,
        "top_label":        top_label,
        "top_confidence":   top_conf,
        "annotated_image":  annotated_b64,
        "model_info": {
            "name":        "best_model.keras",
            "classes":     len(SKIN_CLASSES),
            "input_shape": inp_shape_str,
        },
    })


@app.route("/predict/xray", methods=["POST"])
def predict_xray():
    """
    Legacy X-Ray endpoint placeholder.
    The original X-Ray model (RetinaNet) may be loaded separately.
    Returns 503 if no xray model loaded.
    """
    if not models["xray"]["loaded"]:
        return jsonify({
            "error": (
                "X-Ray RetinaNet model not loaded. "
                "Place the RetinaNet weights in models/ and update this endpoint."
            )
        }), 503

    # If you have the RetinaNet model, add inference logic here.
    return jsonify({"error": "X-Ray model integration pending."}), 501


@app.route("/predict/dental", methods=["POST"])
def predict_dental():
    """
    POST multipart/form-data:
        image               — image file (jpg/png/webp) of dental X-ray or oral photo
        top_k               — (int, default 3) return top-K predictions
        confidence_threshold — (float, default 0.0)
    Returns JSON:
        {
          predictions: [{label, confidence, color, rank}],
          top_label:   str,
          top_confidence: float,
          model_info:  {name, classes, input_shape},
          annotated_image: base64_png
        }
    """
    # ── 1. Validate model loaded ──
    if not models["dental"]["loaded"]:
        return jsonify({"error": "Dental model not loaded. " + (models["dental"]["error"] or "Unknown error")}), 503

    # ── 2. Get image ──
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Key must be 'image'."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    top_k  = int(request.form.get("top_k", 3))
    top_k  = max(1, min(top_k, len(DENTAL_CLASSES)))
    thresh = float(request.form.get("confidence_threshold", 0.0))

    try:
        img_bytes = file.read()
        img       = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    # ── 3. Inference ──
    try:
        inp   = preprocess_dental(img)
        preds = models["dental"]["model"].predict(inp, verbose=0)[0]   # shape (3,)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {e}"}), 500

    # ── 4. Build ranked predictions ──
    ranked = sorted(enumerate(preds.tolist()), key=lambda x: x[1], reverse=True)
    top_results = []
    for rank, (class_idx, conf) in enumerate(ranked[:top_k]):
        if conf < thresh:
            continue
        label = DENTAL_CLASSES[class_idx] if class_idx < len(DENTAL_CLASSES) else f"Class_{class_idx}"
        top_results.append({
            "rank":       rank + 1,
            "label":      label,
            "confidence": round(float(conf), 4),
            "color":      DENTAL_CLASS_COLORS.get(label, "#0d9488"),
        })

    top_label = top_results[0]["label"] if top_results else "Unknown"
    top_conf  = top_results[0]["confidence"] if top_results else 0.0

    # ── 5. Create annotated thumbnail ──
    thumb   = img.convert("RGB")
    tw, th  = thumb.size
    scale   = 400 / max(tw, th)
    thumb   = thumb.resize((int(tw * scale), int(th * scale)), Image.LANCZOS)
    draw    = ImageDraw.Draw(thumb)

    top_color_hex  = top_results[0]["color"] if top_results else "#0d9488"
    hex_c          = top_color_hex.lstrip("#")
    overlay_color  = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))

    tw2, th2 = thumb.size
    draw.rectangle([0, 0, tw2, 52], fill=(0, 0, 0, 180))
    draw.rectangle([0, 0, tw2, 6], fill=overlay_color)

    conf_text = f"{top_label}  {top_conf*100:.1f}%"
    try:
        font_title = ImageFont.truetype("arial.ttf", 18)
        font_sub   = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub   = font_title

    draw.text((12, 12), conf_text, fill=overlay_color, font=font_title)
    draw.text((12, 34), "BIHOS · Dental AI  (Research use only)", fill=(200, 200, 200), font=font_sub)

    annotated_b64 = img_to_b64(thumb, "PNG")

    m = models["dental"]["model"]
    inp_shape_str = str(m.input_shape)

    return jsonify({
        "predictions":      top_results,
        "top_label":        top_label,
        "top_confidence":   top_conf,
        "annotated_image":  annotated_b64,
        "model_info": {
            "name":        "best_dental_model.keras",
            "classes":     len(DENTAL_CLASSES),
            "input_shape": inp_shape_str,
        },
    })


@app.route("/predict/nail", methods=["POST"])
def predict_nail():
    """
    POST multipart/form-data:
        image               — image file of a nail
        top_k               — (int, default 3)
        confidence_threshold — (float, default 0.0)
    """
    if not models["nail"]["loaded"]:
        return jsonify({"error": "Nail model not loaded. " + (models["nail"]["error"] or "Unknown error")}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Key must be 'image'."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    top_k  = int(request.form.get("top_k", 3))
    top_k  = max(1, min(top_k, len(NAIL_CLASSES)))
    thresh = float(request.form.get("confidence_threshold", 0.0))

    try:
        img_bytes = file.read()
        img       = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    try:
        inp   = preprocess_nail(img)
        preds = models["nail"]["model"].predict(inp, verbose=0)[0]
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {e}"}), 500

    ranked = sorted(enumerate(preds.tolist()), key=lambda x: x[1], reverse=True)
    top_results = []
    for rank, (class_idx, conf) in enumerate(ranked[:top_k]):
        if conf < thresh:
            continue
        label = NAIL_CLASSES[class_idx] if class_idx < len(NAIL_CLASSES) else f"Class_{class_idx}"
        top_results.append({
            "rank":       rank + 1,
            "label":      label,
            "confidence": round(float(conf), 4),
            "color":      NAIL_CLASS_COLORS.get(label, "#f43f5e"),
        })

    top_label = top_results[0]["label"] if top_results else "Unknown"
    top_conf  = top_results[0]["confidence"] if top_results else 0.0

    thumb   = img.convert("RGB")
    tw, th  = thumb.size
    scale   = 400 / max(tw, th)
    thumb   = thumb.resize((int(tw * scale), int(th * scale)), Image.LANCZOS)
    draw    = ImageDraw.Draw(thumb)

    top_color_hex  = top_results[0]["color"] if top_results else "#f43f5e"
    hex_c          = top_color_hex.lstrip("#")
    overlay_color  = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))

    tw2, th2 = thumb.size
    draw.rectangle([0, 0, tw2, 52], fill=(0, 0, 0, 180))
    draw.rectangle([0, 0, tw2, 6], fill=overlay_color)

    conf_text = f"{top_label}  {top_conf*100:.1f}%"
    try:
        font_title = ImageFont.truetype("arial.ttf", 18)
        font_sub   = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub   = font_title

    draw.text((12, 12), conf_text, fill=overlay_color, font=font_title)
    draw.text((12, 34), "BIHOS · Nail AI  (Research use only)", fill=(200, 200, 200), font=font_sub)

    annotated_b64 = img_to_b64(thumb, "PNG")
    inp_shape_str = str(models["nail"]["model"].input_shape)

    return jsonify({
        "predictions":      top_results,
        "top_label":        top_label,
        "top_confidence":   top_conf,
        "annotated_image":  annotated_b64,
        "model_info": {
            "name":        "best_nail_model.keras",
            "classes":     len(NAIL_CLASSES),
            "input_shape": inp_shape_str,
        },
    })


@app.route("/predict/eye", methods=["POST"])
def predict_eye():
    """
    POST multipart/form-data:
        image               — image file of an eye
        top_k               — (int, default 3)
        confidence_threshold — (float, default 0.0)
    """
    if not models["eye"]["loaded"]:
        return jsonify({"error": "Eye model not loaded. " + (models["eye"]["error"] or "Unknown error")}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Key must be 'image'."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    top_k  = int(request.form.get("top_k", 3))
    top_k  = max(1, min(top_k, len(EYE_CLASSES)))
    thresh = float(request.form.get("confidence_threshold", 0.0))

    try:
        img_bytes = file.read()
        img       = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    try:
        inp   = preprocess_eye(img)
        preds = models["eye"]["model"].predict(inp, verbose=0)[0]
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {e}"}), 500

    ranked = sorted(enumerate(preds.tolist()), key=lambda x: x[1], reverse=True)
    top_results = []
    for rank, (class_idx, conf) in enumerate(ranked[:top_k]):
        if conf < thresh:
            continue
        label = EYE_CLASSES[class_idx] if class_idx < len(EYE_CLASSES) else f"Class_{class_idx}"
        top_results.append({
            "rank":       rank + 1,
            "label":      label,
            "confidence": round(float(conf), 4),
            "color":      EYE_CLASS_COLORS.get(label, "#06b6d4"),
        })

    top_label = top_results[0]["label"] if top_results else "Unknown"
    top_conf  = top_results[0]["confidence"] if top_results else 0.0

    thumb   = img.convert("RGB")
    tw, th  = thumb.size
    scale   = 400 / max(tw, th)
    thumb   = thumb.resize((int(tw * scale), int(th * scale)), Image.LANCZOS)
    draw    = ImageDraw.Draw(thumb)

    top_color_hex  = top_results[0]["color"] if top_results else "#06b6d4"
    hex_c          = top_color_hex.lstrip("#")
    overlay_color  = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))

    tw2, th2 = thumb.size
    draw.rectangle([0, 0, tw2, 52], fill=(0, 0, 0, 180))
    draw.rectangle([0, 0, tw2, 6], fill=overlay_color)

    conf_text = f"{top_label}  {top_conf*100:.1f}%"
    try:
        font_title = ImageFont.truetype("arial.ttf", 18)
        font_sub   = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub   = font_title

    draw.text((12, 12), conf_text, fill=overlay_color, font=font_title)
    draw.text((12, 34), "BIHOS · Eye AI  (Research use only)", fill=(200, 200, 200), font=font_sub)

    annotated_b64 = img_to_b64(thumb, "PNG")
    inp_shape_str = str(models["eye"]["model"].input_shape)

    return jsonify({
        "predictions":      top_results,
        "top_label":        top_label,
        "top_confidence":   top_conf,
        "annotated_image":  annotated_b64,
        "model_info": {
            "name":        "best_eye_model.keras",
            "classes":     len(EYE_CLASSES),
            "input_shape": inp_shape_str,
        },
    })


@app.route("/predict/oral", methods=["POST"])
def predict_oral():
    """
    POST multipart/form-data:
        image               — image file of oral cavity
        top_k               — (int, default 3)
        confidence_threshold — (float, default 0.0)
    """
    if not models["oral"]["loaded"]:
        return jsonify({"error": "Oral model not loaded. " + (models["oral"]["error"] or "Unknown error")}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Key must be 'image'."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    top_k  = int(request.form.get("top_k", 3))
    top_k  = max(1, min(top_k, len(ORAL_CLASSES)))
    thresh = float(request.form.get("confidence_threshold", 0.0))

    try:
        img_bytes = file.read()
        img       = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    try:
        inp   = preprocess_oral(img)
        preds = models["oral"]["model"].predict(inp, verbose=0)[0]
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {e}"}), 500

    ranked = sorted(enumerate(preds.tolist()), key=lambda x: x[1], reverse=True)
    top_results = []
    for rank, (class_idx, conf) in enumerate(ranked[:top_k]):
        if conf < thresh:
            continue
        label = ORAL_CLASSES[class_idx] if class_idx < len(ORAL_CLASSES) else f"Class_{class_idx}"
        top_results.append({
            "rank":       rank + 1,
            "label":      label,
            "confidence": round(float(conf), 4),
            "color":      ORAL_CLASS_COLORS.get(label, "#a855f7"),
        })

    top_label = top_results[0]["label"] if top_results else "Unknown"
    top_conf  = top_results[0]["confidence"] if top_results else 0.0

    thumb   = img.convert("RGB")
    tw, th  = thumb.size
    scale   = 400 / max(tw, th)
    thumb   = thumb.resize((int(tw * scale), int(th * scale)), Image.LANCZOS)
    draw    = ImageDraw.Draw(thumb)

    top_color_hex  = top_results[0]["color"] if top_results else "#a855f7"
    hex_c          = top_color_hex.lstrip("#")
    overlay_color  = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))

    tw2, th2 = thumb.size
    draw.rectangle([0, 0, tw2, 52], fill=(0, 0, 0, 180))
    draw.rectangle([0, 0, tw2, 6], fill=overlay_color)

    conf_text = f"{top_label}  {top_conf*100:.1f}%"
    try:
        font_title = ImageFont.truetype("arial.ttf", 18)
        font_sub   = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub   = font_title

    draw.text((12, 12), conf_text, fill=overlay_color, font=font_title)
    draw.text((12, 34), "BIHOS · Oral AI  (Research use only)", fill=(200, 200, 200), font=font_sub)

    annotated_b64 = img_to_b64(thumb, "PNG")
    inp_shape_str = str(models["oral"]["model"].input_shape)

    return jsonify({
        "predictions":      top_results,
        "top_label":        top_label,
        "top_confidence":   top_conf,
        "annotated_image":  annotated_b64,
        "model_info": {
            "name":        "best_oral_model.keras",
            "classes":     len(ORAL_CLASSES),
            "input_shape": inp_shape_str,
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

# ── Load models at module import time (required for Gunicorn / Render) ─────────
print("=" * 60)
print("  BIHOS · AI Backend Server — Loading models …")
print("=" * 60)
load_skin_model()
load_dental_model()
load_nail_model()
load_eye_model()
load_oral_model()
print(f"\n  Skin model loaded   : {models['skin']['loaded']}")
print(f"  Dental model loaded : {models['dental']['loaded']}")
print(f"  Nail model loaded   : {models['nail']['loaded']}")
print(f"  Eye model loaded    : {models['eye']['loaded']}")
print(f"  Oral model loaded   : {models['oral']['loaded']}")
print("=" * 60 + "\n")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
