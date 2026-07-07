"""
BIHOS — Bangladesh Intelligent Healthcare Operating System
Flask Backend · app.py

4 EfficientNetB2 FP16 TFLite Models:
  Eye   — 4 classes  (Cataract, Diabetic Retinopathy, Glaucoma, Normal)
  Hair  — 6+ classes (Alopecia Areata, Androgenetic Alopecia, Telogen Effluvium, Normal, Dandruff, Tinea Capitis, ...)
  Nail  — 6 classes  (Onychomycosis, Nail Psoriasis, Paronychia, Melanonychia, Onycholysis, Normal)
  Skin  — 19 classes (Acne, Eczema, Melanoma, Psoriasis, etc.)

API Endpoints:
  GET  /                    → serves index.html
  GET  /health              → combined health check (all models)
  GET  /status              → combined status (same as /health — for backward compat)
  POST /predict/<model>     → run inference (eye | hair | nail | skin)

Deployment: Render.com free tier (gunicorn, 1 worker, 120s timeout)
"""

import os
import io
import time
import base64
import logging
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("bihos")

# ── TFLite Backend Detection ─────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tflite = None
TFLITE_BACKEND = None

# 1. Try ai-edge-litert (official successor to tflite-runtime, works on Python 3.11+ Linux)
try:
    from ai_edge_litert.interpreter import Interpreter as _LiteRTInterpreter
    class _LiteRTModule:
        Interpreter = _LiteRTInterpreter
    tflite = _LiteRTModule()
    TFLITE_BACKEND = "ai_edge_litert"
    log.info("TFLite backend: ai_edge_litert")
except ImportError:
    pass

# 2. Try tflite_runtime (legacy, Python <=3.11)
if tflite is None:
    try:
        import tflite_runtime.interpreter as tflite
        TFLITE_BACKEND = "tflite_runtime"
        log.info("TFLite backend: tflite_runtime")
    except ImportError:
        pass

# 3. Try tensorflow.lite (bundled with full TensorFlow)
if tflite is None:
    try:
        import tensorflow.lite as tflite
        TFLITE_BACKEND = "tensorflow.lite"
        log.info("TFLite backend: tensorflow.lite")
    except (ImportError, AttributeError):
        pass

# 4. Try tensorflow (fallback for local dev)
if tflite is None:
    try:
        import tensorflow as _tf
        tflite = _tf.lite
        TFLITE_BACKEND = f"tensorflow {_tf.__version__}"
        log.info(f"TFLite backend: {TFLITE_BACKEND}")
    except ImportError:
        pass

if tflite is None:
    log.warning("No TFLite backend found! Install ai-edge-litert or tensorflow.")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "Models"

# ── Class Labels ─────────────────────────────────────────────────────────────
EYE_CLASSES = [
    "Cataract",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Normal / Healthy Eye",
]

HAIR_CLASSES = [
    "Alopecia Areata",
    "Androgenetic Alopecia",
    "Telogen Effluvium",
    "Normal / Healthy Hair",
    "Dandruff / Seborrheic Dermatitis",
    "Tinea Capitis",
    "Trichotillomania",
    "Traction Alopecia",
    "Folliculitis",
    "Lichen Planopilaris",
]

NAIL_CLASSES = [
    "Onychomycosis (Fungal Infection)",
    "Nail Psoriasis",
    "Paronychia (Bacterial/Fungal Inflammation)",
    "Melanonychia (Pigmentation/Stripes)",
    "Onycholysis (Nail Separation)",
    "Normal / Healthy Nail",
]

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
]

# ── Color Maps ───────────────────────────────────────────────────────────────
EYE_COLORS = {
    "Cataract":               "#f59e0b",
    "Diabetic Retinopathy":   "#ef4444",
    "Glaucoma":               "#8b5cf6",
    "Normal / Healthy Eye":   "#10b981",
}

HAIR_COLORS = {
    "Alopecia Areata":                   "#f43f5e",
    "Androgenetic Alopecia":             "#f97316",
    "Telogen Effluvium":                 "#eab308",
    "Normal / Healthy Hair":             "#10b981",
    "Dandruff / Seborrheic Dermatitis":  "#06b6d4",
    "Tinea Capitis":                     "#8b5cf6",
    "Trichotillomania":                  "#ec4899",
    "Traction Alopecia":                 "#6366f1",
    "Folliculitis":                      "#ef4444",
    "Lichen Planopilaris":               "#a855f7",
}

NAIL_COLORS = {
    "Onychomycosis (Fungal Infection)":           "#f43f5e",
    "Nail Psoriasis":                             "#f59e0b",
    "Paronychia (Bacterial/Fungal Inflammation)": "#ef4444",
    "Melanonychia (Pigmentation/Stripes)":        "#8b5cf6",
    "Onycholysis (Nail Separation)":              "#6366f1",
    "Normal / Healthy Nail":                      "#10b981",
}

SKIN_COLORS = {
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
}

# ── Severity Map ─────────────────────────────────────────────────────────────
SEVERITY_MAP = {
    # Eye
    "Cataract":               ("High",     "🔴"),
    "Diabetic Retinopathy":   ("High",     "🔴"),
    "Glaucoma":               ("Critical", "🆘"),
    "Normal / Healthy Eye":   ("None",     "🟢"),
    # Hair
    "Alopecia Areata":                   ("Medium", "🟠"),
    "Androgenetic Alopecia":             ("Medium", "🟠"),
    "Telogen Effluvium":                 ("Low",    "🟡"),
    "Normal / Healthy Hair":             ("None",   "🟢"),
    "Dandruff / Seborrheic Dermatitis":  ("Low",    "🟡"),
    "Tinea Capitis":                     ("Medium", "🟠"),
    "Trichotillomania":                  ("Medium", "🟠"),
    "Traction Alopecia":                 ("Medium", "🟠"),
    "Folliculitis":                      ("High",   "🔴"),
    "Lichen Planopilaris":               ("High",   "🔴"),
    # Nail
    "Onychomycosis (Fungal Infection)":           ("Medium", "🟠"),
    "Nail Psoriasis":                             ("Medium", "🟠"),
    "Paronychia (Bacterial/Fungal Inflammation)": ("High",   "🔴"),
    "Melanonychia (Pigmentation/Stripes)":        ("Medium", "🟠"),
    "Onycholysis (Nail Separation)":              ("Medium", "🟠"),
    "Normal / Healthy Nail":                      ("None",   "🟢"),
    # Skin
    "Acne":              ("Low",      "🟡"),
    "Actinic Carcinoma": ("High",     "🔴"),
    "Atopic Dermatitis": ("Medium",   "🟠"),
    "Bullous Disease":   ("High",     "🔴"),
    "Cellulitis":        ("High",     "🔴"),
    "Eczema":            ("Medium",   "🟠"),
    "Drug Eruptions":    ("High",     "🔴"),
    "Herpes HPV":        ("Medium",   "🟠"),
    "Light Diseases":    ("Low",      "🟡"),
    "Lupus":             ("High",     "🔴"),
    "Melanoma":          ("Critical", "🆘"),
    "Poison IVY":        ("Low",      "🟡"),
    "Psoriasis":         ("Medium",   "🟠"),
    "Benign Tumors":     ("Low",      "🟡"),
    "Systemic Disease":  ("High",     "🔴"),
    "Ringworm":          ("Low",      "🟡"),
    "Urticarial Hives":  ("Medium",   "🟠"),
    "Vascular Tumors":   ("Medium",   "🟠"),
    "Vasculitis":        ("High",     "🔴"),
}

# ── Model Registry ───────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "eye": {
        "file":    "eye_efficientnetb2_fp16.tflite",
        "classes": EYE_CLASSES,
        "colors":  EYE_COLORS,
        "topk":    4,
        "icon":    "👁️",
        "name":    "eye_efficientnetb2_fp16.tflite",
    },
    "hair": {
        "file":    "hair_efficientnetb2_fp16.tflite",
        "classes": HAIR_CLASSES,
        "colors":  HAIR_COLORS,
        "topk":    4,
        "icon":    "💇",
        "name":    "hair_efficientnetb2_fp16.tflite",
    },
    "nail": {
        "file":    "nail_efficientnetb2_fp16.tflite",
        "classes": NAIL_CLASSES,
        "colors":  NAIL_COLORS,
        "topk":    4,
        "icon":    "💅",
        "name":    "nail_efficientnetb2_fp16.tflite",
    },
    "skin": {
        "file":    "skin_efficientnetb2_fp16.tflite",
        "classes": SKIN_CLASSES,
        "colors":  SKIN_COLORS,
        "topk":    5,
        "icon":    "🧴",
        "name":    "skin_efficientnetb2_fp16.tflite",
    },
}

# ── Lazy Model Cache ─────────────────────────────────────────────────────────
_model_cache: dict = {}

def load_model(model_key: str):
    """Load and cache a TFLite interpreter. Raises on error."""
    if model_key in _model_cache:
        return _model_cache[model_key]

    if tflite is None:
        raise RuntimeError("No TFLite backend available. Install tflite-runtime.")

    cfg = MODEL_REGISTRY[model_key]
    model_path = MODELS_DIR / cfg["file"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Ensure '{cfg['file']}' is in the Models/ directory."
        )

    log.info(f"Loading model: {model_path.name} ({model_path.stat().st_size / 1e6:.1f} MB)")
    interp = tflite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    _model_cache[model_key] = interp
    log.info(f"Model loaded: {model_path.name}")
    return interp


def get_model_status(model_key: str) -> dict:
    """Return status dict for a model."""
    cfg = MODEL_REGISTRY.get(model_key, {})
    model_path = MODELS_DIR / cfg.get("file", "")

    if not model_path.exists():
        return {"loaded": False, "error": f"File not found: {cfg.get('file', '')}"}

    try:
        interp = load_model(model_key)
        inp = interp.get_input_details()[0]
        shape = list(inp["shape"])
        return {
            "loaded":      True,
            "file":        cfg["file"],
            "input_shape": str(shape),
            "classes":     len(cfg["classes"]),
            "size_mb":     round(model_path.stat().st_size / 1e6, 1),
            "backend":     TFLITE_BACKEND,
        }
    except Exception as e:
        return {"loaded": False, "error": str(e)}


# ── Inference Helpers ────────────────────────────────────────────────────────
def run_inference(interp, img: Image.Image) -> np.ndarray:
    """Preprocess PIL image and run TFLite inference. Returns float32 probability array."""
    inp_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]
    shape = inp_details["shape"]          # e.g. [1, 260, 260, 3] for EfficientNetB2
    h = int(shape[1]) if shape[1] else 260
    w = int(shape[2]) if shape[2] else 260

    img_rgb = img.convert("RGB").resize((w, h), Image.LANCZOS)
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    tensor = np.expand_dims(arr, axis=0)

    if tensor.dtype != inp_details["dtype"]:
        tensor = tensor.astype(inp_details["dtype"])

    interp.set_tensor(inp_details["index"], tensor)
    interp.invoke()
    preds = interp.get_tensor(out_details["index"])[0]
    return preds.astype(np.float32)


def top_predictions(preds: np.ndarray, classes: list, colors: dict, top_k: int) -> list:
    """Return sorted top-k predictions as list of dicts."""
    ranked = sorted(enumerate(preds.tolist()), key=lambda x: x[1], reverse=True)
    results = []
    for rank, (idx, conf) in enumerate(ranked[:top_k]):
        label = classes[idx] if idx < len(classes) else f"Class_{idx}"
        severity, sev_icon = SEVERITY_MAP.get(label, ("Unknown", "❓"))
        results.append({
            "rank":          rank + 1,
            "label":         label,
            "confidence":    round(float(conf), 4),
            "color":         colors.get(label, "#6366f1"),
            "severity":      severity,
            "severity_icon": sev_icon,
        })
    return results


def annotate_image_b64(img: Image.Image, label: str, conf: float, color_hex: str) -> str:
    """Draw AI result overlay on image and return as base64 PNG string."""
    thumb = img.convert("RGB")
    w, h = thumb.size
    # Resize to max 480px on largest side
    scale = min(480 / max(w, h), 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    thumb = thumb.resize((nw, nh), Image.LANCZOS)

    draw = ImageDraw.Draw(thumb)
    tw, th = thumb.size

    # Parse hex color
    hx  = color_hex.lstrip("#")
    rgb = tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))

    # Header bar
    draw.rectangle([0, 0, tw, 58], fill=(0, 0, 0, 200))
    draw.rectangle([0, 0, tw, 5],  fill=rgb)

    text  = f"{label}  {conf * 100:.1f}%"
    sub   = "BIHOS · CUET AI  (Research use only)"

    try:
        ft = ImageFont.truetype("arial.ttf", 17)
        fs = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        ft = ImageFont.load_default()
        fs = ft

    draw.text((12, 10), text, fill=rgb, font=ft)
    draw.text((12, 36), sub,  fill=(170, 170, 170), font=fs)

    # Encode to base64
    buf = io.BytesIO()
    thumb.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)  # Allow all origins (needed for same-origin HTML → backend calls)

# Max upload size: 16 MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


@app.route("/")
def index():
    """Serve the main HTML frontend."""
    html_path = BASE_DIR / "index.html"
    if html_path.exists():
        return send_file(str(html_path))
    return "<h1>BIHOS API Running</h1><p>index.html not found.</p>", 200


@app.route("/health", methods=["GET"])
@app.route("/status", methods=["GET"])
def health():
    """
    Combined health/status endpoint.

    Response shape expected by index.html:
    {
      "eye":  {"loaded": true, "input_shape": "[1,260,260,3]", ...},
      "hair": {...},
      "nail": {...},
      "skin": {...},
      "model_loaded": true,   ← for xray-compat
      "backend": "tflite_runtime",
      ...
    }
    """
    status = {}
    all_loaded = True

    for key in MODEL_REGISTRY:
        s = get_model_status(key)
        status[key] = s
        if not s.get("loaded"):
            all_loaded = False

    return jsonify({
        **status,
        "model_loaded": all_loaded,
        "backend":      TFLITE_BACKEND or "not_found",
        "models_dir":   str(MODELS_DIR),
        "uptime":       "ok",
    })


@app.route("/predict/<model_key>", methods=["POST"])
def predict(model_key: str):
    """
    Run TFLite inference for the requested model.

    Form params:
      image              — image file (required)
      top_k              — int, number of predictions to return (optional, default from registry)
      confidence_threshold — float 0-1 (optional, unused but accepted for compat)

    Response:
    {
      "success":         true,
      "predictions":     [{rank, label, confidence, color, severity, severity_icon}, ...],
      "top_label":       "Glaucoma",
      "top_confidence":  0.9132,
      "annotated_image": "<base64 PNG>",
      "model_info":      {name, classes, input_shape},
      "processing_ms":   312
    }
    """
    if model_key not in MODEL_REGISTRY:
        return jsonify({
            "success": False,
            "error":   f"Unknown model '{model_key}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        }), 400

    # Get image file
    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"success": False, "error": "No image file provided. Send as 'image' field."}), 400

    # Parse optional params
    cfg   = MODEL_REGISTRY[model_key]
    top_k = int(request.form.get("top_k", cfg["topk"]))
    top_k = max(1, min(top_k, len(cfg["classes"])))

    t0 = time.time()

    try:
        # Load image
        img = Image.open(io.BytesIO(image_file.read()))

        # Load/get cached interpreter
        interp = load_model(model_key)

        # Run inference
        preds = run_inference(interp, img)

        # Build predictions list
        results = top_predictions(preds, cfg["classes"], cfg["colors"], top_k)
        top     = results[0]

        # Generate annotated image
        ann_b64 = annotate_image_b64(img, top["label"], top["confidence"], top["color"])

        # Model info
        inp     = interp.get_input_details()[0]
        out     = interp.get_output_details()[0]
        model_info = {
            "name":        cfg["name"],
            "classes":     len(cfg["classes"]),
            "input_shape": str(list(inp["shape"])),
            "output_shape": str(list(out["shape"])),
            "backend":     TFLITE_BACKEND,
        }

        elapsed_ms = round((time.time() - t0) * 1000)
        log.info(f"[{model_key}] Predicted '{top['label']}' ({top['confidence']*100:.1f}%) in {elapsed_ms}ms")

        return jsonify({
            "success":         True,
            "model":           model_key,
            "predictions":     results,
            "top_label":       top["label"],
            "top_confidence":  top["confidence"],
            "annotated_image": ann_b64,
            "model_info":      model_info,
            "processing_ms":   elapsed_ms,
        })

    except FileNotFoundError as e:
        log.error(f"[{model_key}] Model file not found: {e}")
        return jsonify({"success": False, "error": str(e)}), 503

    except RuntimeError as e:
        log.error(f"[{model_key}] Runtime error: {e}")
        return jsonify({"success": False, "error": str(e)}), 503

    except Exception as e:
        log.error(f"[{model_key}] Inference failed: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error":   f"Inference failed: {str(e)}",
            "trace":   traceback.format_exc()
        }), 500


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"success": False, "error": "File too large. Maximum size is 16 MB."}), 413


@app.errorhandler(404)
def not_found(e):
    # For SPA navigation, serve index.html on unknown routes
    html_path = BASE_DIR / "index.html"
    if html_path.exists():
        return send_file(str(html_path))
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error"}), 500


# ── Pre-warm models at startup (optional, reduces first-request latency) ─────
def prewarm_models():
    """Try to load all models at startup so first requests are fast."""
    log.info("Pre-warming models...")
    for key in MODEL_REGISTRY:
        try:
            load_model(key)
        except Exception as e:
            log.warning(f"Could not pre-warm model '{key}': {e}")
    log.info("Model pre-warming complete.")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info(f"Starting BIHOS Flask server on port {port}")
    log.info(f"TFLite backend: {TFLITE_BACKEND}")
    log.info(f"Models directory: {MODELS_DIR}")

    # Pre-warm on startup in development
    prewarm_models()

    app.run(host="0.0.0.0", port=port, debug=False)
