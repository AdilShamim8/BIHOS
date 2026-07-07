"""
BIHOS — CUET AI Health Screening System
Streamlit App · streamlit_app.py

4 EfficientNetB2 FP16 TFLite Models:
  Eye   — 4 classes  (Cataract, Diabetic Retinopathy, Glaucoma, Normal)
  Hair  — 6 classes  (Alopecia Areata, Androgenetic Alopecia, Telogen Effluvium, Normal, Dandruff, Tinea Capitis)
  Nail  — 6 classes  (Onychomycosis, Nail Psoriasis, Paronychia, Melanonychia, Onycholysis, Normal)
  Skin  — 20 classes (Acne, Eczema, Melanoma, Psoriasis, etc.)

Run with Python 3.10-3.12 (TensorFlow required):
  d:\\CUET Doc Project\\.venv312\\Scripts\\python.exe -m streamlit run streamlit_app.py
"""

import os
import io
import traceback
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ── TFLite Backend ────────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tflite = None
TFLITE_BACKEND = None

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_BACKEND = "tflite_runtime"
except ImportError:
    pass

if tflite is None:
    try:
        import tensorflow.lite as tflite
        TFLITE_BACKEND = "tensorflow.lite"
    except (ImportError, AttributeError):
        pass

if tflite is None:
    try:
        import tensorflow as _tf
        tflite = _tf.lite
        TFLITE_BACKEND = f"tensorflow {_tf.__version__}"
    except ImportError:
        pass

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "Models"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — MUST be first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BIHOS · CUET AI Health Screening",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "BIHOS — CUET AI Health Screening System. 4 EfficientNetB2 models for Eye, Hair, Nail & Skin analysis.",
    },
)

# ══════════════════════════════════════════════════════════════════════════════
# CLASS LABELS
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# COLORS PER CLASS
# ══════════════════════════════════════════════════════════════════════════════
EYE_COLORS = {
    "Cataract":                  "#f59e0b",
    "Diabetic Retinopathy":      "#ef4444",
    "Glaucoma":                  "#8b5cf6",
    "Normal / Healthy Eye":      "#10b981",
}

HAIR_COLORS = {
    "Alopecia Areata":                    "#f43f5e",
    "Androgenetic Alopecia":              "#f97316",
    "Telogen Effluvium":                  "#eab308",
    "Normal / Healthy Hair":              "#10b981",
    "Dandruff / Seborrheic Dermatitis":   "#06b6d4",
    "Tinea Capitis":                      "#8b5cf6",
    "Trichotillomania":                   "#ec4899",
    "Traction Alopecia":                  "#6366f1",
    "Folliculitis":                       "#ef4444",
    "Lichen Planopilaris":                "#a855f7",
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

# ══════════════════════════════════════════════════════════════════════════════
# SEVERITY MAP
# ══════════════════════════════════════════════════════════════════════════════
SEVERITY_MAP = {
    # Eye
    "Cataract":                  ("High",     "🔴"),
    "Diabetic Retinopathy":      ("High",     "🔴"),
    "Glaucoma":                  ("Critical", "🆘"),
    "Normal / Healthy Eye":      ("None",     "🟢"),
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
    "Viral Infections":  ("Medium",   "🟠"),
}

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium Dark Glassmorphism Theme
# ══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #060610 0%, #0a0a1a 35%, #0d1120 65%, #060612 100%);
    min-height: 100vh;
}

#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}
.stDeployButton {display: none;}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a1a 0%, #0f1320 100%);
    border-right: 1px solid rgba(99,102,241,0.15);
}

/* ── Hero ── */
.hero-banner {
    background: linear-gradient(135deg,
        rgba(99,102,241,0.12) 0%,
        rgba(6,182,212,0.08) 50%,
        rgba(16,185,129,0.06) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 24px; padding: 44px 52px; margin-bottom: 32px;
    text-align: center; position: relative; overflow: hidden;
    backdrop-filter: blur(20px);
}
.hero-banner::before {
    content: ''; position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(99,102,241,0.06) 0%, transparent 70%);
    animation: pulse-bg 5s ease-in-out infinite;
}
@keyframes pulse-bg {
    0%, 100% { opacity: 0.5; transform: scale(1); }
    50%       { opacity: 1;   transform: scale(1.04); }
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.4rem; font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #06b6d4 50%, #10b981 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin: 0; line-height: 1.1;
}
.hero-subtitle {
    color: #94a3b8; font-size: 1.05rem; margin-top: 12px;
}
.hero-badge {
    display: inline-block; color: white;
    padding: 5px 14px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin: 4px;
    letter-spacing: 0.05em; text-transform: uppercase;
}

/* ── Cards ── */
.model-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px; padding: 22px; margin: 10px 0;
    transition: all 0.3s ease; backdrop-filter: blur(10px);
}
.model-card:hover {
    border-color: rgba(99,102,241,0.45);
    background: rgba(99,102,241,0.04);
    transform: translateY(-3px);
    box-shadow: 0 10px 35px rgba(99,102,241,0.12);
}
.result-card {
    background: linear-gradient(135deg, rgba(17,24,39,0.85) 0%, rgba(31,41,55,0.65) 100%);
    border: 1px solid rgba(99,102,241,0.28);
    border-radius: 18px; padding: 26px; margin: 16px 0;
    backdrop-filter: blur(20px);
}
.result-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.75rem; font-weight: 700; color: #e2e8f0;
}
.result-conf { font-size: 1.05rem; color: #94a3b8; }

/* ── Prediction bars ── */
.pred-bar {
    background: rgba(255,255,255,0.05); border-radius: 8px;
    height: 8px; margin: 7px 0; overflow: hidden;
}
.pred-bar-fill { height: 100%; border-radius: 8px; transition: width 0.8s ease; }

/* ── Metric Cards ── */
.metric-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 18px; text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 800; font-family: 'Space Grotesk', sans-serif; }
.metric-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }

/* ── Status dot ── */
.status-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 5px;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.5; }
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem; font-weight: 700; color: #e2e8f0; margin-bottom: 4px;
}
.section-sub { color: #64748b; font-size: 0.88rem; margin-bottom: 20px; }

/* ── Disclaimer ── */
.disclaimer-box {
    background: rgba(245,158,11,0.07);
    border: 1px solid rgba(245,158,11,0.28);
    border-left: 4px solid #f59e0b;
    border-radius: 8px; padding: 13px 17px; margin: 14px 0;
    color: #fcd34d; font-size: 0.84rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important; border: none !important; border-radius: 11px !important;
    padding: 10px 24px !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important; transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.28) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.48) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.018) !important;
    border: 2px dashed rgba(99,102,241,0.28) !important;
    border-radius: 12px !important; transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.55) !important;
    background: rgba(99,102,241,0.025) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.025) !important;
    border-radius: 14px !important; padding: 5px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 9px !important;
    color: #94a3b8 !important; font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #060610; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.65); }

/* ── Footer ── */
.bihos-footer {
    text-align: center; padding: 28px; color: #374151;
    font-size: 0.78rem; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 60px;
}

/* ── Alerts ── */
.stSuccess {
    background: rgba(16,185,129,0.08) !important;
    border: 1px solid rgba(16,185,129,0.28) !important; color: #6ee7b7 !important;
}
.stError {
    background: rgba(239,68,68,0.08) !important;
    border: 1px solid rgba(239,68,68,0.28) !important; color: #fca5a5 !important;
}
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
MODEL_CONFIG = {
    "eye": {
        "file":         "eye_efficientnetb2_fp16.tflite",
        "classes":      EYE_CLASSES,
        "colors":       EYE_COLORS,
        "default_topk": 4,
        "icon":         "👁️",
        "short":        "Eye",
        "label":        "👁️ Eye Disease AI",
        "accent":       "#8b5cf6",
        "description":  "Screens for Cataract, Diabetic Retinopathy & Glaucoma using EfficientNetB2 FP16.",
        "tip":          "Use retinal fundus images or clear frontal eye photographs for best results.",
        "examples":     "Fundus images, OCT scans, clinical eye photos",
    },
    "hair": {
        "file":         "hair_efficientnetb2_fp16.tflite",
        "classes":      HAIR_CLASSES,
        "colors":       HAIR_COLORS,
        "default_topk": 4,
        "icon":         "💇",
        "short":        "Hair",
        "label":        "💇 Hair Disease AI",
        "accent":       "#06b6d4",
        "description":  "Identifies 6 scalp & hair conditions: Alopecia, Dandruff, Tinea Capitis & more.",
        "tip":          "Upload a clear close-up photo of the scalp or hair-affected area.",
        "examples":     "Scalp photos, close-up hair / scalp images",
    },
    "nail": {
        "file":         "nail_efficientnetb2_fp16.tflite",
        "classes":      NAIL_CLASSES,
        "colors":       NAIL_COLORS,
        "default_topk": 4,
        "icon":         "💅",
        "short":        "Nail",
        "label":        "💅 Nail Disease AI",
        "accent":       "#f43f5e",
        "description":  "Detects 6 nail conditions: Onychomycosis, Psoriasis, Paronychia, Melanonychia & more.",
        "tip":          "Ensure the nail is well-lit, in focus and clearly visible.",
        "examples":     "Fingernail / toenail close-up photos",
    },
    "skin": {
        "file":         "skin_efficientnetb2_fp16.tflite",
        "classes":      SKIN_CLASSES,
        "colors":       SKIN_COLORS,
        "default_topk": 5,
        "icon":         "🧴",
        "short":        "Skin",
        "label":        "🧴 Skin Disease AI",
        "accent":       "#f59e0b",
        "description":  "Classifies 20 dermatological conditions including Melanoma, Eczema, Psoriasis & more.",
        "tip":          "Upload a clear, well-lit photo of the affected skin area.",
        "examples":     "Rashes, lesions, spots, discolorations, growths",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# CORE INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load and cache a TFLite interpreter. Raises on error."""
    if tflite is None:
        raise RuntimeError(
            "No TFLite backend found.\n"
            "Run with the venv that has TensorFlow:\n"
            "  d:\\CUET Doc Project\\.venv312\\Scripts\\python.exe -m streamlit run streamlit_app.py"
        )
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found:\n{p}")
    interp = tflite.Interpreter(model_path=str(p))
    interp.allocate_tensors()
    return interp


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
    """Return sorted top-k predictions as a list of dicts."""
    ranked = sorted(enumerate(preds.tolist()), key=lambda x: x[1], reverse=True)
    results = []
    for rank, (idx, conf) in enumerate(ranked[:top_k]):
        label = classes[idx] if idx < len(classes) else f"Class_{idx}"
        results.append({
            "rank":       rank + 1,
            "label":      label,
            "confidence": round(float(conf), 4),
            "color":      colors.get(label, "#6366f1"),
        })
    return results


def annotate_image(img: Image.Image, label: str, conf: float, color_hex: str) -> Image.Image:
    """Draw AI result banner overlay on a resized copy of the image."""
    thumb = img.convert("RGB")
    w, h = thumb.size
    scale = 480 / max(w, h)
    thumb = thumb.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(thumb)
    tw, th = thumb.size
    hx  = color_hex.lstrip("#")
    rgb = tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))
    draw.rectangle([0, 0, tw, 56], fill=(0, 0, 0, 200))
    draw.rectangle([0, 0, tw, 5],  fill=rgb)
    text = f"{label}  {conf * 100:.1f}%"
    try:
        ft = ImageFont.truetype("arial.ttf", 18)
        fs = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        ft = ImageFont.load_default()
        fs = ft
    draw.text((12, 10), text,                                    fill=rgb,           font=ft)
    draw.text((12, 34), "BIHOS · CUET AI  (Research use only)", fill=(170, 170, 170), font=fs)
    return thumb


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:22px 0 12px;">
            <div style="font-size:2.8rem;">🏥</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.35rem; font-weight:800;
                        background:linear-gradient(135deg,#6366f1,#06b6d4);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">BIHOS</div>
            <div style="color:#64748b; font-size:0.7rem; margin-top:2px;">CUET AI Health Screening</div>
            <div style="color:#374151; font-size:0.65rem;">4 EfficientNetB2 FP16 Models</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Model status
        st.markdown("**🤖 Model Status**")
        for key, cfg in MODEL_CONFIG.items():
            path   = MODELS_DIR / cfg["file"]
            exists = path.exists()
            dot_c  = "#10b981" if exists else "#ef4444"
            sz     = f"{path.stat().st_size / 1e6:.1f} MB" if exists else "missing ⚠️"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:7px 10px; border-radius:8px; margin:3px 0;
                        background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05);">
                <span style="font-size:0.82rem; color:#cbd5e1;">{cfg['icon']} {cfg['short']}</span>
                <span style="font-size:0.75rem; color:{dot_c};">
                    <span class="status-dot" style="background:{dot_c};"></span>{sz}
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Quick stats
        st.markdown("**📊 System Info**")
        n_loaded  = sum(1 for c in MODEL_CONFIG.values() if (MODELS_DIR / c["file"]).exists())
        n_total   = len(MODEL_CONFIG)
        n_classes = sum(len(c["classes"]) for c in MODEL_CONFIG.values())
        c1, c2 = st.columns(2)
        with c1: st.metric("Models", f"{n_loaded}/{n_total}")
        with c2: st.metric("Classes", n_classes)

        st.markdown(
            f"<div style='color:#374151; font-size:0.72rem; margin-top:4px;'>"
            f"Backend: {TFLITE_BACKEND or 'Not found ⚠️'}</div>",
            unsafe_allow_html=True
        )

        st.divider()

        with st.expander("❓ How to use"):
            st.markdown("""
1. **Pick a tab** — Eye, Hair, Nail or Skin
2. **Upload** a medical image (JPG/PNG/WebP)
3. **Set Top-K** with the slider
4. **Click Analyze** → view AI results
5. **Always** consult a real doctor!
            """)

        st.markdown(
            "<div style='color:#374151; font-size:0.68rem; text-align:center; margin-top:8px;'>"
            "© 2025 BIHOS · CUET · Research Only</div>",
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_home():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🏥 BIHOS</div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; color:#94a3b8; margin-top:6px; font-weight:500;">
            CUET AI Health Screening System
        </div>
        <div class="hero-subtitle">
            4 EfficientNetB2 FP16 deep-learning models for<br>
            Eye · Hair · Nail · Skin disease screening
        </div>
        <div style="margin-top:18px; display:flex; gap:8px; justify-content:center; flex-wrap:wrap;">
            <span class="hero-badge" style="background:linear-gradient(135deg,#6366f1,#8b5cf6);">🧠 EfficientNetB2</span>
            <span class="hero-badge" style="background:linear-gradient(135deg,#0d9488,#06b6d4);">⚡ TFLite FP16</span>
            <span class="hero-badge" style="background:linear-gradient(135deg,#dc2626,#f97316);">🏥 Medical AI</span>
            <span class="hero-badge" style="background:linear-gradient(135deg,#7c3aed,#6366f1);">🔬 Research</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    cols = st.columns(4)
    stats = [
        ("4",    "AI Models",       "#6366f1"),
        ("36",   "Disease Classes", "#06b6d4"),
        ("FP16", "Precision",       "#10b981"),
        ("B2",   "EfficientNet",    "#f59e0b"),
    ]
    for col, (val, lbl, color) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color};">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model cards
    st.markdown("<div class='section-header'>🤖 Available AI Models</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Select a model tab above to upload an image and run inference</div>",
                unsafe_allow_html=True)

    cols2 = st.columns(2)
    for i, (key, cfg) in enumerate(MODEL_CONFIG.items()):
        with cols2[i % 2]:
            class_preview = "  ·  ".join(cfg["classes"][:3]) + ("  …" if len(cfg["classes"]) > 3 else "")
            path = MODELS_DIR / cfg["file"]
            ok = path.exists()
            dot = '<span style="color:#10b981;">●</span>' if ok else '<span style="color:#ef4444;">● missing</span>'
            st.markdown(f"""
            <div class="model-card" style="border-color:{cfg['accent']}40;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                    <span style="font-size:2rem;">{cfg['icon']}</span>
                    <div>
                        <div style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                                    color:#e2e8f0; font-size:1rem;">{cfg['label']} {dot}</div>
                        <div style="color:#64748b; font-size:0.75rem;">
                            {len(cfg['classes'])} classes · EfficientNetB2 FP16
                        </div>
                    </div>
                </div>
                <div style="color:#94a3b8; font-size:0.82rem; margin-bottom:8px;">{cfg['description']}</div>
                <div style="color:{cfg['accent']}; font-size:0.72rem;">{class_preview}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>Research &amp; Educational Use Only.</strong>
        BIHOS is developed at CUET for research purposes. It is NOT a certified medical device
        and must NOT be used for clinical diagnosis. Always consult a licensed healthcare provider.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC MODEL INFERENCE PAGE  (shared by all 4 models)
# ══════════════════════════════════════════════════════════════════════════════
def render_model_page(model_key: str):
    """Render the inference UI for a given model key (eye | hair | nail | skin)."""
    cfg        = MODEL_CONFIG[model_key]
    model_file = MODELS_DIR / cfg["file"]
    classes    = cfg["classes"]
    colors     = cfg["colors"]
    icon       = cfg["icon"]
    short      = cfg["short"]
    accent     = cfg["accent"]

    # Header
    st.markdown(f"""
    <div class="model-card" style="border-color:{accent}50;
         background:linear-gradient(135deg,{accent}08,transparent);">
        <div style="display:flex; align-items:center; gap:14px;">
            <span style="font-size:2.8rem;">{icon}</span>
            <div>
                <div class="section-header">{cfg['label']}</div>
                <div class="section-sub" style="margin-bottom:0;">{cfg['description']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>Research &amp; Educational Use Only.</strong>
        This AI tool is NOT a substitute for professional medical diagnosis.
        Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)

    # Guard: model file
    if not model_file.exists():
        st.error(
            f"❌ Model file not found:\n`{model_file}`\n\n"
            f"Make sure `{cfg['file']}` is in the `Models/` folder next to this script."
        )
        return

    # Guard: TFLite backend
    if tflite is None:
        st.error(
            "❌ No TFLite backend found.\n\n"
            "Run this app with Python 3.10–3.12 that has TensorFlow installed:\n"
            "`d:\\\\CUET Doc Project\\\\.venv312\\\\Scripts\\\\python.exe -m streamlit run streamlit_app.py`"
        )
        return

    col1, col2 = st.columns([1, 1.25], gap="large")

    with col1:
        st.markdown(
            f"<div style='color:#94a3b8; font-size:0.84rem; margin-bottom:8px;'>💡 <em>{cfg['tip']}</em></div>",
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            f"Upload {short} image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key=f"up_{short}",
            help=f"Examples: {cfg['examples']}",
        )

        top_k = st.slider(
            "Top-K predictions to show",
            min_value=1,
            max_value=len(classes),
            value=min(cfg["default_topk"], len(classes)),
            key=f"topk_{short}",
        )

        analyze_btn = st.button(
            f"🔬 Analyze {short} Image",
            key=f"btn_{short}",
            use_container_width=True,
        )

        if uploaded:
            img_preview = Image.open(uploaded)
            st.image(img_preview, caption="📤 Uploaded Image", use_container_width=True)

    with col2:
        if uploaded and analyze_btn:
            img = Image.open(uploaded)

            with st.spinner(f"🤖 Running {short} AI inference (EfficientNetB2 FP16)…"):
                try:
                    interp  = load_model(str(model_file))
                    preds   = run_inference(interp, img)
                    results = top_predictions(preds, classes, colors, top_k=top_k)
                    top     = results[0]
                    severity, sev_icon = SEVERITY_MAP.get(top["label"], ("Unknown", "❓"))
                    ann_img = annotate_image(img, top["label"], top["confidence"], top["color"])

                    st.success(f"✅ {short} AI inference complete!")

                    # ── Primary result card ──
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="color:#64748b; font-size:0.72rem; text-transform:uppercase;
                                    letter-spacing:0.1em; margin-bottom:8px;">PRIMARY DIAGNOSIS</div>
                        <div class="result-label" style="color:{top['color']};">{top['label']}</div>
                        <div class="result-conf">
                            Confidence: <strong style="color:white;">{top['confidence']*100:.1f}%</strong>
                        </div>
                        <div style="margin-top:10px;">
                            <span style="background:rgba(255,255,255,0.05);
                                         border:1px solid rgba(255,255,255,0.1);
                                         border-radius:20px; padding:4px 14px;
                                         font-size:0.82rem; color:#e2e8f0;">
                                {sev_icon} Severity: <strong>{severity}</strong>
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Confidence bars ──
                    st.markdown(
                        "<div style='margin-top:14px;'><strong style='color:#e2e8f0;'>All Predictions</strong></div>",
                        unsafe_allow_html=True,
                    )
                    for r in results:
                        pct = r["confidence"] * 100
                        sev, sev_ico = SEVERITY_MAP.get(r["label"], ("Unknown", "❓"))
                        st.markdown(f"""
                        <div style="margin:7px 0;">
                            <div style="display:flex; justify-content:space-between;
                                        align-items:center; margin-bottom:3px;">
                                <span style="color:#e2e8f0; font-size:0.84rem; font-weight:500;">
                                    #{r['rank']} {r['label']}
                                </span>
                                <span style="color:{r['color']}; font-size:0.84rem; font-weight:700;">
                                    {pct:.1f}%
                                </span>
                            </div>
                            <div class="pred-bar">
                                <div class="pred-bar-fill"
                                     style="width:{pct:.1f}%;
                                            background:linear-gradient(90deg,{r['color']}99,{r['color']});">
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Annotated image ──
                    st.markdown(
                        "<div style='margin-top:18px;'><strong style='color:#e2e8f0;'>Annotated Result</strong></div>",
                        unsafe_allow_html=True,
                    )
                    st.image(ann_img, caption=f"BIHOS {short} AI — Result Overlay", use_container_width=True)

                    # ── Technical details ──
                    with st.expander("📊 Model Technical Details"):
                        inp_shape = interp.get_input_details()[0]["shape"]
                        out_shape = interp.get_output_details()[0]["shape"]
                        st.markdown(f"""
| Property | Value |
|---|---|
| Model File | `{cfg['file']}` |
| Input Shape | `{list(inp_shape)}` |
| Output Shape | `{list(out_shape)}` |
| Output Classes | `{len(classes)}` |
| TFLite Backend | `{TFLITE_BACKEND}` |
| Precision | `FP16 (float16)` |
| Top Prediction | `{top['label']}` |
| Confidence | `{top['confidence']*100:.2f}%` |
| Severity | `{severity}` |
                        """)

                except FileNotFoundError as e:
                    st.error(f"❌ Model file not found: {e}")
                except RuntimeError as e:
                    st.error(f"❌ Runtime Error: {e}")
                except Exception as e:
                    st.error(f"❌ Inference failed: {e}")
                    with st.expander("🔍 Full Error Traceback"):
                        st.code(traceback.format_exc())

        elif not uploaded:
            st.markdown(f"""
            <div style="text-align:center; padding:70px 20px;">
                <div style="font-size:4.5rem; margin-bottom:16px;">{icon}</div>
                <div style="font-size:1rem; color:#4b5563;">
                    Upload a <strong style="color:{accent};">{short.lower()}</strong> image,
                    then click <strong style="color:#6366f1;">Analyze</strong>
                </div>
                <div style="font-size:0.8rem; color:#374151; margin-top:10px;">
                    Supported: JPG · PNG · WebP · BMP
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👆 Click **Analyze** to run the AI model")


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_about():
    st.markdown("""
    <div class="hero-banner" style="padding:32px;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:2.2rem;
                    font-weight:800; color:#e2e8f0;">About BIHOS</div>
        <div style="color:#94a3b8; margin-top:6px;">
            CUET AI Health Screening System — 4 EfficientNetB2 FP16 Models
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### 🎓 Project Background
BIHOS is developed at **Chittagong University of Engineering & Technology (CUET)**.
The goal is to provide accessible, AI-powered healthcare screening.

### 🔬 Technology Stack
- **Architecture**: EfficientNetB2 (CNN)
- **Precision**: FP16 (half-precision — optimised size & speed)
- **Runtime**: TensorFlow Lite (TFLite)
- **Frontend**: Streamlit
- **Language**: Python 3.10+

### 📊 Model Summary
| Model | File | Classes | Size |
|---|---|---|---|
| Eye AI  | eye_efficientnetb2_fp16.tflite  | 4  | ~15 MB |
| Hair AI | hair_efficientnetb2_fp16.tflite | 6  | ~15 MB |
| Nail AI | nail_efficientnetb2_fp16.tflite | 6  | ~15 MB |
| Skin AI | skin_efficientnetb2_fp16.tflite | 20 | ~15 MB |
        """)

    with col2:
        st.markdown("""
### 👁️ Eye Model (4 classes)
Cataract · Diabetic Retinopathy · Glaucoma · Normal

### 💇 Hair Model (6 classes)
Alopecia Areata · Androgenetic Alopecia · Telogen Effluvium  
Normal / Healthy Hair · Dandruff / Seborrheic Dermatitis · Tinea Capitis

### 💅 Nail Model (6 classes)
Onychomycosis · Nail Psoriasis · Paronychia  
Melanonychia · Onycholysis · Normal
        """)

    with st.expander("🧴 View all 20 Skin Disease Classes"):
        for i, c in enumerate(SKIN_CLASSES, 1):
            color = SKIN_COLORS.get(c, "#6366f1")
            sev, sev_ico = SEVERITY_MAP.get(c, ("Unknown", "❓"))
            st.markdown(
                f"**{i}.** <span style='color:{color};'>{c}</span> — {sev_ico} {sev}",
                unsafe_allow_html=True
            )

    st.markdown("""
    ---
    <div class="bihos-footer">
        🏥 BIHOS · CUET AI Health Screening System<br>
        4 EfficientNetB2 FP16 Models · Eye · Hair · Nail · Skin<br>
        Developed at <strong>CUET</strong> · © 2025 · Research Use Only
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_sidebar()

    tab_home, tab_eye, tab_hair, tab_nail, tab_skin, tab_about = st.tabs([
        "🏠 Home",
        "👁️ Eye AI",
        "💇 Hair AI",
        "💅 Nail AI",
        "🧴 Skin AI",
        "ℹ️ About",
    ])

    with tab_home:  render_home()
    with tab_eye:   render_model_page("eye")
    with tab_hair:  render_model_page("hair")
    with tab_nail:  render_model_page("nail")
    with tab_skin:  render_model_page("skin")
    with tab_about: render_about()


if __name__ == "__main__":
    main()
