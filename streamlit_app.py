"""
BIHOS — Bangladesh Intelligent Healthcare Operating System
Streamlit Web Application — streamlit_app.py

🏥 Full end-to-end medical AI inference system
📦 Models: Skin (20 classes) | Dental (3) | Nail (6) | Eye (4) | Oral (6)
🌐 Live deployment target: https://bihos.onrender.com/
"""

import os
import io
import base64
import traceback
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ── TensorFlow / TFLite ───────────────────────────────────────────────────────
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
MODELS_DIR = BASE_DIR / "models"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — must be first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BIHOS · Bangladesh Intelligent Healthcare Operating System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://bihos.onrender.com/",
        "Report a bug": "https://bihos.onrender.com/",
        "About": "BIHOS — Bangladesh Intelligent Healthcare Operating System. AI-powered medical imaging analysis.",
    },
)

# ══════════════════════════════════════════════════════════════════════════════
# LABEL CLASSES
# ══════════════════════════════════════════════════════════════════════════════
SKIN_CLASSES = [
    "Acne", "Actinic Carcinoma", "Atopic Dermatitis", "Bullous Disease",
    "Cellulitis", "Eczema", "Drug Eruptions", "Herpes HPV", "Light Diseases",
    "Lupus", "Melanoma", "Poison IVY", "Psoriasis", "Benign Tumors",
    "Systemic Disease", "Ringworm", "Urticarial Hives", "Vascular Tumors",
    "Vasculitis", "Viral Infections",
]

DENTAL_CLASSES = [
    "Dental Caries / Cavities",
    "Missing Teeth",
    "Crowned Teeth / Dental Crowns",
]

NAIL_CLASSES = [
    "Onychomycosis (Fungal Infection)",
    "Nail Psoriasis",
    "Paronychia (Bacterial/Fungal Inflammation)",
    "Melanonychia (Pigmentation/Stripes)",
    "Onycholysis (Nail Separation)",
    "Normal / Healthy Nail",
]

EYE_CLASSES = [
    "Cataract",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Normal / Healthy Eye",
]

ORAL_CLASSES = [
    "Oral Squamous Cell Carcinoma",
    "Leukoplakia",
    "Oral Lichen Planus",
    "Aphthous Ulcer (Canker Sore)",
    "Oral Candidiasis (Thrush)",
    "Normal / Healthy Oral Cavity",
]

# ══════════════════════════════════════════════════════════════════════════════
# CLASS COLORS
# ══════════════════════════════════════════════════════════════════════════════
CLASS_COLORS = {
    "Acne": "#f59e0b", "Actinic Carcinoma": "#ef4444", "Atopic Dermatitis": "#f97316",
    "Bullous Disease": "#8b5cf6", "Cellulitis": "#ef4444", "Eczema": "#f97316",
    "Drug Eruptions": "#ec4899", "Herpes HPV": "#6366f1", "Light Diseases": "#06b6d4",
    "Lupus": "#ef4444", "Melanoma": "#dc2626", "Poison IVY": "#22c55e",
    "Psoriasis": "#f59e0b", "Benign Tumors": "#10b981", "Systemic Disease": "#6366f1",
    "Ringworm": "#f59e0b", "Urticarial Hives": "#f97316", "Vascular Tumors": "#8b5cf6",
    "Vasculitis": "#ef4444", "Viral Infections": "#06b6d4",
}

DENTAL_COLORS = {
    "Dental Caries / Cavities": "#0d9488",
    "Missing Teeth": "#6366f1",
    "Crowned Teeth / Dental Crowns": "#f59e0b",
}

NAIL_COLORS = {
    "Onychomycosis (Fungal Infection)": "#f43f5e",
    "Nail Psoriasis": "#f59e0b",
    "Paronychia (Bacterial/Fungal Inflammation)": "#ef4444",
    "Melanonychia (Pigmentation/Stripes)": "#8b5cf6",
    "Onycholysis (Nail Separation)": "#6366f1",
    "Normal / Healthy Nail": "#10b981",
}

EYE_COLORS = {
    "Cataract": "#f59e0b",
    "Diabetic Retinopathy": "#ef4444",
    "Glaucoma": "#8b5cf6",
    "Normal / Healthy Eye": "#10b981",
}

ORAL_COLORS = {
    "Oral Squamous Cell Carcinoma": "#dc2626",
    "Leukoplakia": "#f97316",
    "Oral Lichen Planus": "#8b5cf6",
    "Aphthous Ulcer (Canker Sore)": "#eab308",
    "Oral Candidiasis (Thrush)": "#f43f5e",
    "Normal / Healthy Oral Cavity": "#10b981",
}

# ══════════════════════════════════════════════════════════════════════════════
# SEVERITY MAPPING
# ══════════════════════════════════════════════════════════════════════════════
SEVERITY_MAP = {
    # Skin
    "Acne": ("Low", "🟡"), "Actinic Carcinoma": ("High", "🔴"),
    "Atopic Dermatitis": ("Medium", "🟠"), "Bullous Disease": ("High", "🔴"),
    "Cellulitis": ("High", "🔴"), "Eczema": ("Medium", "🟠"),
    "Drug Eruptions": ("High", "🔴"), "Herpes HPV": ("Medium", "🟠"),
    "Light Diseases": ("Low", "🟡"), "Lupus": ("High", "🔴"),
    "Melanoma": ("Critical", "🆘"), "Poison IVY": ("Low", "🟡"),
    "Psoriasis": ("Medium", "🟠"), "Benign Tumors": ("Low", "🟡"),
    "Systemic Disease": ("High", "🔴"), "Ringworm": ("Low", "🟡"),
    "Urticarial Hives": ("Medium", "🟠"), "Vascular Tumors": ("Medium", "🟠"),
    "Vasculitis": ("High", "🔴"), "Viral Infections": ("Medium", "🟠"),
    # Dental
    "Dental Caries / Cavities": ("Medium", "🟠"),
    "Missing Teeth": ("Medium", "🟠"),
    "Crowned Teeth / Dental Crowns": ("Low", "🟡"),
    # Nail
    "Onychomycosis (Fungal Infection)": ("Medium", "🟠"),
    "Nail Psoriasis": ("Medium", "🟠"),
    "Paronychia (Bacterial/Fungal Inflammation)": ("High", "🔴"),
    "Melanonychia (Pigmentation/Stripes)": ("Medium", "🟠"),
    "Onycholysis (Nail Separation)": ("Medium", "🟠"),
    "Normal / Healthy Nail": ("None", "🟢"),
    # Eye
    "Cataract": ("High", "🔴"),
    "Diabetic Retinopathy": ("High", "🔴"),
    "Glaucoma": ("Critical", "🆘"),
    "Normal / Healthy Eye": ("None", "🟢"),
    # Oral
    "Oral Squamous Cell Carcinoma": ("Critical", "🆘"),
    "Leukoplakia": ("High", "🔴"),
    "Oral Lichen Planus": ("Medium", "🟠"),
    "Aphthous Ulcer (Canker Sore)": ("Low", "🟡"),
    "Oral Candidiasis (Thrush)": ("Medium", "🟠"),
    "Normal / Healthy Oral Cavity": ("None", "🟢"),
}

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Dark Theme + Premium Glassmorphism
# ══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
/* ─── Google Fonts ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ─── Global Reset ─── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ─── Main Background ─── */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 35%, #0a1628 65%, #060612 100%);
    min-height: 100vh;
}

/* ─── Hide Streamlit branding ─── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e2e8f0;
}

/* ─── Hero Banner ─── */
.hero-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(6,182,212,0.1) 50%, rgba(16,185,129,0.08) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 20px;
    padding: 40px 50px;
    margin-bottom: 30px;
    text-align: center;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(99,102,241,0.05) 0%, transparent 70%);
    animation: pulse-bg 4s ease-in-out infinite;
}

@keyframes pulse-bg {
    0%, 100% { opacity: 0.5; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.05); }
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #06b6d4 50%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 12px;
    font-weight: 400;
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 15px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ─── Model Cards ─── */
.model-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px;
    margin: 10px 0;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.model-card:hover {
    border-color: rgba(99, 102, 241, 0.5);
    background: rgba(99, 102, 241, 0.05);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
}

/* ─── Result Cards ─── */
.result-card {
    background: linear-gradient(135deg, rgba(17,24,39,0.8) 0%, rgba(31,41,55,0.6) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 24px;
    margin: 15px 0;
    backdrop-filter: blur(20px);
}

.result-top-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e2e8f0;
}

.result-confidence {
    font-size: 1.1rem;
    color: #94a3b8;
}

/* ─── Progress Bars ─── */
.prediction-bar {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    height: 8px;
    margin: 8px 0;
    overflow: hidden;
}

.prediction-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s ease;
}

/* ─── Metric Cards ─── */
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    font-family: 'Space Grotesk', sans-serif;
}

.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

/* ─── Status Indicator ─── */
.status-online {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #10b981;
    font-size: 0.85rem;
    font-weight: 500;
}

.status-dot {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
    50% { opacity: 0.8; box-shadow: 0 0 0 6px rgba(16,185,129,0); }
}

/* ─── Section Headers ─── */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 5px;
}

.section-subheader {
    color: #64748b;
    font-size: 0.9rem;
    margin-bottom: 20px;
}

/* ─── Mission Card ─── */
.mission-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(6,182,212,0.05) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 20px;
    padding: 30px;
    margin: 20px 0;
}

.mission-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #6366f1;
}

/* ─── Warning Box ─── */
.disclaimer-box {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 15px 0;
    color: #fcd34d;
    font-size: 0.85rem;
}

/* ─── Button Styles ─── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
}

/* ─── File Uploader ─── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed rgba(99, 102, 241, 0.3) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(99, 102, 241, 0.6) !important;
    background: rgba(99, 102, 241, 0.03) !important;
}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}

/* ─── Info/Warning boxes ─── */
.stInfo {
    background: rgba(6, 182, 212, 0.08) !important;
    border: 1px solid rgba(6, 182, 212, 0.3) !important;
    border-radius: 10px !important;
    color: #67e8f9 !important;
}

/* ─── Spinner ─── */
.stSpinner {
    color: #6366f1 !important;
}

/* ─── Selectbox ─── */
.stSelectbox [data-baseweb="select"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}

/* ─── Slider ─── */
.stSlider [data-baseweb="slider"] {
    color: #6366f1 !important;
}

/* ─── Expander ─── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ─── Success/Error messages ─── */
.stSuccess {
    background: rgba(16, 185, 129, 0.1) !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    color: #6ee7b7 !important;
}

.stError {
    background: rgba(239, 68, 68, 0.1) !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    color: #fca5a5 !important;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a1a; }
::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.7); }

/* ─── Live Link Banner ─── */
.live-link-banner {
    background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(6,182,212,0.1) 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 12px;
    padding: 16px 22px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 15px 0;
}

.live-link-text {
    color: #6ee7b7;
    font-weight: 600;
    font-size: 0.95rem;
}

/* ─── Footer ─── */
.bihos-footer {
    text-align: center;
    padding: 30px;
    color: #374151;
    font-size: 0.8rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 60px;
}

/* ─── Roadmap items ─── */
.roadmap-item {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e2e8f0;
    font-size: 0.9rem;
}
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_tflite_model(model_path: str):
    """Load a TFLite model and return the interpreter."""
    if tflite is None:
        raise RuntimeError(
            "No TFLite backend found.\n"
            "This app requires Python 3.10–3.12.\n"
            "Please run: .venv312\\Scripts\\python.exe -m streamlit run streamlit_app.py"
        )
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    interp = tflite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp


def run_inference(interp, img: Image.Image):
    """Run TFLite inference. Returns numpy array of predictions."""
    inp_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]
    shape = inp_details['shape']
    h = shape[1] if shape[1] else 224
    w = shape[2] if shape[2] else 224
    img_rgb = img.convert("RGB").resize((w, h), Image.LANCZOS)
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    tensor = np.expand_dims(arr, axis=0)
    if tensor.dtype != inp_details['dtype']:
        tensor = tensor.astype(inp_details['dtype'])
    interp.set_tensor(inp_details['index'], tensor)
    interp.invoke()
    preds = interp.get_tensor(out_details['index'])[0]
    return preds


def get_top_predictions(preds, classes, colors, top_k=5):
    """Return sorted top-K predictions."""
    ranked = sorted(enumerate(preds.tolist()), key=lambda x: x[1], reverse=True)
    results = []
    for rank, (idx, conf) in enumerate(ranked[:top_k]):
        label = classes[idx] if idx < len(classes) else f"Class_{idx}"
        results.append({
            "rank": rank + 1,
            "label": label,
            "confidence": round(float(conf), 4),
            "color": colors.get(label, "#6366f1"),
        })
    return results


def annotate_image(img: Image.Image, top_label: str, top_conf: float, color_hex: str, model_name: str) -> Image.Image:
    """Add AI result overlay to image."""
    thumb = img.convert("RGB")
    tw, th = thumb.size
    scale = 500 / max(tw, th)
    thumb = thumb.resize((int(tw * scale), int(th * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(thumb)
    hex_c = color_hex.lstrip("#")
    overlay_color = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
    tw2, th2 = thumb.size
    draw.rectangle([0, 0, tw2, 58], fill=(0, 0, 0, 200))
    draw.rectangle([0, 0, tw2, 5], fill=overlay_color)
    conf_text = f"{top_label}  {top_conf*100:.1f}%"
    try:
        font_title = ImageFont.truetype("arial.ttf", 18)
        font_sub = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = font_title
    draw.text((12, 10), conf_text, fill=overlay_color, font=font_title)
    draw.text((12, 36), f"BIHOS · {model_name}  (Research use only)", fill=(180, 180, 180), font=font_sub)
    return thumb


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE PAGES
# ══════════════════════════════════════════════════════════════════════════════

MODEL_CONFIG = {
    "🧴 Skin Disease AI": {
        "file": "best_model.tflite",
        "classes": SKIN_CLASSES,
        "colors": CLASS_COLORS,
        "default_topk": 5,
        "icon": "🧴",
        "short": "Skin",
        "description": "Analyzes skin images for 20 dermatological conditions including Melanoma, Eczema, Psoriasis, and more.",
        "tip": "Upload a clear, well-lit photo of the affected skin area.",
        "examples": "Rashes, lesions, spots, discolorations, growths",
        "color": "#f59e0b",
    },
    "🦷 Dental AI": {
        "file": "best_dental_model.tflite",
        "classes": DENTAL_CLASSES,
        "colors": DENTAL_COLORS,
        "default_topk": 3,
        "icon": "🦷",
        "short": "Dental",
        "description": "Detects Dental Caries, Missing Teeth, and Crowned Teeth from dental X-rays or oral photographs.",
        "tip": "Use a dental X-ray image or a clear intraoral photo.",
        "examples": "Dental X-rays, intraoral photos",
        "color": "#0d9488",
    },
    "💅 Nail Disease AI": {
        "file": "best_nail_model.tflite",
        "classes": NAIL_CLASSES,
        "colors": NAIL_COLORS,
        "default_topk": 4,
        "icon": "💅",
        "short": "Nail",
        "description": "Identifies 6 nail conditions including Onychomycosis (fungal), Psoriasis, Paronychia, and Melanonychia.",
        "tip": "Ensure the nail is clearly visible and well-focused.",
        "examples": "Fingernail/toenail close-up photos",
        "color": "#f43f5e",
    },
    "👁️ Eye Disease AI": {
        "file": "best_eye_model.tflite",
        "classes": EYE_CLASSES,
        "colors": EYE_COLORS,
        "default_topk": 4,
        "icon": "👁️",
        "short": "Eye",
        "description": "Screens for Cataract, Diabetic Retinopathy, and Glaucoma from fundus or clinical eye images.",
        "tip": "Use retinal fundus images or clear frontal eye photos.",
        "examples": "Fundus images, OCT scans, clinical eye photos",
        "color": "#8b5cf6",
    },
    "👄 Oral Health AI": {
        "file": "best_oral_model.tflite",
        "classes": ORAL_CLASSES,
        "colors": ORAL_COLORS,
        "default_topk": 4,
        "icon": "👄",
        "short": "Oral",
        "description": "Screens for Oral Cancer (OSCC), Leukoplakia, Lichen Planus, Candidiasis and more — 6 oral conditions.",
        "tip": "Photograph the oral cavity clearly, especially the region of concern.",
        "examples": "Tongue, gums, cheek mucosa, floor of mouth",
        "color": "#ec4899",
    },
}


def render_model_page(model_key: str):
    """Render inference UI for any model."""
    cfg = MODEL_CONFIG[model_key]
    model_file = MODELS_DIR / cfg["file"]
    classes = cfg["classes"]
    colors = cfg["colors"]
    icon = cfg["icon"]
    short = cfg["short"]

    # ── Header ──
    st.markdown(f"""
    <div class="model-card" style="border-color: {cfg['color']}40; background: linear-gradient(135deg, {cfg['color']}08, transparent);">
        <div style="display:flex; align-items:center; gap:12px;">
            <span style="font-size:2.5rem">{icon}</span>
            <div>
                <div class="section-header">{model_key}</div>
                <div class="section-subheader">{cfg['description']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer ──
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>Research & Educational Use Only.</strong> This AI tool is NOT a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown(f"<div style='color:#94a3b8; font-size:0.85rem; margin-bottom:8px;'>💡 <em>{cfg['tip']}</em></div>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            f"Upload {short} image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key=f"uploader_{short}",
            help=f"Supported: {cfg['examples']}",
        )

        top_k = st.slider(
            "Top predictions to show",
            min_value=1,
            max_value=len(classes),
            value=min(cfg["default_topk"], len(classes)),
            key=f"topk_{short}",
        )

        analyze_btn = st.button(f"🔬 Analyze {short} Image", key=f"btn_{short}", use_container_width=True)

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        if uploaded and analyze_btn:
            with st.spinner(f"🤖 Loading {short} AI model and running inference…"):
                try:
                    interp = load_tflite_model(str(model_file))
                    preds = run_inference(interp, img)
                    results = get_top_predictions(preds, classes, colors, top_k=top_k)
                    top = results[0]
                    severity, sev_icon = SEVERITY_MAP.get(top["label"], ("Unknown", "❓"))

                    # Annotated image
                    ann_img = annotate_image(img, top["label"], top["confidence"], top["color"], short + " AI")

                    st.success(f"✅ Analysis complete — {short} AI inference successful")

                    # ── Top result card ──
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">PRIMARY DIAGNOSIS</div>
                        <div class="result-top-label" style="color:{top['color']}">{top['label']}</div>
                        <div class="result-confidence">Confidence: <strong style="color:white">{top['confidence']*100:.1f}%</strong></div>
                        <div style="margin-top:10px;">
                            <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:20px; padding:4px 14px; font-size:0.82rem; color:#e2e8f0;">
                                {sev_icon} Severity: <strong>{severity}</strong>
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Confidence bars ──
                    st.markdown("<div style='margin-top:15px;'><strong style='color:#e2e8f0;'>All Predictions</strong></div>", unsafe_allow_html=True)
                    for r in results:
                        conf_pct = r["confidence"] * 100
                        bar_width = conf_pct
                        sev, sev_ico = SEVERITY_MAP.get(r["label"], ("Unknown", "❓"))
                        st.markdown(f"""
                        <div style="margin:6px 0;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:3px;">
                                <span style="color:#e2e8f0; font-size:0.85rem; font-weight:500;">#{r['rank']} {r['label']}</span>
                                <span style="color:{r['color']}; font-size:0.85rem; font-weight:700;">{conf_pct:.1f}%</span>
                            </div>
                            <div class="prediction-bar">
                                <div class="prediction-bar-fill" style="width:{bar_width:.1f}%; background: linear-gradient(90deg, {r['color']}aa, {r['color']});"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Annotated image ──
                    st.markdown("<div style='margin-top:20px;'><strong style='color:#e2e8f0;'>Annotated Result</strong></div>", unsafe_allow_html=True)
                    st.image(ann_img, caption=f"BIHOS {short} AI Analysis", use_container_width=True)

                    # ── Model info expander ──
                    with st.expander("📊 Model Technical Details"):
                        inp_shape = interp.get_input_details()[0]['shape']
                        st.markdown(f"""
                        | Property | Value |
                        |---|---|
                        | Model File | `{cfg['file']}` |
                        | Input Shape | `{inp_shape}` |
                        | Output Classes | `{len(classes)}` |
                        | Backend | `{TFLITE_BACKEND}` |
                        | Top Prediction | `{top['label']}` |
                        | Confidence | `{top['confidence']*100:.2f}%` |
                        """)

                except FileNotFoundError as e:
                    st.error(f"❌ Model not found: {e}\n\nEnsure `{cfg['file']}` is in the `models/` directory.")
                except RuntimeError as e:
                    st.error(f"❌ Runtime Error: {e}")
                except Exception as e:
                    st.error(f"❌ Inference failed: {e}")
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
        elif not uploaded:
            st.markdown(f"""
            <div style="text-align:center; padding:60px 20px; color:#374151;">
                <div style="font-size:4rem; margin-bottom:15px;">{icon}</div>
                <div style="font-size:1rem; color:#4b5563;">Upload a {short.lower()} image and click <strong style="color:#6366f1;">Analyze</strong> to begin</div>
                <div style="font-size:0.82rem; color:#374151; margin-top:10px;">Supported: JPG, PNG, WebP, BMP</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👆 Click **Analyze** button to run the AI model")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:20px 0 10px;">
            <div style="font-size:2.5rem">🏥</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.3rem; font-weight:800;
                        background:linear-gradient(135deg,#6366f1,#06b6d4); -webkit-background-clip:text;
                        -webkit-text-fill-color:transparent;">BIHOS</div>
            <div style="color:#64748b; font-size:0.72rem; margin-top:2px;">Bangladesh Intelligent</div>
            <div style="color:#64748b; font-size:0.72rem;">Healthcare Operating System</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Model status
        st.markdown("**🤖 AI Models**")
        for key, cfg in MODEL_CONFIG.items():
            model_path = MODELS_DIR / cfg["file"]
            exists = model_path.exists()
            status = "🟢" if exists else "🔴"
            size_mb = f"{model_path.stat().st_size / 1e6:.1f}MB" if exists else "missing"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:6px 8px; border-radius:6px; margin:3px 0;
                        background:rgba(255,255,255,0.02);">
                <span style="font-size:0.82rem; color:#cbd5e1;">{cfg['icon']} {cfg['short']}</span>
                <span style="font-size:0.75rem; color:#64748b;">{status} {size_mb}</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Live link
        st.markdown("**🌐 Live Deployment**")
        st.markdown("""
        <div class="live-link-banner" style="flex-direction:column; gap:6px;">
            <div class="live-link-text">🚀 Production Target:</div>
            <a href="https://bihos.onrender.com/" target="_blank"
               style="color:#06b6d4; font-size:0.82rem; word-break:break-all;">
               https://bihos.onrender.com/
            </a>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Quick stats
        st.markdown("**📊 System Stats**")
        total_models = len(MODEL_CONFIG)
        loaded_models = sum(1 for cfg in MODEL_CONFIG.values() if (MODELS_DIR / cfg["file"]).exists())
        total_classes = sum(len(cfg["classes"]) for cfg in MODEL_CONFIG.values())

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Models", f"{loaded_models}/{total_models}")
        with col_b:
            st.metric("Classes", total_classes)

        st.divider()
        st.markdown("<div style='color:#374151; font-size:0.7rem; text-align:center;'>© 2025 BIHOS · CUET · Research Use Only</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_home():
    # Hero
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🏥 BIHOS</div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; color:#94a3b8; margin-top:6px; font-weight:500;">
            Bangladesh Intelligent Healthcare Operating System
        </div>
        <div class="hero-subtitle">
            AI-powered medical imaging analysis — 5 specialized deep learning models<br>
            for skin, dental, nail, eye & oral health screening
        </div>
        <div style="margin-top:18px; display:flex; gap:10px; justify-content:center; flex-wrap:wrap;">
            <span class="hero-badge" style="background:linear-gradient(135deg,#6366f1,#8b5cf6);">🧠 Deep Learning</span>
            <span class="hero-badge" style="background:linear-gradient(135deg,#0d9488,#06b6d4);">⚡ TFLite Models</span>
            <span class="hero-badge" style="background:linear-gradient(135deg,#dc2626,#f97316);">🏥 Medical AI</span>
            <span class="hero-badge" style="background:linear-gradient(135deg,#7c3aed,#6366f1);">🔬 Research Use</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4, c5 = st.columns(5)
    stats = [
        ("5", "AI Models", "#6366f1"),
        ("39", "Disease Classes", "#06b6d4"),
        ("224×224", "Input Resolution", "#10b981"),
        ("TFLite", "Runtime", "#f59e0b"),
        ("🚀", "Render Cloud", "#ec4899"),
    ]
    for col, (val, label, color) in zip([c1, c2, c3, c4, c5], stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color};">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model cards grid
    st.markdown("<div class='section-header'>🤖 Available AI Models</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subheader'>Select a model from the tabs above to begin analysis</div>", unsafe_allow_html=True)

    cols = st.columns(3)
    model_items = list(MODEL_CONFIG.items())
    for i, (name, cfg) in enumerate(model_items):
        with cols[i % 3]:
            class_list = "  ·  ".join(cfg["classes"][:3]) + ("  …" if len(cfg["classes"]) > 3 else "")
            st.markdown(f"""
            <div class="model-card" style="border-color:{cfg['color']}40;">
                <div style="font-size:2rem; margin-bottom:8px;">{cfg['icon']}</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; color:#e2e8f0; font-size:1rem;">{name}</div>
                <div style="color:#64748b; font-size:0.78rem; margin:6px 0 10px;">{cfg['description'][:80]}…</div>
                <div style="color:{cfg['color']}; font-size:0.72rem;">{len(cfg['classes'])} classes</div>
                <div style="color:#374151; font-size:0.68rem; margin-top:4px;">{class_list}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Next Mission / Roadmap
    st.markdown("""
    <div class="mission-card">
        <div class="mission-title">🎯 Our Next Mission — Make It Live!</div>
        <div style="color:#94a3b8; font-size:0.9rem; margin:10px 0 20px;">
            We are committed to making BIHOS fully accessible to the world through our production deployment.
            The goal is to bring this AI healthcare system to life at:
        </div>
        <div style="text-align:center; padding:20px; background:rgba(99,102,241,0.08); border-radius:12px; margin-bottom:20px;">
            <div style="font-size:1.8rem; font-weight:800; font-family:'Space Grotesk',sans-serif;">
                <a href="https://bihos.onrender.com/" target="_blank"
                   style="background:linear-gradient(135deg,#6366f1,#06b6d4);
                          -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                          text-decoration:none;">
                    🌐 bihos.onrender.com
                </a>
            </div>
            <div style="color:#64748b; font-size:0.85rem; margin-top:8px;">Production · Render Cloud Hosting · 24/7 Availability</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Roadmap
    st.markdown("<div class='section-header' style='font-size:1.2rem;'>🗺️ Deployment Roadmap</div>", unsafe_allow_html=True)

    roadmap = [
        ("✅", "Build & train all 5 AI models (Skin, Dental, Nail, Eye, Oral)", "#10b981", "Completed"),
        ("✅", "Develop Flask REST API backend with TFLite inference", "#10b981", "Completed"),
        ("✅", "Create premium index.html frontend (BIHOS website)", "#10b981", "Completed"),
        ("✅", "Build Streamlit web application (this app)", "#10b981", "Completed"),
        ("✅", "Containerize with Docker for Render deployment", "#10b981", "Completed"),
        ("🔄", "Deploy to https://bihos.onrender.com/ — ACTIVE MISSION", "#f59e0b", "In Progress"),
        ("⏳", "Enable real-time X-Ray detection (Chest RetinaNet model)", "#6366f1", "Planned"),
        ("⏳", "Add patient report generation (PDF export)", "#6366f1", "Planned"),
        ("⏳", "Integrate EHR (Electronic Health Record) module", "#6366f1", "Planned"),
        ("⏳", "Mobile app (Android/iOS) powered by BIHOS API", "#8b5cf6", "Future"),
        ("⏳", "Hospital integration & tele-health platform", "#ec4899", "Future"),
        ("⏳", "Multi-language support (Bengali / English)", "#06b6d4", "Future"),
    ]

    for icon, text, color, status in roadmap:
        st.markdown(f"""
        <div class="roadmap-item" style="border-left-color:{color};">
            <span style="margin-right:8px;">{icon}</span>
            <span>{text}</span>
            <span style="float:right; font-size:0.72rem; color:{color}; background:{color}20;
                         padding:2px 8px; border-radius:10px;">{status}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How to use
    with st.expander("📖 How to Use BIHOS"):
        st.markdown("""
        ### Step-by-Step Guide

        1. **Select a Model Tab** — Choose from Skin, Dental, Nail, Eye, or Oral AI above
        2. **Upload an Image** — Drag & drop or browse for your medical image (JPG/PNG/WebP)
        3. **Adjust Settings** — Set the number of top predictions you want to see
        4. **Click Analyze** — The AI model will run inference on your image
        5. **View Results** — See ranked predictions with confidence scores and severity levels
        6. **Consult a Doctor** — Always verify AI results with a qualified healthcare professional

        ### Image Quality Tips
        - Use high-resolution, clear images for best results
        - Ensure good lighting and focus
        - Avoid blurry or obscured images
        - For skin: show the affected area clearly
        - For dental: dental X-rays or clear intraoral photos work best
        - For eye: fundus images or clear frontal eye photos
        """)

    with st.expander("⚕️ Medical Disclaimer"):
        st.warning("""
        **IMPORTANT DISCLAIMER**

        BIHOS is an AI-powered research and educational tool developed at CUET (Chittagong University of Engineering & Technology).

        - This system is intended for **research and educational purposes ONLY**
        - It is **NOT a certified medical device** and should not be used for clinical diagnosis
        - AI predictions may contain errors and should **never replace professional medical advice**
        - Always consult a licensed healthcare provider for proper diagnosis and treatment
        - The developers assume no liability for any decisions made based on this system's output
        """)


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_about():
    st.markdown("""
    <div class="hero-banner" style="padding:30px;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:800; color:#e2e8f0;">
            About BIHOS
        </div>
        <div style="color:#94a3b8; margin-top:8px;">Bangladesh Intelligent Healthcare Operating System</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎓 Project Background
        BIHOS is a final year project developed at **Chittagong University of Engineering & Technology (CUET)**.
        It aims to democratize AI-powered healthcare screening in Bangladesh and beyond.

        ### 🔬 Technology Stack
        - **Deep Learning**: TensorFlow / Keras
        - **Models**: EfficientNet-based CNN (TFLite optimized)
        - **Backend**: Flask REST API
        - **Frontend**: HTML/CSS/JavaScript + Streamlit
        - **Cloud**: Render.com deployment
        - **Runtime**: TFLite for optimized inference

        ### 📊 Model Details
        | Model | File | Classes | Size |
        |---|---|---|---|
        | Skin AI | best_model.tflite | 20 | ~5MB |
        | Dental AI | best_dental_model.tflite | 3 | ~5MB |
        | Nail AI | best_nail_model.tflite | 6 | ~5MB |
        | Eye AI | best_eye_model.tflite | 4 | ~5MB |
        | Oral AI | best_oral_model.tflite | 6 | ~5MB |
        """)

    with col2:
        st.markdown("""
        ### 🌐 Live Website
        Our production deployment is available at:
        """)
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(6,182,212,0.1));
                    border:1px solid rgba(99,102,241,0.3); border-radius:12px; padding:20px; text-align:center;">
            <div style="font-size:1.3rem; font-weight:700; color:#6366f1;">🌐 bihos.onrender.com</div>
            <a href="https://bihos.onrender.com/" target="_blank"
               style="color:#06b6d4; font-size:0.85rem;">https://bihos.onrender.com/</a>
            <div style="color:#64748b; font-size:0.8rem; margin-top:8px;">24/7 Cloud Deployment · Render Free Tier</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ### 🎯 Mission
        > *"Making world-class AI healthcare accessible to everyone in Bangladesh — from villages to cities."*

        ### 🤝 Contributing
        This is an open research project. The team welcomes collaboration from:
        - Medical professionals for dataset validation
        - ML engineers for model improvement
        - Developers for feature additions
        - Healthcare organizations for real-world testing

        ### 📧 Contact
        For research collaboration or inquiries, reach out through the CUET Computer Science & Engineering department.
        """)

    st.markdown("""
    ---
    <div class="bihos-footer">
        🏥 BIHOS · Bangladesh Intelligent Healthcare Operating System<br>
        Developed at <strong>CUET</strong> · © 2025 · Research Use Only<br>
        <a href="https://bihos.onrender.com/" target="_blank" style="color:#6366f1;">bihos.onrender.com</a>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Render sidebar
    render_sidebar()

    # Main navigation via tabs
    tab_home, tab_skin, tab_dental, tab_nail, tab_eye, tab_oral, tab_about = st.tabs([
        "🏠 Home",
        "🧴 Skin AI",
        "🦷 Dental AI",
        "💅 Nail AI",
        "👁️ Eye AI",
        "👄 Oral AI",
        "ℹ️ About",
    ])

    with tab_home:
        render_home()

    with tab_skin:
        render_model_page("🧴 Skin Disease AI")

    with tab_dental:
        render_model_page("🦷 Dental AI")

    with tab_nail:
        render_model_page("💅 Nail Disease AI")

    with tab_eye:
        render_model_page("👁️ Eye Disease AI")

    with tab_oral:
        render_model_page("👄 Oral Health AI")

    with tab_about:
        render_about()


if __name__ == "__main__":
    main()
