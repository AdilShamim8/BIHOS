"""
Convert all .keras models to TensorFlow Lite (.tflite) format.
Run once locally:  .venv312/Scripts/python convert_to_tflite.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONIOENCODING"] = "utf-8"

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import tensorflow as tf
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
OUT_DIR    = MODELS_DIR

models_to_convert = [
    "best_model.keras",
    "best_dental_model.keras",
    "best_eye_model.keras",
    "best_nail_model.keras",
    "best_oral_model.keras",
]

for name in models_to_convert:
    src = MODELS_DIR / name
    dst = OUT_DIR / name.replace(".keras", ".tflite")

    if not src.exists():
        print(f"[SKIP] {src} not found")
        continue

    print(f"[CONVERT] {src.name} -> {dst.name} ...")
    model = tf.keras.models.load_model(str(src))
    print(f"  Input shape : {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable float16 quantization for smaller file size
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    dst.write_bytes(tflite_model)
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"  [OK] Saved {dst.name} ({size_mb:.2f} MB)")

    # Quick validation: load and check shapes
    interp = tf.lite.Interpreter(model_path=str(dst))
    interp.allocate_tensors()
    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]
    print(f"  TFLite input : {inp_detail['shape']}  dtype={inp_detail['dtype']}")
    print(f"  TFLite output: {out_detail['shape']}  dtype={out_detail['dtype']}")
    print()

    # Free memory
    del model
    tf.keras.backend.clear_session()

print("=== All conversions done ===")
