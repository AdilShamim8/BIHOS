import re
import sys

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Imports
content = content.replace(
    'import tensorflow as tf',
    'try:\n    import tflite_runtime.interpreter as tflite\nexcept ImportError:\n    import tensorflow.lite as tflite'
)

# 2. Update model extension in load functions
content = content.replace('best_model.keras', 'best_model.tflite')
content = content.replace('best_dental_model.keras', 'best_dental_model.tflite')
content = content.replace('best_nail_model.keras', 'best_nail_model.tflite')
content = content.replace('best_eye_model.keras', 'best_eye_model.tflite')
content = content.replace('best_oral_model.keras', 'best_oral_model.tflite')

# 3. Refactor load functions
# We need to replace the tf.keras.models.load_model block with tflite.Interpreter block
def refactor_load_func(match):
    model_key = match.group(1) # e.g. "skin"
    path_var = match.group(2)  # e.g. skin_path
    
    new_code = f"""models["{model_key}"]["model"] = tflite.Interpreter(model_path=str({path_var}))
        models["{model_key}"]["model"].allocate_tensors()
        models["{model_key}"]["loaded"] = True
        inp = models["{model_key}"]["model"].get_input_details()[0]['shape']
        models["{model_key}"]["input_shape"] = tuple(inp)"""
    return new_code

content = re.sub(
    r'models\["([^"]+)"\]\["model"\] = tf\.keras\.models\.load_model\(str\(([^)]+)\)\)\s*models\["[^"]+"\]\["loaded"\] = True\s*inp = models\["[^"]+"\]\["model"\]\.input_shape',
    refactor_load_func,
    content
)

# 4. Refactor preprocess functions
content = re.sub(
    r'inp_shape = m\.input_shape',
    r'inp_shape = models[m_key]["input_shape"]',
    content
)
# To make m_key available, we need to extract it in preprocess
def refactor_preprocess(match):
    m_key = match.group(1)
    return f'm_key = "{m_key}"\n    m = models["{m_key}"]["model"]\n    inp_shape = models["{m_key}"]["input_shape"]'

content = re.sub(
    r'm = models\["([^"]+)"\]\["model"\]\s*# Auto-detect expected spatial size from model input_shape\s*inp_shape = m\.input_shape',
    refactor_preprocess,
    content
)
content = re.sub(
    r'm = models\["([^"]+)"\]\["model"\]\s*inp_shape = m\.input_shape',
    refactor_preprocess,
    content
)

# 5. Refactor status endpoint
content = re.sub(
    r'skin_shape\s*=\s*str\(skin_m\["model"\]\.input_shape\)\s*if skin_m\["loaded"\]\s*and skin_m\["model"\]\s*else None',
    'skin_shape   = str(skin_m.get("input_shape")) if skin_m["loaded"] else None',
    content
)
content = re.sub(
    r'dental_shape\s*=\s*str\(dental_m\["model"\]\.input_shape\)\s*if dental_m\["loaded"\]\s*and dental_m\["model"\]\s*else None',
    'dental_shape = str(dental_m.get("input_shape")) if dental_m["loaded"] else None',
    content
)
content = re.sub(
    r'nail_shape\s*=\s*str\(nail_m\["model"\]\.input_shape\)\s*if nail_m\["loaded"\]\s*and nail_m\["model"\]\s*else None',
    'nail_shape   = str(nail_m.get("input_shape")) if nail_m["loaded"] else None',
    content
)
content = re.sub(
    r'eye_shape\s*=\s*str\(eye_m\["model"\]\.input_shape\)\s*if eye_m\["loaded"\]\s*and eye_m\["model"\]\s*else None',
    'eye_shape    = str(eye_m.get("input_shape")) if eye_m["loaded"] else None',
    content
)
content = re.sub(
    r'oral_shape\s*=\s*str\(oral_m\["model"\]\.input_shape\)\s*if oral_m\["loaded"\]\s*and oral_m\["model"\]\s*else None',
    'oral_shape   = str(oral_m.get("input_shape")) if oral_m["loaded"] else None',
    content
)

# 6. Refactor predict endpoints
def refactor_predict(match):
    m_key = match.group(1)
    return f"""inp   = preprocess_{m_key}(img)
        interp = models["{m_key}"]["model"]
        in_idx = interp.get_input_details()[0]['index']
        out_idx = interp.get_output_details()[0]['index']
        if inp.dtype != interp.get_input_details()[0]['dtype']:
            inp = inp.astype(interp.get_input_details()[0]['dtype'])
        interp.set_tensor(in_idx, inp)
        interp.invoke()
        preds = interp.get_tensor(out_idx)[0]"""

content = re.sub(
    r'inp\s*=\s*preprocess_([a-z]+)\(img\)\s*preds\s*=\s*models\["[^"]+"\]\["model"\]\.predict\(inp,\s*verbose=0\)\[0\](?:   # shape \(\d+,\))?',
    refactor_predict,
    content
)

# 7. Refactor model_info input_shape logic in predict endpoints
content = re.sub(
    r'm = models\["[^"]+"\]\["model"\]\s*inp_shape_str = str\(m\.input_shape\)',
    'inp_shape_str = str(models["skin"]["input_shape"])',
    content
)
content = re.sub(
    r'inp_shape_str = str\(models\["([^"]+)"\]\["model"\]\.input_shape\)',
    r'inp_shape_str = str(models["\1"]["input_shape"])',
    content
)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Refactored app.py successfully.")
