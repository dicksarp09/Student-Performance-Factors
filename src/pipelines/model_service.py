import joblib
import os

# Point to your notebooks directory where models are saved
MODELS_DIR = r"C:\Users\Dickson\Student Performance Factor\notebooks"

def load_model(name: str):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
