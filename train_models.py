# train_models.py
from pathlib import Path
import joblib
import pandas as pd

# Update these imports if your training functions live in model.py instead
# from model import train_baselines, train_improved, load_artifacts  # if available
from apputil import load_processed, train_baseline, train_improved

OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

def save_artifact(obj, name: str):
    path = OUT_DIR / name
    joblib.dump(obj, path)
    print(f"Saved {name} ({path})")
    return path

def main():
    # load data (will try processed then fallback)
    print("Loading data...")
    df = load_processed()

    print("Training baseline (fast) model...")
    model_base, features_base, r2_base = train_baseline(df)
    print("Baseline R2:", r2_base)
    save_artifact({
        "model": model_base,
        "features": features_base,
        "r2": r2_base,
        "type": "baseline"
    }, "model_baseline.joblib")

    print("Training improved model (may take longer)...")
    model_improved, features_imp, r2_imp = train_improved(df)
    print("Improved model R2:", r2_imp)
    save_artifact({
        "model": model_improved,
        "features": features_imp,
        "r2": r2_imp,
        "type": "improved"
    }, "model_improved.joblib")

    print("Done. Artifacts saved in models/")

if __name__ == "__main__":
    main()