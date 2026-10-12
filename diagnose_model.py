
from pathlib import Path

import joblib
import pandas as pd

from feature_extraction import extract_features, FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

saved = joblib.load(MODEL_PATH)
model = saved["model"]

print("\n=== SAVED MODEL DIAGNOSTIC ===")
print("Model classes:", model.classes_)
print("Feature names match:", saved["feature_names"] == FEATURE_NAMES)

test_urls = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://example.com",
]

for url in test_urls:
    features = extract_features(url)
    X = pd.DataFrame([features], columns=FEATURE_NAMES)

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    phishing_index = list(model.classes_).index(0)
    phishing_probability = probabilities[phishing_index] * 100

    print(f"\nURL: {url}")
    print("Predicted label:", prediction)
    print(f"Phishing probability: {phishing_probability:.2f}%")

print("\n=== TOP FEATURE IMPORTANCES ===")
importance = pd.Series(
    model.feature_importances_,
    index=FEATURE_NAMES,
).sort_values(ascending=False)

print(importance.head(10).to_string())
