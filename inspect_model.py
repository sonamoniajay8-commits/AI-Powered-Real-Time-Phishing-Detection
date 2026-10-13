
from pathlib import Path
import warnings

import joblib
import pandas as pd

from feature_extraction import extract_features, FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_model.pkl"

# Load the saved model
saved = joblib.load(MODEL_PATH)
model = saved["model"]

urls = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://example.com",
]

# Extract features as dictionaries, then build the DataFrame
feature_rows = [extract_features(url) for url in urls]
X = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)

print("=== FEATURE VALUES ===")
print(X.set_axis(urls).to_string())

print("\n=== PREDICTIONS FROM INDIVIDUAL TREES ===")

for url, (_, row) in zip(urls, X.iterrows()):
    # Individual trees were trained without feature names
    sample = row.to_numpy(dtype=float).reshape(1, -1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tree_predictions = [
            int(tree.predict(sample)[0])
            for tree in model.estimators_
        ]

    phishing_votes = tree_predictions.count(0)
    legitimate_votes = tree_predictions.count(1)

    print(f"\nURL: {url}")
    print("Phishing votes:", phishing_votes)
    print("Legitimate votes:", legitimate_votes)

print("\n=== RANDOM FOREST FINAL PREDICTIONS ===")

for url, (_, row) in zip(urls, X.iterrows()):
    sample = pd.DataFrame([row], columns=FEATURE_NAMES)
    probabilities = model.predict_proba(sample)[0]
    phishing_index = list(model.classes_).index(0)
    predicted_label = int(model.predict(sample)[0])

    result = "Potential Phishing" if predicted_label == 0 else "Likely Legitimate"

    print(f"\nURL: {url}")
    print("Prediction:", result)
    print(
        f"Phishing probability: "
        f"{probabilities[phishing_index] * 100:.2f}%"
    )
