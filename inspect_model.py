
from pathlib import Path

import joblib
import pandas as pd
from feature_extraction import extract_features, FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_model.pkl"

saved = joblib.load(MODEL_PATH)
model = saved["model"]

urls = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://example.com",
]

X = pd.DataFrame(
    [extract_features(url) for url in urls],
    columns=FEATURE_NAMES,
)

print("=== FEATURE VALUES ===")
print(pd.DataFrame(X, index=urls).to_string())

print("\n=== PREDICTIONS FROM INDIVIDUAL TREES ===")
for url, row in zip(urls, X.to_dict("records")):
    sample = pd.DataFrame([row], columns=FEATURE_NAMES)
    tree_predictions = [
        int(tree.predict(sample)[0])
        for tree in model.estimators_
    ]

    phishing_votes = tree_predictions.count(0)
    legitimate_votes = tree_predictions.count(1)

    print(f"\n{url}")
    print("Phishing votes:", phishing_votes)
    print("Legitimate votes:", legitimate_votes)
