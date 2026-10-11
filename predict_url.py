
from pathlib import Path

import joblib
import pandas as pd

from feature_extraction import extract_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )

    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["feature_names"]


def predict_url(url, model, feature_names):
    features = extract_features(url)

    # Keep feature order identical to training.
    X = pd.DataFrame([features], columns=feature_names)

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]

    classes = list(model.classes_)
    phishing_probability = float(
        probabilities[classes.index(0)]
    )

    if prediction == 0:
        verdict = "POTENTIAL PHISHING"
    else:
        verdict = "LIKELY LEGITIMATE"

    return {
        "verdict": verdict,
        "phishing_probability": phishing_probability,
        "features": features,
    }


def main():
    try:
        model, feature_names = load_model()
    except (FileNotFoundError, KeyError, OSError) as exc:
        print(f"Cannot load model: {exc}")
        return

    print("\nAI-Powered Phishing Detection")
    print("Enter a URL or type 'exit' to stop.")

    while True:
        url = input("\nEnter URL: ").strip()

        if url.lower() == "exit":
            print("Detector stopped.")
            break

        try:
            result = predict_url(url, model, feature_names)

            print(f"\nResult: {result['verdict']}")
            print(
                "Estimated phishing probability: "
                f"{result['phishing_probability']:.1%}"
            )

            print("\nExtracted features:")
            for name, value in result["features"].items():
                print(f"  {name}: {value}")

            print(
                "\nWarning: This is a model prediction, "
                "not a guarantee of website safety."
            )

        except (ValueError, TypeError) as exc:
            print(f"Invalid URL: {exc}")


if __name__ == "__main__":
    main()
