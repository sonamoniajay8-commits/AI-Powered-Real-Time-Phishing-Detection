
from pathlib import Path

import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from feature_extraction import extract_features, FEATURE_NAMES


# Project settings
RANDOM_STATE = 42
MAX_SAMPLES = 30000

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "phishing_model.pkl"


def main():
    # 1. Download the dataset
    print("Downloading the phishing URL dataset...")

    dataset = fetch_ucirepo(id=967)

    urls = dataset.data.features["URL"]
    labels = dataset.data.targets["label"]

    # 2. Create the DataFrame
    data = pd.DataFrame({
        "url": urls,
        "label": labels,
    })

    # 3. Clean the dataset
    data = data.dropna()
    data["label"] = pd.to_numeric(
        data["label"], errors="coerce"
    )
    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)

    # Dataset convention:
    # 0 = phishing, 1 = legitimate
    data = data[data["label"].isin([0, 1])]
    data = data.drop_duplicates(subset=["url"])

    print(f"Usable URLs: {len(data)}")

    # 4. Sample the data if the dataset is large
    if len(data) > MAX_SAMPLES:
        data = data.sample(
            n=MAX_SAMPLES,
            random_state=RANDOM_STATE,
        )

    print(f"URLs selected for training: {len(data)}")
    print("Extracting URL features...")

    # 5. Extract features
    feature_rows = []
    valid_labels = []

    for url, label in zip(data["url"], data["label"]):
        try:
            features = extract_features(url)
            feature_rows.append(features)
            valid_labels.append(label)
        except (ValueError, TypeError):
            continue

    X = pd.DataFrame(
        feature_rows,
        columns=FEATURE_NAMES,
    )

    y = pd.Series(valid_labels, name="label")

    if len(X) < 100 or y.nunique() != 2:
        raise RuntimeError(
            "Not enough valid URLs or both label classes "
            "are not available."
        )

    print(f"Valid URLs: {len(X)}")

    # 6. Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("Training Random Forest model...")

    # 7. Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # 8. Evaluate the model
    predictions = model.predict(X_test)

    print("\n========== MODEL RESULTS ==========")

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Phishing", "Legitimate"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print("Rows = actual; columns = predicted")
    print("Class order: Phishing (0), Legitimate (1)")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # 9. Save the trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
        },
        MODEL_PATH,
    )

    print("\nModel saved successfully!")
    print(f"Model location: {MODEL_PATH}")


if __name__ == "__main__":
    main()
