
from urllib.parse import urlsplit

import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from feature_extraction import extract_features, FEATURE_NAMES

MAX_SAMPLES = 30000
RANDOM_STATE = 42

print("Downloading dataset...")
dataset = fetch_ucirepo(id=967)

data = pd.DataFrame({
    "url": dataset.data.features["URL"],
    "label": dataset.data.targets["label"],
})

data = data.dropna()
data["label"] = pd.to_numeric(data["label"], errors="coerce")
data = data.dropna(subset=["label"])
data["label"] = data["label"].astype(int)
data = data[data["label"].isin([0, 1])]
data = data.drop_duplicates(subset=["url"])

# Keep the evaluation manageable and reproducible.
if len(data) > MAX_SAMPLES:
    data = data.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE)

def get_hostname(url):
    try:
        value = str(url).strip()
        parsed = urlsplit(
            value if "://" in value else "//" + value
        )
        return (parsed.hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""

rows = []
labels = []
groups = []

print("Extracting features...")
for url, label in zip(data["url"], data["label"]):
    try:
        hostname = get_hostname(url)
        if not hostname:
            continue

        rows.append(extract_features(url))
        labels.append(label)
        groups.append(hostname)
    except (ValueError, TypeError):
        continue

X = pd.DataFrame(rows, columns=FEATURE_NAMES)
y = pd.Series(labels)
groups = pd.Series(groups)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training hostnames:", groups.iloc[train_idx].nunique())
print("Testing hostnames:", groups.iloc[test_idx].nunique())
print(
    "Hostnames overlap:",
    len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=18,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("Training evaluation model...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\n=== HOSTNAME-BASED EVALUATION ===")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Phishing", "Legitimate"],
    zero_division=0,
))
print("Confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))
