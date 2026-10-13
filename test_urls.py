import joblib
from predict_url import predict_url

MODEL_PATH = "models/phishing_model.pkl"

saved = joblib.load(MODEL_PATH)
model = saved["model"]
feature_names = saved["feature_names"]

test_urls = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.wikipedia.org",
    "https://www.apple.com",
    "https://www.python.org",
    "https://example.com",
]

print("=" * 55)
print("       PHISHING DETECTION TEST REPORT")
print("=" * 55)

for url in test_urls:
    result = predict_url(url, model, feature_names)

    verdict = result["verdict"]
    probability = result["phishing_probability"] * 100

    print(f"\nURL: {url}")
    print(f"Prediction: {verdict}")
    print(f"Phishing probability: {probability:.2f}%")

print("\n" + "=" * 55)
print("Testing complete.")