
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from feature_extraction import extract_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_model.pkl"

st.set_page_config(
    page_title="AI Phishing Detector",
    page_icon="🛡️",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run python train_model.py first."
        )

    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["feature_names"]


st.title("🛡️ AI-Powered Phishing Detection")
st.write(
    "Analyze a URL using a machine learning model "
    "to estimate whether it may be phishing."
)

st.info(
    "This tool analyzes URL patterns. It does not visit the website "
    "or guarantee that a URL is safe."
)

try:
    model, feature_names = load_model()
except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()

url = st.text_input(
    "Enter a URL",
    placeholder="https://example.com",
)

if st.button("Check URL", type="primary", use_container_width=True):
    if not url.strip():
        st.warning("Please enter a URL first.")
    else:
        try:
            features = extract_features(url)
            feature_table = pd.DataFrame(
                [features],
                columns=feature_names,
            )

            prediction = int(model.predict(feature_table)[0])
            probabilities = model.predict_proba(feature_table)[0]
            classes = list(model.classes_)

            phishing_probability = float(
                probabilities[classes.index(0)]
            )

            st.subheader("Analysis Result")

            if prediction == 0:
                st.error("⚠️ Potential Phishing")
            else:
                st.success("✅ Likely Legitimate")

            st.metric(
                "Estimated phishing probability",
                f"{phishing_probability:.1%}",
            )

            st.progress(phishing_probability)

            st.caption(
                "This percentage is the model's estimated probability, "
                "not a verified measure of real-world risk."
            )

            with st.expander("View extracted URL features"):
                st.dataframe(
                    pd.DataFrame(
                        list(features.items()),
                        columns=["Feature", "Value"],
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

        except (ValueError, TypeError) as exc:
            st.warning(f"Could not analyze this input: {exc}")

st.divider()
st.caption(
    "Educational project • Never enter passwords or sensitive data. "
    "A legitimate-looking result does not guarantee safety."
)
