
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from urllib.parse import urlparse
from feature_extraction import extract_features

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Phishing Detection",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ AI-Powered Real-Time Phishing Detection")

st.write(
    "Enter a URL to analyze its features and estimate "
    "whether it may be associated with phishing."
)

st.info(
    "This application analyzes URL patterns only. "
    "It does not visit or block websites, and predictions "
    "cannot guarantee that a website is safe."
)

# --------------------------------------------------
# Load model correctly from the saved dictionary
# --------------------------------------------------
MODEL_PATH = Path("models") / "phishing_model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python train_model.py"
        )

    saved_data = joblib.load(MODEL_PATH)

    if isinstance(saved_data, dict):
        model = saved_data["model"]
        feature_names = saved_data.get("feature_names")
    else:
        model = saved_data
        feature_names = None

    return model, feature_names


# --------------------------------------------------
# URL input
# --------------------------------------------------
url = st.text_input(
    "Enter URL to check",
    placeholder="https://example.com"
)

check_button = st.button(
    "🔍 Check URL",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# Prediction and popup notifications
# --------------------------------------------------
if check_button:
    cleaned_url = url.strip()

    if not cleaned_url:
        st.warning("Please enter a URL first.")

    else:
        try:
            parsed_url = urlparse(cleaned_url)

            if (
                parsed_url.scheme not in ("http", "https")
                or not parsed_url.hostname
                or "." not in parsed_url.hostname
            ):
                st.error(
                    "Please enter a valid HTTP or HTTPS URL, "
                    "such as https://example.com"
                )
                st.stop()

            with st.spinner("Analyzing URL..."):
                model, feature_names = load_model()

                # Extract URL features
                features = extract_features(cleaned_url)

                # Ensure features are in the correct order
                if isinstance(features, pd.DataFrame):
                    feature_df = features.copy()
                else:
                    feature_df = pd.DataFrame([features])

                if feature_names is not None:
                    feature_df = feature_df.reindex(
                        columns=list(feature_names),
                        fill_value=0
                    )

                # Predict using the actual classifier
                prediction = int(model.predict(feature_df)[0])

                # Calculate phishing score using class labels
                classes = list(model.classes_)
                phishing_index = classes.index(0)

                phishing_score = float(
                    model.predict_proba(feature_df)[0][phishing_index]
                ) * 100

            # Display result
            if prediction == 0:
                st.toast(
                    "🚨 Potential phishing detected!",
                    icon="🚨"
                )

                st.error(
                    "🚨 POTENTIALLY UNSAFE — POSSIBLE PHISHING"
                )

                st.metric(
                    "Estimated phishing score",
                    f"{phishing_score:.2f}%"
                )

                st.warning(
                    "Avoid entering passwords, payment details, "
                    "or personal information unless you have "
                    "independently verified the website."
                )

            elif prediction == 1:
                st.toast(
                    "✅ No phishing indicators detected.",
                    icon="✅"
                )

                st.success(
                    "✅ LIKELY LEGITIMATE — NO PHISHING "
                    "INDICATORS DETECTED"
                )

                st.metric(
                    "Estimated phishing score",
                    f"{phishing_score:.2f}%"
                )

                st.info(
                    "This is a model estimate, not proof of safety. "
                    "A legitimate-looking URL can still be malicious."
                )

            else:
                st.warning(
                    f"The model returned an unexpected class: {prediction}"
                )

        except FileNotFoundError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                "Could not analyze this URL. "
                "Please check the URL and try again."
            )
            st.caption(f"Technical details: {error}")


st.divider()

st.caption(
    "AI-Powered Real-Time Phishing Detection | "
    "Predictions are estimates, not security guarantees."
)
