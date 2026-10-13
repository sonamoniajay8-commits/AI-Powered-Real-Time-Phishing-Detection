# AI-Powered Real-Time Phishing Detection

A machine learning project that analyzes URLs and predicts whether they may be phishing or legitimate. It uses URL-based feature extraction and a Random Forest classifier, with a Streamlit dashboard for interactive predictions.

**Important:** This is an educational project. Its predictions are not a guarantee that a website is safe or malicious.

## Features

- Extracts numerical features from URLs.
- Identifies potentially suspicious URL patterns.
- Uses a Random Forest machine learning classifier.
- Predicts whether a URL may be phishing or legitimate.
- Displays an estimated phishing probability.
- Provides an interactive Streamlit dashboard.
- Evaluates model performance using accuracy, precision, recall, and F1-score.
- Includes hostname-based evaluation to test performance on URLs whose hostnames do not overlap between the training and testing sets.

## Technologies Used

- Python 3.12
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit
- UCI phishing URL dataset (PhiUSIIL)

## Project Structure

```text
AI-Phishing-Detection/
├── app.py
├── feature_extraction.py
├── train_model.py
├── predict_url.py
├── diagnose_model.py
├── inspect_model.py
├── test_urls.py
├── evaluate_by_domain.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    └── phishing_model.pkl
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sonamoniajay8-commits/AI-Powered-Real-Time-Phishing-Detection.git
cd AI-Powered-Real-Time-Phishing-Detection
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, consult the Python documentation or use the VS Code terminal with your project's existing virtual environment.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Train the Model

Run:

```powershell
python train_model.py
```

The script trains the Random Forest classifier and saves the trained model in the `models/` directory.

## Run the URL Detector

Run:

```powershell
python predict_url.py
```

Enter a URL when prompted and follow the terminal instructions. Type `exit` to stop if supported by the running program.

## Run the Web Dashboard

Start the Streamlit application:

```powershell
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the dashboard.

## Testing and Evaluation

Run the URL test script:

```powershell
python test_urls.py
```

Run the hostname-based evaluation:

```powershell
python evaluate_by_domain.py
```

Additional diagnostic scripts:

```powershell
python diagnose_model.py
python inspect_model.py
```

## Model Results

### Random train-test split

- Dataset samples selected: 30,000
- Test samples: 6,000
- Accuracy: **98.87%**

### Hostname-based evaluation

- Training rows: 23,844
- Testing rows: 6,156
- Hostnames shared between training and testing sets: 0
- Accuracy: **98.96%**

These results come from evaluations on the same dataset source. The hostname-based split reduces overlap of exact hostnames between training and testing, but it does not establish performance on every type of unseen domain or on live phishing attacks. Independent, up-to-date testing is still needed.

## Limitations

- URL features alone may not identify every phishing attack.
- Phishing techniques and malicious domains change over time.
- A high test accuracy does not guarantee reliable predictions for real-world URLs.
- The displayed phishing probability is a model estimate, not a verified measure of the actual risk.
- The system should not be used as the only security measure when deciding whether to trust a website.

## Future Improvements

- Evaluate on an independent, recent phishing dataset.
- Add more robust URL and domain features.
- Improve detection of previously unseen phishing techniques.
- Add further validation and monitoring for model performance.
- Improve dashboard usability and prediction explanations.

## Disclaimer

This project is intended for educational and research purposes only. Never rely solely on this model to decide whether a website is safe. Avoid entering passwords or sensitive information on suspicious websites.
