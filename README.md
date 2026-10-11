# AI-Powered Real-Time Phishing Detection

A machine learning project that analyzes URLs and predicts whether they may be phishing or legitimate.

## Features

- Extracts numerical features from URLs
- Identifies suspicious URL patterns
- Uses a Random Forest machine learning classifier
- Predicts potential phishing URLs
- Displays an estimated phishing probability
- Evaluates model performance using accuracy, precision, recall, and F1-score

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- UCI phishing URL dataset

## Project Structure

```text
AI-Phishing-Detection/
├── feature_extraction.py
├── train_model.py
├── predict_url.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    └── phishing_model.pkl
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Powered-Real-Time-Phishing-Detection.git
cd AI-Powered-Real-Time-Phishing-Detection
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train_model.py
```

## Run the Phishing Detector

```bash
python predict_url.py
```

Enter a URL when prompted. Type `exit` to stop the program.

## Initial Model Results

- Dataset samples used: 30,000
- Test samples: 6,000
- Accuracy: 98.87%

These are initial results from a held-out split of the selected dataset. They do not guarantee the same performance on new or real-world URLs.

## Future Improvements

- Build a Streamlit web dashboard
- Add more URL features
- Test on an independent dataset
- Improve detection of previously unseen phishing URLs

## Disclaimer

This project is for educational and research purposes. A model prediction does not guarantee that a website is safe or malicious.