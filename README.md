# Telecom AI System

## Project Overview

This is a beginner-friendly Python machine learning project for telecom operations and customer experience.

The goal is to build an AI system with two modules:

1. Mobile network quality and BTS monitoring
2. Social media rapid response and customer experience

## Project Structure

```text
telecom-ai-system/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── telecom_ai_exploration.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train_network_model.py
│   ├── train_social_model.py
│   ├── combined_decision.py
│   └── app.py
├── models/
├── requirements.txt
└── README.md
```

## Modules

### 1. Mobile Network Quality and BTS Monitoring

This module focuses on telecom network health using technical data such as:

- Signal strength
- Call drop rate
- Data latency
- BTS status
- Network congestion

The objective is to identify areas with poor performance and predict network quality issues.

### 2. Social Media Rapid Response and Customer Experience

This module focuses on customer feedback from social or support-related data such as:

- Complaint volume
- Sentiment score
- Response delay
- Escalation count

The objective is to detect customer dissatisfaction early and support faster response decisions.

## Datasets

You can place your datasets in:

- `data/raw/` for original input files
- `data/processed/` for cleaned or transformed files

Suggested example files:

- `data/raw/network_data.csv`
- `data/raw/social_data.csv`

If no dataset is available yet, the training scripts can generate simple sample data so the project stays runnable.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How To Run

### 1. Preprocess data

```bash
python src/data_preprocessing.py
```

### 2. Train the network monitoring model

```bash
python src/train_network_model.py
```

### 3. Train the social response model

```bash
python src/train_social_model.py
```

### 4. Start the Streamlit app

```bash
streamlit run src/app.py
```

## Output

After running the project:

- cleaned datasets can be saved in `data/processed/`
- trained models are saved in `models/`
- the Streamlit app can be used to test simple decision inputs

## Future Improvements

- connect to real telecom BTS monitoring data
- add real social media sentiment analysis
- improve evaluation and dashboards
- deploy the app for internal team use

## Notes

This project is intentionally simple and modular so it is easy to understand, extend, and run in VS Code.
