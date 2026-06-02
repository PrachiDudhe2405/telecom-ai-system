from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from combined_decision import predict_priority


BASE_DIR = Path(__file__).resolve().parent.parent
NETWORK_MODEL_FILE = BASE_DIR / "models" / "network_model.pkl"
SOCIAL_MODEL_FILE = BASE_DIR / "models" / "social_model.joblib"


def load_model(model_path: Path):
    if model_path.exists():
        return joblib.load(model_path)
    return None


st.set_page_config(page_title="Telecom AI System", layout="centered")
st.title("Telecom AI System")
st.write("A simple AI project for telecom network monitoring and customer experience.")

network_model = load_model(NETWORK_MODEL_FILE)
social_model = load_model(SOCIAL_MODEL_FILE)

st.header("1. Network KPI Input")
kpi_value = st.number_input("KPI Value", min_value=0.0, max_value=1000.0, value=500.0)
lag_1 = st.number_input("Lag 1", min_value=0.0, max_value=1000.0, value=495.0)
lag_2 = st.number_input("Lag 2", min_value=0.0, max_value=1000.0, value=490.0)
rolling_mean_5 = st.number_input(
    "Rolling Mean 5", min_value=0.0, max_value=1000.0, value=496.0
)
rolling_std_5 = st.number_input(
    "Rolling Std 5", min_value=0.0, max_value=500.0, value=12.0
)
rolling_mean_10 = st.number_input(
    "Rolling Mean 10", min_value=0.0, max_value=1000.0, value=498.0
)
rolling_std_10 = st.number_input(
    "Rolling Std 10", min_value=0.0, max_value=500.0, value=15.0
)
kpi_diff = st.number_input("KPI Diff", min_value=-1000.0, max_value=1000.0, value=5.0)
deviation_from_mean = st.number_input(
    "Deviation From Mean", min_value=-1000.0, max_value=1000.0, value=2.0
)
normalized_deviation = st.number_input(
    "Normalized Deviation", min_value=-100.0, max_value=100.0, value=0.13
)

st.header("2. Social Response Input")
complaint_count = st.slider("Complaint Count", 0, 50, 8)
sentiment_score = st.slider("Sentiment Score", -1.0, 1.0, 0.4)
response_delay_min = st.slider("Response Delay (min)", 0, 120, 20)
escalation_count = st.slider("Escalation Count", 0, 10, 1)

if st.button("Run Decision"):
    if network_model is None or social_model is None:
        st.warning("Train both models first by running the training scripts in src/.")
    else:
        network_input = pd.DataFrame(
            [
                {
                    "kpi_value": kpi_value,
                    "lag_1": lag_1,
                    "lag_2": lag_2,
                    "rolling_mean_5": rolling_mean_5,
                    "rolling_std_5": rolling_std_5,
                    "rolling_mean_10": rolling_mean_10,
                    "rolling_std_10": rolling_std_10,
                    "kpi_diff": kpi_diff,
                    "deviation_from_mean": deviation_from_mean,
                    "normalized_deviation": normalized_deviation,
                }
            ]
        )

        social_input = pd.DataFrame(
            [
                {
                    "complaint_count": complaint_count,
                    "sentiment_score": sentiment_score,
                    "response_delay_min": response_delay_min,
                    "escalation_count": escalation_count,
                }
            ]
        )

        network_prediction = int(network_model.predict(network_input)[0])
        social_prediction = int(social_model.predict(social_input)[0])
        final_decision = predict_priority(network_prediction, social_prediction)

        st.subheader("Results")
        st.write(f"Network Prediction: {network_prediction}")
        st.write(f"Social Prediction: {social_prediction}")
        st.success(final_decision)
