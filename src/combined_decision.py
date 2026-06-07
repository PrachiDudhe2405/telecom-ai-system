from pathlib import Path
import re

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
NETWORK_MODEL_FILE = BASE_DIR / "models" / "network_model.pkl"
SOCIAL_MODEL_FILE = BASE_DIR / "models" / "social_model.pkl"

_network_model = None
_social_model = None


def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_priority(network_kpi_features: dict, customer_message: str) -> str:
    """Combine network and social predictions into one priority decision."""
    global _network_model, _social_model
    if _network_model is None:
        _network_model = joblib.load(NETWORK_MODEL_FILE)
    if _social_model is None:
        _social_model = joblib.load(SOCIAL_MODEL_FILE)

    network_df = pd.DataFrame([network_kpi_features]).copy()

    # The saved network model expects a wider feature set than the caller passes,
    # so derive the remaining columns from the provided KPI summary values.
    network_df["lag_1"] = network_df["rolling_mean_5"]
    network_df["lag_2"] = network_df["rolling_mean_5"]
    network_df["rolling_mean_10"] = network_df["rolling_mean_5"]
    network_df["rolling_std_10"] = network_df["rolling_std_5"]
    network_df["deviation_from_mean"] = (
        network_df["kpi_value"] - network_df["rolling_mean_5"]
    )
    network_df["normalized_deviation"] = network_df["deviation_from_mean"] / (
        network_df["rolling_std_5"].replace(0, 1)
    )
    network_df = network_df.reindex(columns=_network_model.feature_names_in_)
    network_prob = _network_model.predict_proba(network_df)[0][1]
    network_prediction = int(network_prob >= 0.47)

    cleaned_text = _clean_text(customer_message)
    social_prediction = int(_social_model.predict([cleaned_text])[0])

    if network_prediction == 1 and social_prediction == 1:
        return "High Priority Telecom Issue"
    if network_prediction == 1 and social_prediction == 0:
        return "Technical Network Issue"
    if network_prediction == 0 and social_prediction == 1:
        return "Customer Experience Issue"
    return "No Major Issue Detected"


def main() -> None:
    test_cases = [
        {
            "name": "Case 1: Network issue and complaint",
            "network_kpi_features": {
                "kpi_value": -5.0,
                "rolling_mean_5": -2.0,
                "rolling_std_5": 1.0,
                "kpi_diff": -5.0,
            },
            "customer_message": "My internet is not working and calls keep dropping",
        },
        {
            "name": "Case 2: Network issue only",
            "network_kpi_features": {
                "kpi_value": -5.0,
                "rolling_mean_5": -2.0,
                "rolling_std_5": 1.0,
                "kpi_diff": -5.0,
            },
            "customer_message": "Thanks for the update",
        },
        {
            "name": "Case 3: Complaint only",
            "network_kpi_features": {
                "kpi_value": -5.0,
                "rolling_mean_5": -5.0,
                "rolling_std_5": 0.0,
                "kpi_diff": -5.0,
            },
            "customer_message": "This service is the worst and I need help fixing it",
        },
        {
            "name": "Case 4: No major issue",
            "network_kpi_features": {
                "kpi_value": -5.0,
                "rolling_mean_5": -5.0,
                "rolling_std_5": 0.0,
                "kpi_diff": -5.0,
            },
            "customer_message": "Thank you for checking on my account",
        },
    ]

    for case in test_cases:
        decision = predict_priority(
            case["network_kpi_features"],
            case["customer_message"],
        )
        print(case["name"])
        print(f"Network KPI features: {case['network_kpi_features']}")
        print(f"Customer message: {case['customer_message']}")
        print(f"Final decision: {decision}")
        print()


if __name__ == "__main__":
    main()
