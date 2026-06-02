import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
NETWORK_MODEL_FILE = BASE_DIR / "models" / "network_model.pkl"
SOCIAL_MODEL_FILE = BASE_DIR / "models" / "social_model.pkl"
CAPACITY_MODEL_FILE = BASE_DIR / "models" / "capacity_model.pkl"
CELL3_DATA_DIR = BASE_DIR / "data" / "Cell3_Data"

NETWORK_THRESHOLD = 0.47

_COMPLAINT_PATTERNS = re.compile(
    r"\b(?:not working|outage|slow|drop\w*|issue|problem|"
    r"can.?t connect|no signal|down|broken|terrible|worst|"
    r"disappoint\w*|frustr\w*|error|fail\w*|keeps|awful|"
    r"bad|horrible|useless|garbage|trash|pathetic)",
    re.IGNORECASE,
)


# ── Model / data loaders ────────────────────────────────────────

@st.cache_resource
def load_network_model():
    return joblib.load(NETWORK_MODEL_FILE)


@st.cache_resource
def load_social_model():
    return joblib.load(SOCIAL_MODEL_FILE)


@st.cache_resource
def load_capacity_model():
    return joblib.load(CAPACITY_MODEL_FILE)


@st.cache_data
def load_capacity_data() -> pd.DataFrame:
    dfs = []
    for f in sorted(CELL3_DATA_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"], unit="ms")
    agg = combined.groupby(["CellID", "datetime"], as_index=False)["internet"].sum()
    agg["internet"] = agg["internet"].fillna(0)
    return agg.sort_values(["CellID", "datetime"]).reset_index(drop=True)


# ── Shared helpers ──────────────────────────────────────────────

def _clean_tweet(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _predict_complaint(model, raw_text: str) -> tuple[int, float]:
    """Return (label, model_probability).

    Keyword patterns catch morphological variants (e.g. 'dropping') that are
    absent from the TF-IDF vocabulary due to exact-match training labels.
    """
    cleaned = _clean_tweet(raw_text)
    prob = model.predict_proba([cleaned])[0][1]
    keyword_hit = bool(_COMPLAINT_PATTERNS.search(raw_text))
    label = 1 if (prob >= 0.5 or keyword_hit) else 0
    return label, prob


def _build_network_input(
    kpi_value: float,
    rolling_mean_5: float,
    rolling_std_5: float,
    kpi_diff: float,
) -> pd.DataFrame:
    deviation_from_mean = kpi_value - rolling_mean_5
    rolling_std_safe = rolling_std_5 if rolling_std_5 != 0 else 1.0
    normalized_deviation = deviation_from_mean / rolling_std_safe
    return pd.DataFrame([{
        "kpi_value": kpi_value,
        "lag_1": rolling_mean_5,
        "lag_2": rolling_mean_5,
        "rolling_mean_5": rolling_mean_5,
        "rolling_std_5": rolling_std_5,
        "rolling_mean_10": rolling_mean_5,
        "rolling_std_10": rolling_std_5,
        "kpi_diff": kpi_diff,
        "deviation_from_mean": deviation_from_mean,
        "normalized_deviation": normalized_deviation,
    }])


# ── Pages ────────────────────────────────────────────────────────

def page_network_monitor() -> None:
    st.title("Network Monitor")
    st.markdown(
        "Real-time KPI anomaly detection for **Cell 1** using a Random Forest "
        "classifier trained on historical incident data with SMOTE oversampling."
    )

    st.subheader("KPI Inputs")
    col1, col2 = st.columns(2)
    with col1:
        kpi_value = st.slider("KPI Value", 0.0, 1000.0, 500.0, 1.0)
        rolling_mean_5 = st.slider("Rolling Mean (5)", 0.0, 1000.0, 500.0, 1.0)
    with col2:
        rolling_std_5 = st.slider("Rolling Std (5)", 0.0, 300.0, 50.0, 1.0)
        kpi_diff = st.slider("KPI Diff", -500.0, 500.0, 0.0, 1.0)

    st.info(
        "Note: lag and rolling features are approximated from rolling_mean_5 for "
        "demo purposes. In production these would be computed from live KPI streams."
    )

    if st.button("Predict", key="network_predict"):
        model = load_network_model()
        input_df = _build_network_input(kpi_value, rolling_mean_5, rolling_std_5, kpi_diff)
        input_df = input_df.reindex(columns=model.feature_names_in_)
        incident_prob = model.predict_proba(input_df)[0][1]
        prediction = int(incident_prob >= NETWORK_THRESHOLD)

        st.divider()
        if prediction == 1:
            st.error(f"**INCIDENT DETECTED**\n\nIncident Probability: {incident_prob:.1%}")
        else:
            st.success(f"**NORMAL OPERATION**\n\nIncident Probability: {incident_prob:.1%}")


def page_complaint_detector() -> None:
    st.title("Complaint Detector")
    st.markdown(
        "Classify inbound customer tweets as complaints using a **TF-IDF + Logistic "
        "Regression** pipeline trained with VADER sentiment-based labels."
    )

    st.subheader("Tweet Analysis")
    tweet_text = st.text_area(
        "Enter customer tweet",
        placeholder="e.g. My internet is not working and calls keep dropping",
        height=120,
    )

    if st.button("Analyze", key="social_predict"):
        if not tweet_text.strip():
            st.warning("Please enter a tweet to analyze.")
        else:
            model = load_social_model()
            prediction, complaint_prob = _predict_complaint(model, tweet_text)
            st.divider()
            if prediction == 1:
                st.error(f"**COMPLAINT DETECTED**\n\nConfidence: {complaint_prob:.1%}")
            else:
                st.success(f"**NO COMPLAINT**\n\nConfidence: {1 - complaint_prob:.1%}")

    st.divider()
    st.subheader("Top 10 Complaint Indicators")
    st.markdown(
        "Words with the highest logistic regression coefficients for the complaint class, "
        "learned from VADER-labeled training data."
    )
    model = load_social_model()
    feature_names = model.named_steps["tfidf"].get_feature_names_out()
    coefficients = model.named_steps["logreg"].coef_[0]
    top_indices = coefficients.argsort()[-10:][::-1]
    chart_df = pd.DataFrame(
        {"Complaint Coefficient": coefficients[top_indices]},
        index=feature_names[top_indices],
    )
    st.bar_chart(chart_df)


def page_capacity_forecast() -> None:
    st.title("Capacity Forecast")
    st.markdown(
        "Monitor internet traffic load per cell and flag capacity risk "
        "at **80% of historical peak traffic** for the selected cell."
    )

    data = load_capacity_data()
    cell_ids = sorted(data["CellID"].unique().tolist())

    st.subheader("Cell Selection")
    default_idx = cell_ids.index(5161) if 5161 in cell_ids else 0
    selected_cell = st.selectbox("Select Cell ID", cell_ids, index=default_idx)

    cell_data = (
        data[data["CellID"] == selected_cell]
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    last_144 = cell_data.tail(144)
    max_traffic = cell_data["internet"].max()
    threshold = 0.8 * max_traffic
    exceeding = int((last_144["internet"] > threshold).sum())

    st.subheader(f"Last 144 Timestamps — Cell {selected_cell}")
    chart_df = (
        last_144.set_index("datetime")[["internet"]]
        .rename(columns={"internet": "Actual Traffic"})
    )
    chart_df["Capacity Threshold (80%)"] = threshold
    st.line_chart(chart_df)

    st.divider()
    st.subheader("Capacity Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Peak Traffic", f"{max_traffic:,.0f}")
    with col2:
        st.metric("Threshold (80%)", f"{threshold:,.0f}")
    with col3:
        st.metric("Last Period Traffic", f"{last_144['internet'].iloc[-1]:,.0f}")
    with col4:
        st.metric(
            "Periods Exceeding Threshold",
            exceeding,
            delta=f"{exceeding / len(last_144):.1%} of window",
            delta_color="inverse",
        )


def page_combined_decision() -> None:
    st.title("Combined Decision Engine")
    st.markdown(
        "Fuses **network KPI anomaly signals** and **customer sentiment** into a "
        "single operational priority decision across four severity levels."
    )

    col_kpi, col_tweet = st.columns(2)

    with col_kpi:
        st.subheader("Network KPI Inputs")
        kpi_value = st.slider("KPI Value", 0.0, 1000.0, 500.0, 1.0, key="comb_kpi")
        rolling_mean_5 = st.slider("Rolling Mean (5)", 0.0, 1000.0, 500.0, 1.0, key="comb_mean")
        rolling_std_5 = st.slider("Rolling Std (5)", 0.0, 300.0, 50.0, 1.0, key="comb_std")
        kpi_diff = st.slider("KPI Diff", -500.0, 500.0, 0.0, 1.0, key="comb_diff")
        st.info(
            "Note: lag and rolling features are approximated from rolling_mean_5 for "
            "demo purposes. In production these would be computed from live KPI streams."
        )

    with col_tweet:
        st.subheader("Customer Message")
        customer_message = st.text_area(
            "Enter customer tweet or message",
            placeholder="e.g. My internet is not working and calls keep dropping",
            height=220,
            key="comb_tweet",
        )

    if st.button("Run Analysis", key="combined_predict"):
        if not customer_message.strip():
            st.warning("Please enter a customer message.")
            return

        network_model = load_network_model()
        social_model = load_social_model()

        network_df = _build_network_input(kpi_value, rolling_mean_5, rolling_std_5, kpi_diff)
        network_df = network_df.reindex(columns=network_model.feature_names_in_)
        network_prob = network_model.predict_proba(network_df)[0][1]
        network_prediction = int(network_prob >= NETWORK_THRESHOLD)

        social_prediction, social_prob = _predict_complaint(social_model, customer_message)

        if network_prediction == 1 and social_prediction == 1:
            decision = "High Priority Telecom Issue"
            bg_color, text_color = "#c0392b", "white"
        elif network_prediction == 1 and social_prediction == 0:
            decision = "Technical Network Issue"
            bg_color, text_color = "#e67e22", "white"
        elif network_prediction == 0 and social_prediction == 1:
            decision = "Customer Experience Issue"
            bg_color, text_color = "#f1c40f", "#333333"
        else:
            decision = "No Major Issue Detected"
            bg_color, text_color = "#27ae60", "white"

        st.divider()
        st.subheader("Individual Model Predictions")
        net_col, soc_col = st.columns(2)
        with net_col:
            net_label = "INCIDENT" if network_prediction == 1 else "NORMAL"
            if network_prediction == 1:
                st.error(f"**Network: {net_label}**\n\nIncident probability: {network_prob:.1%}")
            else:
                st.success(f"**Network: {net_label}**\n\nIncident probability: {network_prob:.1%}")
        with soc_col:
            soc_label = "COMPLAINT" if social_prediction == 1 else "NO COMPLAINT"
            if social_prediction == 1:
                st.error(f"**Sentiment: {soc_label}**\n\nComplaint probability: {social_prob:.1%}")
            else:
                st.success(f"**Sentiment: {soc_label}**\n\nComplaint probability: {social_prob:.1%}")

        st.subheader("Final Decision")
        st.markdown(
            f"""
            <div style="
                background-color:{bg_color};
                padding:32px 24px;
                border-radius:12px;
                text-align:center;
                margin-top:8px;
            ">
                <h2 style="color:{text_color}; margin:0; font-size:1.9rem; font-weight:700;">
                    {decision}
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Entry point ─────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Telecom AI System", layout="wide")

    st.sidebar.title("Telecom AI System")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Network Monitor",
            "Complaint Detector",
            "Capacity Forecast",
            "Combined Decision",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Models: Random Forest (network anomaly) · "
        "TF-IDF + LogReg (complaint) · "
        "Random Forest (capacity)"
    )

    if page == "Network Monitor":
        page_network_monitor()
    elif page == "Complaint Detector":
        page_complaint_detector()
    elif page == "Capacity Forecast":
        page_capacity_forecast()
    elif page == "Combined Decision":
        page_combined_decision()


if __name__ == "__main__":
    main()
