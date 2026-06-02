from pathlib import Path

import numpy as np
import pandas as pd


np.random.seed(42)

N_NORMAL = 1200
N_ISSUE = 300
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
    / "dummy_telecom_kpi.csv"
)


def build_dummy_dataset() -> pd.DataFrame:
    """Create a simple synthetic telecom KPI dataset."""
    normal_data = pd.DataFrame(
        {
            "signal_strength_dbm": np.random.normal(
                loc=-80, scale=6, size=N_NORMAL
            ),
            "latency_ms": np.random.normal(loc=40, scale=10, size=N_NORMAL),
            "throughput_mbps": np.random.normal(loc=65, scale=15, size=N_NORMAL),
            "call_drop_rate": np.random.normal(loc=1.2, scale=0.5, size=N_NORMAL),
            "issue_status": 0,
        }
    )

    issue_data = pd.DataFrame(
        {
            "signal_strength_dbm": np.random.normal(
                loc=-108, scale=7, size=N_ISSUE
            ),
            "latency_ms": np.random.normal(loc=145, scale=35, size=N_ISSUE),
            "throughput_mbps": np.random.normal(loc=12, scale=6, size=N_ISSUE),
            "call_drop_rate": np.random.normal(loc=7.0, scale=2.0, size=N_ISSUE),
            "issue_status": 1,
        }
    )

    df = pd.concat([normal_data, issue_data], ignore_index=True)

    df["throughput_mbps"] = df["throughput_mbps"].clip(lower=0.5)
    df["latency_ms"] = df["latency_ms"].clip(lower=1)
    df["call_drop_rate"] = df["call_drop_rate"].clip(lower=0)

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main() -> None:
    df = build_dummy_dataset()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dummy telecom KPI dataset saved to: {OUTPUT_PATH}")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nClass distribution:")
    print(df["issue_status"].value_counts())


if __name__ == "__main__":
    main()
