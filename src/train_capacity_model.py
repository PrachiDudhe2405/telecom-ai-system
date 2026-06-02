from math import sqrt
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "Cell3_Data"
PLOT_FILE = BASE_DIR / "notebooks" / "cell3_forecast.png"
MODEL_FILE = BASE_DIR / "models" / "capacity_model.pkl"
TARGET_CELL_ID = 5161
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_6",
    "lag_12",
    "lag_144",
]


def load_and_preprocess_data() -> pd.DataFrame:
    """Load and preprocess Cell3 data the same way as load_capacity_data.py."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    dataframes = []
    for file_path in csv_files:
        file_df = pd.read_csv(file_path)
        source_date = "-".join(file_path.stem.split("-")[-3:])
        file_df["source_date"] = pd.Timestamp(source_date)
        dataframes.append(file_df)
    df = pd.concat(dataframes, ignore_index=True)

    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")

    aggregated_df = (
        df.groupby(["CellID", "datetime", "source_date"], as_index=False)[
            ["smsin", "smsout", "callin", "callout", "internet"]
        ]
        .sum()
    )

    aggregated_df["internet"] = aggregated_df["internet"].fillna(0)
    aggregated_df["hour"] = aggregated_df["datetime"].dt.hour
    aggregated_df["day_of_week"] = aggregated_df["datetime"].dt.dayofweek
    aggregated_df["is_weekend"] = (aggregated_df["day_of_week"] >= 5).astype(int)
    aggregated_df = aggregated_df.sort_values(["CellID", "datetime"]).reset_index(
        drop=True
    )

    grouped_internet = aggregated_df.groupby("CellID")["internet"]
    aggregated_df["lag_1"] = grouped_internet.shift(1)
    aggregated_df["lag_6"] = grouped_internet.shift(6)
    aggregated_df["lag_12"] = grouped_internet.shift(12)
    aggregated_df["lag_144"] = grouped_internet.shift(144)
    aggregated_df[["lag_1", "lag_6", "lag_12", "lag_144"]] = aggregated_df[
        ["lag_1", "lag_6", "lag_12", "lag_144"]
    ].fillna(0)

    return aggregated_df


def split_train_test(cell_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use November 4-9, 2013 for training and November 10, 2013 for test."""
    train_mask = cell_df["source_date"] < pd.Timestamp("2013-11-10")
    test_mask = cell_df["source_date"] == pd.Timestamp("2013-11-10")
    return cell_df.loc[train_mask].copy(), cell_df.loc[test_mask].copy()


def save_forecast_plot(test_df: pd.DataFrame, predictions: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        test_df["datetime"],
        test_df["internet"],
        label="Actual internet",
        color="navy",
        linewidth=1.5,
    )
    ax.plot(
        test_df["datetime"],
        predictions,
        label="Predicted internet",
        color="darkorange",
        linewidth=1.5,
    )
    ax.set_title(f"Cell {TARGET_CELL_ID} Actual vs Predicted Internet Traffic on Nov 10")
    ax.set_xlabel("datetime")
    ax.set_ylabel("internet")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_capacity_model() -> None:
    df = load_and_preprocess_data()
    cell_df = df[df["CellID"] == TARGET_CELL_ID].sort_values("datetime").reset_index(
        drop=True
    )

    train_df, test_df = split_train_test(cell_df)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["internet"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["internet"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = pd.Series(model.predict(X_test), index=test_df.index, name="prediction")
    capacity_threshold = 0.8 * y_train.max()
    flagged_df = test_df[["datetime"]].copy()
    flagged_df["predicted_internet"] = predictions
    flagged_df = flagged_df[flagged_df["predicted_internet"] > capacity_threshold].copy()
    flagged_df["status"] = "At Risk"

    mae = mean_absolute_error(y_test, predictions)
    rmse = sqrt(mean_squared_error(y_test, predictions))

    save_forecast_plot(test_df, predictions)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print(f"Target CellID: {TARGET_CELL_ID}")
    print(
        "Training range: "
        f"{train_df['datetime'].min()} to {train_df['datetime'].max()}"
    )
    print(
        "Test range: "
        f"{test_df['datetime'].min()} to {test_df['datetime'].max()}"
    )
    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print()
    print("Section 1: Capacity risk flags")
    print(f"Capacity threshold: {capacity_threshold:.4f}")
    print(f"Flagged timestamps: {len(flagged_df)}")
    print("Top 5 flagged timestamps:")
    if flagged_df.empty:
        print("No timestamps exceeded the capacity threshold.")
    else:
        print(
            flagged_df.nlargest(5, "predicted_internet")[
                ["datetime", "predicted_internet", "status"]
            ].to_string(index=False)
        )
    print()
    print("Section 2: Model persistence")
    print(f"Saved forecast plot to: {PLOT_FILE}")
    print(f"Model saved to: {MODEL_FILE}")
    print("Capacity model saved")


if __name__ == "__main__":
    train_capacity_model()
