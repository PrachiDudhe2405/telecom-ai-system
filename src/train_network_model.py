from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import GroupShuffleSplit


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "network_kpi.csv"
MODEL_FILE = BASE_DIR / "models" / "network_model.pkl"


def load_dataset() -> pd.DataFrame:
    """Load the processed network KPI dataset."""
    return pd.read_csv(DATA_FILE)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple features for incident detection.

    Features are created separately for each source_file so that
    rolling statistics and differences stay within the same series.
    """
    df = df.copy()

    # Sort values so rolling features follow time order inside each source file.
    df = df.sort_values(["source_file", "timestamp"]).reset_index(drop=True)

    grouped_kpi = df.groupby("source_file")["kpi_value"]

    df["lag_1"] = grouped_kpi.transform(lambda series: series.shift(1))
    df["lag_2"] = grouped_kpi.transform(lambda series: series.shift(2))

    df["rolling_mean_5"] = grouped_kpi.transform(
        lambda series: series.rolling(window=5, min_periods=1).mean()
    )
    df["rolling_std_5"] = grouped_kpi.transform(
        lambda series: series.rolling(window=5, min_periods=1).std()
    )
    df["rolling_mean_10"] = grouped_kpi.transform(
        lambda series: series.rolling(window=10, min_periods=1).mean()
    )
    df["rolling_std_10"] = grouped_kpi.transform(
        lambda series: series.rolling(window=10, min_periods=1).std()
    )
    df["kpi_diff"] = grouped_kpi.transform(lambda series: series.diff())
    df["deviation_from_mean"] = df["kpi_value"] - df["rolling_mean_10"]

    rolling_std_10_safe = df["rolling_std_10"].replace(0, np.nan)
    df["normalized_deviation"] = (
        df["deviation_from_mean"] / rolling_std_10_safe
    )

    # Fill missing values created by shifts, rolling std, diff, and normalization.
    df["lag_1"] = df["lag_1"].fillna(0)
    df["lag_2"] = df["lag_2"].fillna(0)
    df["rolling_std_5"] = df["rolling_std_5"].fillna(0)
    df["rolling_std_10"] = df["rolling_std_10"].fillna(0)
    df["kpi_diff"] = df["kpi_diff"].fillna(0)
    df["normalized_deviation"] = df["normalized_deviation"].fillna(0)

    return df


def prepare_training_data(df: pd.DataFrame):
    """Build feature matrix X and target y."""
    feature_df = df[
        [
            "kpi_value",
            "lag_1",
            "lag_2",
            "rolling_mean_5",
            "rolling_std_5",
            "rolling_mean_10",
            "rolling_std_10",
            "kpi_diff",
            "deviation_from_mean",
            "normalized_deviation",
        ]
    ].copy()

    target = df["issue_status"]
    return feature_df, target


def train_network_model() -> None:
    threshold = 0.47

    df = load_dataset()
    df = create_features(df)
    X, y = prepare_training_data(df)
    groups = df["source_file"]

    # Group-based split prevents data leakage.
    # Rows from the same source file are highly related, so each full file
    # should appear in either the training set or the test set, not both.
    group_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index = next(group_split.split(X, y, groups=groups))

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    # Telecom incident rows are much fewer than normal rows.
    # This class imbalance can make the model ignore issue_status = 1.
    # SMOTE creates additional minority-class training examples to help
    # the model learn incident patterns better.
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train_resampled, y_train_resampled)

    # Threshold tuning is used because the default 0.5 cutoff can be too strict
    # for rare incidents. A lower threshold increases sensitivity to issue_status = 1.
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)
    positive_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print("Using SMOTE")
    print(f"Using custom threshold: {threshold}")
    print(f"Model saved to: {MODEL_FILE}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)
    print(f"\nIssue recall: {positive_recall:.4f}")
    print(f"True positives:  {matrix[1, 1]}")
    print(f"False positives: {matrix[0, 1]}")


if __name__ == "__main__":
    train_network_model()
