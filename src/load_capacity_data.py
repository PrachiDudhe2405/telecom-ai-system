from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "Cell3_Data"


def main() -> None:
    # Load every CSV file from the Cell3 dataset folder so the model can use
    # all available days instead of a single file.
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    dataframes = [pd.read_csv(file_path) for file_path in csv_files]
    df = pd.concat(dataframes, ignore_index=True)

    # Show the overall size of the dataset so we know how many rows and columns
    # are available before any cleaning or feature engineering.
    print(f"Loaded {len(csv_files)} CSV files")
    print(f"Dataset shape: {df.shape}")

    # Print the column names to inspect what measurements are available
    # in the raw file.
    print("\nColumn names:")
    print(df.columns.tolist())

    # Print data types to verify which columns are numeric and which may need
    # conversion during preprocessing.
    print("\nData types:")
    print(df.dtypes)

    # Count missing values in each column to understand data quality issues
    # before training a model.
    print("\nNull value counts:")
    print(df.isnull().sum())

    # Print descriptive statistics for numeric columns to summarize traffic levels,
    # spread, and possible outliers in the dataset.
    print("\nDescriptive statistics:")
    print(df.describe())

    # Show how many unique cell towers are present in the dataset.
    print(f"\nUnique CellIDs: {df['CellID'].nunique()}")

    # Show how many unique timestamps are present, which helps estimate
    # the time granularity and coverage of the traffic data.
    print(f"Unique timestamps: {df['datetime'].nunique()}")

    # Convert Unix timestamps in milliseconds into pandas datetime format
    # so the time column is easier to read and work with.
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")

    # Aggregate traffic by CellID and datetime. This combines activity across
    # all country codes so each row represents total activity for one cell
    # in one time window.
    aggregated_df = (
        df.groupby(["CellID", "datetime"], as_index=False)[
            ["smsin", "smsout", "callin", "callout", "internet"]
        ]
        .sum()
    )

    # Print the shape after aggregation so we can see how many rows remain
    # once duplicate CellID/timestamp combinations are merged.
    print(f"\nShape after aggregation: {aggregated_df.shape}")

    # Check whether the aggregated internet column still contains any missing
    # values before we prepare it for modeling.
    print("\nRemaining null values in internet:")
    print(aggregated_df["internet"].isnull().sum())

    # Fill any missing internet values with 0 so the dataset is numeric
    # and ready for downstream modeling steps.
    aggregated_df["internet"] = aggregated_df["internet"].fillna(0)

    # Print basic summary values for internet traffic after cleaning
    # to understand the range and average load per cell-time window.
    print("\nInternet column summary after cleaning:")
    print(f"Min internet: {aggregated_df['internet'].min()}")
    print(f"Max internet: {aggregated_df['internet'].max()}")
    print(f"Mean internet: {aggregated_df['internet'].mean()}")

    # Extract hour of day because network traffic often follows strong
    # intraday usage cycles such as morning, afternoon, and evening peaks.
    aggregated_df["hour"] = aggregated_df["datetime"].dt.hour

    # Extract day of week because usage patterns can differ between weekdays
    # and specific days such as Friday versus Monday.
    aggregated_df["day_of_week"] = aggregated_df["datetime"].dt.dayofweek

    # Mark weekends because telecom demand often shifts on Saturdays and Sundays
    # compared with normal workdays.
    aggregated_df["is_weekend"] = aggregated_df["day_of_week"] >= 5

    # Sort by CellID and time before building lag features so each lag really
    # refers to previous observations from the same cell tower.
    aggregated_df = aggregated_df.sort_values(["CellID", "datetime"]).reset_index(
        drop=True
    )

    # lag_1 captures very recent traffic momentum from 10 minutes ago.
    aggregated_df["lag_1"] = aggregated_df.groupby("CellID")["internet"].shift(1)

    # lag_6 captures short-term hourly patterns from 1 hour earlier.
    aggregated_df["lag_6"] = aggregated_df.groupby("CellID")["internet"].shift(6)

    # lag_12 captures a slightly longer recent trend from 2 hours earlier.
    aggregated_df["lag_12"] = aggregated_df.groupby("CellID")["internet"].shift(12)

    # lag_144 captures daily seasonality by using the same cell's traffic
    # from the same time one day earlier.
    aggregated_df["lag_144"] = aggregated_df.groupby("CellID")["internet"].shift(144)

    # Fill missing lag values with 0 for the earliest timestamps where
    # historical observations are not yet available.
    aggregated_df[["lag_1", "lag_6", "lag_12", "lag_144"]] = aggregated_df[
        ["lag_1", "lag_6", "lag_12", "lag_144"]
    ].fillna(0)

    print(f"\nFinal shape after cleaning and feature engineering: {aggregated_df.shape}")
    print(
        f"Date range covered: {aggregated_df['datetime'].min()} to "
        f"{aggregated_df['datetime'].max()}"
    )

    print("\nFeature engineering preview:")
    print(
        aggregated_df[
            [
                "CellID",
                "datetime",
                "hour",
                "day_of_week",
                "is_weekend",
                "internet",
                "lag_1",
                "lag_6",
                "lag_144",
            ]
        ].head(5)
    )


if __name__ == "__main__":
    main()
