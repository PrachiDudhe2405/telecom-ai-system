from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "Cell3_Data"
OUTPUT_PATH = BASE_DIR / "notebooks" / "cell3_eda.png"


def load_cell3_data() -> pd.DataFrame:
    cell3_files = sorted(DATA_DIR.glob("sms-call-internet-mi-2013-11-*.csv"))
    cell3_df = pd.concat(
        (pd.read_csv(path, usecols=["CellID", "datetime", "internet"]) for path in cell3_files),
        ignore_index=True,
    )
    cell3_df = cell3_df.dropna(subset=["internet"]).copy()

    # Collapse duplicate CellID-timestamp rows into one traffic value per time step.
    cell3_df = cell3_df.groupby(["CellID", "datetime"], as_index=False)["internet"].sum()
    cell3_df["datetime"] = pd.to_datetime(cell3_df["datetime"], unit="ms")
    return cell3_df


def build_plots(busy_cell_df: pd.DataFrame, busy_cell_id: int) -> None:
    hourly_avg = busy_cell_df.groupby("hour")["internet"].mean().reindex(range(24))
    daily_avg = busy_cell_df.groupby("day_of_week")["internet"].mean().reindex(range(7))
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

    axes[0].plot(busy_cell_df["datetime"], busy_cell_df["internet"], color="navy", linewidth=1)
    axes[0].set_title(f"Cell {busy_cell_id} Internet Traffic Across 7 Days")
    axes[0].set_xlabel("datetime")
    axes[0].set_ylabel("internet")

    axes[1].bar(hourly_avg.index, hourly_avg.values, color="steelblue")
    axes[1].set_title(f"Cell {busy_cell_id} Average Internet Traffic by Hour of Day")
    axes[1].set_xlabel("hour")
    axes[1].set_ylabel("mean internet")
    axes[1].set_xticks(range(24))

    axes[2].bar(day_labels, daily_avg.values, color="darkorange")
    axes[2].set_title(f"Cell {busy_cell_id} Average Internet Traffic by Day of Week")
    axes[2].set_xlabel("day_of_week")
    axes[2].set_ylabel("mean internet")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cell3_df = load_cell3_data()
    cell_mean_traffic = cell3_df.groupby("CellID")["internet"].mean().sort_values(ascending=False)
    busy_cell_id = int(cell_mean_traffic.idxmax())

    busy_cell_df = cell3_df[cell3_df["CellID"] == busy_cell_id].sort_values("datetime").copy()
    busy_cell_df["hour"] = busy_cell_df["datetime"].dt.hour
    busy_cell_df["day_of_week"] = busy_cell_df["datetime"].dt.dayofweek

    build_plots(busy_cell_df, busy_cell_id)

    print(f"Files loaded: {len(sorted(DATA_DIR.glob('sms-call-internet-mi-2013-11-*.csv')))}")
    print(f"Combined shape: {cell3_df.shape}")
    print(f"Busiest CellID: {busy_cell_id}")
    print(f"Weekly mean internet traffic: {cell_mean_traffic.iloc[0]:.2f}")
    print()
    print("Top 10 CellIDs by mean internet traffic:")
    print(cell_mean_traffic.head(10).to_string())
    print()
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
