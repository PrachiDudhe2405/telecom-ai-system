from pathlib import Path

import pandas as pd


# Project root folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset folders and files
CELL1_DATA_DIR = BASE_DIR / "data" / "raw" / "Cell1_Data"
DATA_REAL_DIR = CELL1_DATA_DIR / "data_real"
INCIDENTS_FILE = CELL1_DATA_DIR / "data_real_incidents.txt"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "network_kpi.csv"


def load_incidents() -> pd.DataFrame:
    """
    Load the incident file and assign clear column names.
    """
    incidents_df = pd.read_csv(INCIDENTS_FILE, sep=r"\s+", header=None)
    incidents_df.columns = ["source_file", "incident_start", "incident_end"]
    return incidents_df


def add_issue_status(data_df: pd.DataFrame, source_name: str, incidents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add issue_status to one source file.

    issue_status = 1 for rows whose index falls inside an incident range.
    issue_status = 0 for all other rows.

    If incident_end is -1, the script treats it as 'until the end of the file'.
    """
    data_df["issue_status"] = 0

    source_incidents = incidents_df[incidents_df["source_file"] == source_name]

    for _, incident_row in source_incidents.iterrows():
        start_index = int(incident_row["incident_start"])
        end_index = int(incident_row["incident_end"])

        if end_index == -1:
            end_index = len(data_df) - 1

        data_df.loc[start_index:end_index, "issue_status"] = 1

    return data_df


def process_single_file(file_path: Path, incidents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load one r*.txt file and prepare it for the final combined dataset.
    """
    data_df = pd.read_csv(file_path, sep=r"\s+", header=None)
    data_df.columns = ["timestamp", "kpi_value"]

    source_name = file_path.stem
    data_df["source_file"] = source_name

    data_df = add_issue_status(data_df, source_name, incidents_df)
    return data_df


def build_network_dataset() -> pd.DataFrame:
    """
    Combine all r*.txt files from data_real into one DataFrame.
    """
    incidents_df = load_incidents()
    combined_dataframes = []

    for file_path in sorted(DATA_REAL_DIR.glob("r*.txt")):
        file_df = process_single_file(file_path, incidents_df)
        combined_dataframes.append(file_df)

    final_df = pd.concat(combined_dataframes, ignore_index=True)
    return final_df


def main() -> None:
    final_df = build_network_dataset()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Final dataset saved to: {OUTPUT_FILE}")
    print(f"Final dataset shape: {final_df.shape}")
    print("\nFirst 5 rows:")
    print(final_df.head())
    print("\nissue_status value counts:")
    print(final_df["issue_status"].value_counts())


if __name__ == "__main__":
    main()
