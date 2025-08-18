import pandas as pd
import zipfile
from pathlib import Path

def ensure_data_directory():
    """Create data directory if it doesn't exist."""
    Path("data/raw").mkdir(parents=True, exist_ok=True)

def download_kaggle_competition():
    """
    Download Playground Series S4E2 competition data using Kaggle CLI.

    Returns:
        bool: True if download successful, False otherwise
    """
    import subprocess

    ensure_data_directory()
    competition = "playground-series-s4e2"

    try:
        print(f"Downloading {competition} dataset...")

        subprocess.run([
            "kaggle", "competitions", "download",
            "-c", competition,
            "-p", "data/raw"
        ], capture_output=True, text=True, check=True)

        print("Download completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Kaggle download failed: {e.stderr}")
        print("Make sure you have:")
        print("1. Kaggle API installed: pip install kaggle")
        print("2. API token in ~/.kaggle/kaggle.json")
        print("3. Joined the competition")
        return False
    except FileNotFoundError:
        print("Kaggle CLI not found. Install with: pip install kaggle")
        return False

def extract_competition_files():
    """
    Extract downloaded ZIP files in data directory.

    Returns:
        list: List of extracted CSV file paths
    """
    data_dir = Path("data/raw")
    zip_files = list(data_dir.glob("*.zip"))

    if not zip_files:
        print("No ZIP files found to extract")
        return []

    extracted_files = []

    for zip_path in zip_files:
        print(f"Extracting {zip_path.name}...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        zip_path.unlink()

    csv_files = list(data_dir.glob("*.csv"))
    extracted_files = [str(f) for f in csv_files]

    print(f"Extracted {len(extracted_files)} files")
    return extracted_files

def download_dataset():
    """
    Main function to download and prepare the dataset.

    Returns:
        str: Path to the main dataset file (train.csv)
    """
    data_dir = Path("data/raw")

    train_file = data_dir / "train.csv"
    if train_file.exists():
        print(f"Dataset already exists at {train_file}")
        return str(train_file)

    if not download_kaggle_competition():
        raise RuntimeError("Failed to download dataset")

    extracted_files = extract_competition_files()

    if not extracted_files:
        raise RuntimeError("No files were extracted")

    for file_path in extracted_files:
        if "train" in Path(file_path).name.lower():
            print(f"Main dataset: {file_path}")
            return file_path

    main_file = extracted_files[0]
    print(f"Using as main dataset: {main_file}")
    return main_file

def load_dataset(file_type="train"):
    """
    Load dataset from data directory.

    Args:
        file_type (str): Type of file to load ('train', 'test', etc.)

    Returns:
        pd.DataFrame: Loaded dataset
    """
    data_dir = Path("data/raw")

    pattern = f"*{file_type}*.csv"
    matching_files = list(data_dir.glob(pattern))

    if not matching_files:
        download_dataset()
        matching_files = list(data_dir.glob(pattern))

        if not matching_files:
            raise FileNotFoundError(f"Could not find {file_type} dataset")

    file_path = matching_files[0]
    print(f"Loading {file_type} data from {file_path.name}...")

    df = pd.read_csv(file_path)
    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

    return df

def get_dataset_summary(df):
    """
    Print basic dataset information.

    Args:
        df (pd.DataFrame): Dataset to summarize
    """
    print("DATASET SUMMARY:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Data types: {df.dtypes.value_counts().to_dict()}")

def main():
    """Test the data ingestion pipeline."""
    try:
        train_df = load_dataset("train")
        get_dataset_summary(train_df)

        return train_df

    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
