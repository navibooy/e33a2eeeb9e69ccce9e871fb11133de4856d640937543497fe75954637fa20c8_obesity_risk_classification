import os
import pandas as pd
import joblib
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

def split_and_engineer_features(input_path, output_dir=None, resample=True, random_seed=42):
    """
    Loads preprocessed CSV, splits into train/test, encodes labels, and optionally resamples.

    Parameters:
        input_path (str): Path to preprocessed CSV file
        output_dir (str, optional): Directory where the split data will be saved as .pkl files
        resample (bool): Whether to apply RandomOverSampler on the training set
        random_seed (int): Seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test: Processed feature and target splits
    """
    df = pd.read_csv(input_path)

    label_mapping = {
        "Underweight": 0,
        "Normal_weight": 1,
        "Overweight": 2,
        "Obesity": 3,
    }

    df = df.drop(columns=["BMI", "Height", "Weight", "BodyFat_Percentage"])
    df["obesity_type"] = df["obesity_type"].map(label_mapping)

    if "id" in df.columns:
        df.set_index("id", inplace=True)

    X = df.drop(columns=["obesity_type"])
    y = df["obesity_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    if resample:
        ros = RandomOverSampler(random_state=random_seed)
        X_train, y_train = ros.fit_resample(X_train, y_train)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(X_train, os.path.join(output_dir, "X_train.pkl"))
        joblib.dump(X_test, os.path.join(output_dir, "X_test.pkl"))
        joblib.dump(y_train, os.path.join(output_dir, "y_train.pkl"))
        joblib.dump(y_test, os.path.join(output_dir, "y_test.pkl"))
        print(f"Feature splits saved to: {output_dir}")

    print("Data split complete:")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test  shape: {X_test.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   y_test  shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test
