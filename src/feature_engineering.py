from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split


def split_and_engineer_features(preprocessed_df, resample=True, random_seed=42):
    """
    Splits the preprocessed DataFrame into train and test sets, applies label encoding
    to the target column, and optionally performs random oversampling to address class imbalance.

    Parameters:
        preprocessed_df (pd.DataFrame): Cleaned data from preprocessing step
        resample (bool): Whether to apply RandomOverSampler on the training set
        random_seed (int): Seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test: Processed feature and target splits
    """
    # Convert target to numeric
    label_mapping = {
        "Underweight": 0,
        "Normal_weight": 1,
        "Overweight": 2,
        "Obesity": 3,
    }

    # Drop irrelevant features and map target labels
    df = preprocessed_df.drop(columns=["BMI", "Height", "Weight", "BodyFat_Percentage"])
    df["obesity_type"] = df["obesity_type"].map(label_mapping)

    # Set up target and features
    df.set_index("id", inplace=True)
    X = df.drop(columns=["obesity_type"])
    y = df["obesity_type"]

    # Split first to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )

    # Resample if enabled
    if resample:
        ros = RandomOverSampler(random_state=random_seed)
        X_train, y_train = ros.fit_resample(X_train, y_train)

    # Print shapes for confirmation
    print("✅ Data split complete:")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test  shape: {X_test.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   y_test  shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test
