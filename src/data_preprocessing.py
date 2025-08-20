import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from config.config_loader import get_config
    config = get_config()
except ImportError:
    print("Warning: YAML configuration not found. Please install PyYAML and set up config files.")
    sys.exit(1)

def read_and_preprocess_data(input_path=None):
    """Load and preprocess the obesity dataset using YAML configuration."""
    if input_path is None:
        input_path = config.get('dataset.paths.raw_data')

    data_df = pd.read_csv(input_path)

    # Handle missing values
    missing_config = config.get_missing_value_config()
    for col in missing_config['columns_to_convert_numeric']:
        if col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors="coerce")

    for col, fill_value in missing_config['numerical_fill_strategy'].items():
        if col in data_df.columns:
            data_df[col].fillna(fill_value, inplace=True)

    # Rename columns
    column_mapping = config.get_column_mapping()
    data_df.rename(columns=column_mapping, inplace=True)

    # Calculate BMI
    bmi_config = config.get_bmi_config()
    if all(col in data_df.columns for col in [bmi_config['height_column'], bmi_config['weight_column']]):
        data_df[bmi_config['new_column_name']] = (
            data_df[bmi_config['weight_column']] /
            (data_df[bmi_config['height_column']] ** 2)
        )

    # Calculate body fat percentage
    bf_config = config.get_body_fat_config()
    required_cols = [bf_config['gender_column'], bf_config['age_column'], bf_config['bmi_column']]

    if all(col in data_df.columns for col in required_cols):
        def calculate_body_fat(row):
            if row[bf_config['gender_column']] == "Male":
                formula = bf_config['formulas']['male']
            else:
                formula = bf_config['formulas']['female']

            return (formula['bmi_coef'] * row[bf_config['bmi_column']] +
                   formula['age_coef'] * row[bf_config['age_column']] +
                   formula['constant'])

        data_df[bf_config['new_column_name']] = data_df.apply(calculate_body_fat, axis=1)

    # Map obesity types
    target_col = config.get('dataset.target.target_column')
    obesity_mapping = config.get_obesity_mapping()

    if target_col in data_df.columns:
        data_df[target_col] = data_df[target_col].str.strip()
        data_df[target_col] = data_df[target_col].replace(obesity_mapping)

    return data_df

def generate_drift_for_numerical_features(data, feature_cols, method='multiply',
                                        multiply_factor=1.2, noise_std_multiplier=0.1,
                                        random_state=42):
    """Generate drift for numerical features."""
    np.random.seed(random_state)
    drifted_data = data.copy()

    for col in feature_cols:
        if col in drifted_data.columns:
            if method == 'multiply':
                drifted_data[col] = drifted_data[col] * multiply_factor
            elif method == 'noise':
                feature_std = data[col].std()
                noise = np.random.normal(0, noise_std_multiplier * feature_std, size=len(drifted_data))
                drifted_data[col] = drifted_data[col] + noise

    return drifted_data

def generate_drift_for_categorical_features(data, feature_cols, flip_percentage=0.125,
                                          random_state=42):
    """Generate drift for categorical features."""
    np.random.seed(random_state)
    drifted_data = data.copy()

    for col in feature_cols:
        if col in drifted_data.columns:
            unique_values = drifted_data[col].unique()

            if len(unique_values) > 1:
                n_samples = len(drifted_data)
                n_to_flip = int(n_samples * flip_percentage)

                available_indices = drifted_data.index.tolist()
                flip_indices = np.random.choice(available_indices, size=n_to_flip, replace=False)

                for idx in flip_indices:
                    current_value = drifted_data.loc[idx, col]
                    other_values = [v for v in unique_values if v != current_value]
                    if other_values:
                        new_value = np.random.choice(other_values)
                        drifted_data.loc[idx, col] = new_value

    return drifted_data

def preprocess_data(train_path=None):
    """
    Main preprocessing function that generates drifted datasets.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, X_train_drifted,
                y_train_drifted, X_test_drifted, y_test_drifted)
    """
    # Use config defaults
    if train_path is None:
        train_path = config.get('dataset.paths.raw_data')

    split_config = config.get_split_config()

    # Load and preprocess data
    df = read_and_preprocess_data(train_path)

    # Get features and target
    target_col = config.get('dataset.target.target_column')
    exclude_cols = config.get_exclude_columns()
    all_features = [col for col in df.columns if col not in exclude_cols]

    X = df[all_features]
    y = df[target_col]

    # Split data
    stratify_target = y if split_config.get('stratify', True) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_config['test_size'],
        random_state=split_config['random_state'],
        stratify=stratify_target
    )

    # Get feature lists from config
    numerical_features = [f for f in config.get_numerical_features() if f in X_train.columns]
    categorical_features = [f for f in config.get_categorical_features() if f in X_train.columns]

    # Generate drifted training data
    drift_config = config.get_drift_config()
    X_train_drifted = X_train.copy()

    if numerical_features:
        num_config = drift_config['numerical']['train']
        X_train_drifted = generate_drift_for_numerical_features(
            X_train_drifted, numerical_features,
            method=num_config['method'],
            multiply_factor=num_config['multiply_factor'],
            noise_std_multiplier=num_config['noise_std_multiplier'],
            random_state=num_config['random_state']
        )

    if categorical_features:
        cat_config = drift_config['categorical']['train']
        X_train_drifted = generate_drift_for_categorical_features(
            X_train_drifted, categorical_features,
            flip_percentage=cat_config['flip_percentage'],
            random_state=cat_config['random_state']
        )

    y_train_drifted = y_train.copy()

    # Generate drifted test data
    X_test_drifted = X_test.copy()

    if numerical_features:
        num_config = drift_config['numerical']['test']
        X_test_drifted = generate_drift_for_numerical_features(
            X_test_drifted, numerical_features,
            method=num_config['method'],
            multiply_factor=num_config['multiply_factor'],
            noise_std_multiplier=num_config['noise_std_multiplier'],
            random_state=num_config['random_state']
        )

    if categorical_features:
        cat_config = drift_config['categorical']['test']
        X_test_drifted = generate_drift_for_categorical_features(
            X_test_drifted, categorical_features,
            flip_percentage=cat_config['flip_percentage'],
            random_state=cat_config['random_state']
        )

    y_test_drifted = y_test.copy()

    paths = config.get_paths()

    train_drifted_df = X_train_drifted.copy()
    train_drifted_df[target_col] = y_train_drifted
    train_drifted_df.to_csv(paths['drifted_train'], index=False)

    test_drifted_df = X_test_drifted.copy()
    test_drifted_df[target_col] = y_test_drifted
    test_drifted_df.to_csv(paths['drifted_test'], index=False)

    # Also save original datasets
    train_original_df = X_train.copy()
    train_original_df[target_col] = y_train
    train_original_df.to_csv(paths['processed_train'], index=False)

    test_original_df = X_test.copy()
    test_original_df[target_col] = y_test
    test_original_df.to_csv(paths['processed_test'], index=False)

    return (X_train, X_test, y_train, y_test,
            X_train_drifted, y_train_drifted, X_test_drifted, y_test_drifted)

def main():
    """Test the preprocessing pipeline."""
    result = preprocess_data()
    return result

if __name__ == "__main__":
    main()
