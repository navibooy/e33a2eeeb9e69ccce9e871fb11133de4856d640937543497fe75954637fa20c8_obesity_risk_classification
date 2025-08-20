import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from imblearn.over_sampling import RandomOverSampler
from config.config_loader import get_config
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

config = get_config()

def engineer_features_from_splits(X_train, X_test, y_train, y_test, resample=None, random_seed=None):
    """
    Apply feature engineering to existing train/test splits.

    Parameters:
        X_train, X_test, y_train, y_test: Existing data splits
        resample (bool, optional): Whether to apply resampling. Uses config if None.
        random_seed (int, optional): Random seed. Uses config if None.

    Returns:
        X_train, X_test, y_train, y_test: Engineered feature and target splits
    """
    if config is not None:
        fe_config = config.get('feature_engineering.pipeline')
        if resample is None:
            resample = fe_config['resampling']['enabled']
        if random_seed is None:
            random_seed = fe_config['resampling']['random_state']

        columns_to_drop = fe_config['columns_to_drop']
        label_mapping = fe_config['label_mapping']
    else:
        if resample is None:
            resample = True
        if random_seed is None:
            random_seed = 42

        columns_to_drop = ["BMI", "Height", "Weight", "BodyFat_Percentage"]
        label_mapping = {
            "Underweight": 0,
            "Normal_weight": 1,
            "Overweight": 2,
            "Obesity": 3,
        }

    X_train_eng = X_train.copy()
    X_test_eng = X_test.copy()

    # Drop specified columns
    for col in columns_to_drop:
        if col in X_train_eng.columns:
            X_train_eng = X_train_eng.drop(columns=[col])
        if col in X_test_eng.columns:
            X_test_eng = X_test_eng.drop(columns=[col])

    # Apply label mapping
    y_train_eng = y_train.map(label_mapping)
    y_test_eng = y_test.map(label_mapping)

    # Apply resampling to training data only
    if resample:
        ros = RandomOverSampler(random_state=random_seed)
        X_train_eng, y_train_eng = ros.fit_resample(X_train_eng, y_train_eng)

    return X_train_eng, X_test_eng, y_train_eng, y_test_eng

def main():
    """Test feature engineering pipeline."""
    from src.data_preprocessing import preprocess_data

    preprocessing_result = preprocess_data()
    X_train, X_test, y_train, y_test, X_train_drifted, y_train_drifted, X_test_drifted, y_test_drifted = preprocessing_result

    X_train_eng, X_test_eng, y_train_eng, y_test_eng = engineer_features_from_splits(
        X_train, X_test, y_train, y_test
    )

    X_train_drift_eng, X_test_drift_eng, y_train_drift_eng, y_test_drift_eng = engineer_features_from_splits(
        X_train_drifted, X_test_drifted, y_train_drifted, y_test_drifted
    )

    # print("Feature engineering completed!")
    # print(f"Original - Train: {X_train_eng.shape}, Test: {X_test_eng.shape}")
    # print(f"Drifted  - Train: {X_train_drift_eng.shape}, Test: {X_test_drift_eng.shape}")

    return {
        'original': (X_train_eng, X_test_eng, y_train_eng, y_test_eng),
        'drifted': (X_train_drift_eng, X_test_drift_eng, y_train_drift_eng, y_test_drift_eng)
    }

if __name__ == "__main__":
    main()
