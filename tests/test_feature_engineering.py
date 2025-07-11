from src.data_preprocessing import read_and_preprocess_data
from src.feature_engineering import split_and_engineer_features


def test_split_shapes_match():
    df = read_and_preprocess_data("tests/fixtures/sample_obesity_data.csv")
    X_train, X_test, y_train, y_test = split_and_engineer_features(df)

    # Validate label/feature alignment
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # Ensure consistent number of features
    assert X_train.shape[1] == X_test.shape[1]

    # Confirm non-empty splits
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0
    assert y_train.shape[0] > 0 and y_test.shape[0] > 0

    # Avoid data leakage from duplicates
    assert X_train.index.is_unique and X_test.index.is_unique
    assert y_train.index.is_unique and y_test.index.is_unique
