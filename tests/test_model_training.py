import os

import pandas as pd

from src.model_training import train_model


def test_train_model_creates_file():
    df = pd.read_csv("tests/fixtures/sample_obesity_data.csv")
    # Minimal label engineering for test
    df["obesity_type"] = df["NObeyesdad"].map(
        {
            "Insufficient_Weight": 0,
            "Normal_Weight": 1,
            "Overweight_Level_I": 2,
            "Overweight_Level_II": 2,
            "Obesity_Type_I": 3,
            "Obesity_Type_II": 3,
            "Obesity_Type_III": 3,
        }
    )
    X = df.drop(["NObeyesdad", "obesity_type"], axis=1)
    y = df["obesity_type"]
    model_path = "models/test_model.pkl"
    if os.path.exists(model_path):
        os.remove(model_path)
    train_model(X, y, save_path=model_path, random_seed=0)
    assert os.path.exists(model_path)
