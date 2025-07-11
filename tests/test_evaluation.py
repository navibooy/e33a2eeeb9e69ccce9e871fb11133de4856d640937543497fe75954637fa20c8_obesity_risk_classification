import os

from src.data_preprocessing import read_and_preprocess_data
from src.evaluation import evaluate_model
from src.model_training import train_model


def test_evaluate_model_creates_reports():
    df = read_and_preprocess_data("tests/fixtures/sample_obesity_data.csv")
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
    model_path = "models/test_model_eval.pkl"
    report_path = "reports/test_metrics.txt"
    plot_path = "reports/test_confusion_matrix.png"
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(report_path):
        os.remove(report_path)
    if os.path.exists(plot_path):
        os.remove(plot_path)
    train_model(X, y, save_path=model_path, random_seed=1)
    evaluate_model(model_path, X, y, report_path=report_path, plot_path=plot_path)
    assert os.path.exists(report_path)
    assert os.path.exists(plot_path)
