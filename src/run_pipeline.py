import os
import sys

sys.path.append(os.path.dirname(__file__))

from data_preprocessing import read_and_preprocess_data
from feature_engineering import split_and_engineer_features
from model_training import train_model
from evaluation import evaluate_model

def main():
    # Define paths
    raw_path = "data/raw/obesity_data.csv"
    processed_path = "data/processed/obesity_clean.csv"
    split_dir = "data/splits/"
    model_path = "models/model.pkl"
    report_path = "reports/metrics.txt"
    plot_path = "reports/confusion_matrix.png"

    # Step 1: Preprocess the raw data
    print("Reading and preprocessing data...")
    read_and_preprocess_data(input_path=raw_path, output_path=processed_path)

    # Step 2: Feature engineering and splitting
    print("Engineering features and splitting data...")
    split_and_engineer_features(
        input_path=processed_path,
        output_dir=split_dir,
        resample=True
    )

    # Step 3: Train model
    print("Training model...")
    train_model(
        X_train_path=os.path.join(split_dir, "X_train.pkl"),
        y_train_path=os.path.join(split_dir, "y_train.pkl"),
        model_path=model_path
    )

    # Step 4: Evaluate model
    print("Evaluating model...")
    evaluate_model(
        model_path=model_path,
        X_test_path=os.path.join(split_dir, "X_test.pkl"),
        y_test_path=os.path.join(split_dir, "y_test.pkl"),
        report_path=report_path,
        plot_path=plot_path
    )

    print("Pipeline execution complete.")


if __name__ == "__main__":
    main()
