import sys
import os
sys.path.append(os.path.dirname(__file__))

from data_preprocessing import read_and_preprocess_data
from feature_engineering import split_and_engineer_features
from model_training import train_model
from evaluation import evaluate_model

def main():
    # Step 1: Preprocess the raw data
    print("🔄 Reading and preprocessing data...")
    df = read_and_preprocess_data("data/raw/obesity_data.csv")

    # Step 2: Split and optionally resample the dataset
    print("🧠 Engineering features and splitting data...")
    X_train, X_test, y_train, y_test = split_and_engineer_features(df, resample=True)

    # Step 3: Train the model and save it
    print("🚀 Training model...")
    model = train_model(X_train, y_train)

    # Step 4: Evaluate the model and save metrics + confusion matrix
    print("📊 Evaluating model...")
    evaluate_model(
        model_path="models/model.pkl",
        X_test=X_test,
        y_test=y_test
    )

    print("✅ Pipeline execution complete.")

if __name__ == "__main__":
    main()
