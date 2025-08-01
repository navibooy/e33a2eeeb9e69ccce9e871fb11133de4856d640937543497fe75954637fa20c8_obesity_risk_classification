import os
import time
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

# Define your feature groups
numerical_cols = [
    "Age",
    "veg_consumption",
    "main_meals_daily",
    "water_daily",
    "physical_weekly",
    "tech_usage_daily",
]
ordinal_cols = ["snack_consumption", "alcohol_consumption"]
binary_cols = [
    "Gender",
    "family_history_with_overweight",
    "highcal_consumption",
    "SMOKE",
    "track_cal_intake",
]
categorical_cols = ["transport_mode"]
onehot_cols = binary_cols + categorical_cols

# Encoders
ordinal = OrdinalEncoder(
    categories=[
        ["no", "Sometimes", "Frequently", "Always"],
        ["no", "Sometimes", "Frequently"],
    ],
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)

onehot = OneHotEncoder(handle_unknown="ignore")

# ColumnTransformer
encoder = ColumnTransformer(
    transformers=[
        ("numerical", "passthrough", numerical_cols),
        ("ordinal", ordinal, ordinal_cols),
        ("onehot", onehot, onehot_cols),
    ]
)


def train_model(X_train_path, y_train_path, model_path="models/model.pkl", random_seed=42):
    """
    Trains an XGBoost classifier using a pipeline with feature encoders and saves it.

    Parameters:
        X_train_path (str): Path to X_train.pkl
        y_train_path (str): Path to y_train.pkl
        model_path (str): Path where the model will be saved
        random_seed (int): Seed for reproducibility

    Returns:
        sklearn.pipeline.Pipeline: Trained model pipeline
    """
    # Load data
    X_train = joblib.load(X_train_path)
    y_train = joblib.load(y_train_path)

    # Model setup
    xgb = XGBClassifier(
        booster="gbtree",
        random_state=random_seed,
        learning_rate=0.3,
        max_depth=7,
        subsample=0.75,
        reg_lambda=0.1,
        reg_alpha=0,
    )

    model_pipeline = Pipeline([("encoder", encoder), ("xgb", xgb)])

    # Train
    start = time.time()
    model_pipeline.fit(X_train, y_train)
    end = time.time()

    # Save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_pipeline, model_path)

    print(f"Model trained and saved to: {model_path}")
    print(f"Training time: {end - start:.2f} seconds")

    return model_pipeline
