import time
import os
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define your feature groups
numerical_cols = ['Age', 'veg_consumption', 'main_meals_daily', 'water_daily', 'physical_weekly', 'tech_usage_daily']
ordinal_cols = ['snack_consumption', 'alcohol_consumption']
binary_cols = ['Gender', 'family_history_with_overweight', 'highcal_consumption', 'SMOKE', 'track_cal_intake']
categorical_cols = ['transport_mode']
onehot_cols = binary_cols + categorical_cols

# Encoders
ordinal = OrdinalEncoder(
    categories=[
        ['no', 'Sometimes', 'Frequently', 'Always'],
        ['no', 'Sometimes', 'Frequently']
    ],
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

onehot = OneHotEncoder(handle_unknown='ignore')

# ColumnTransformer
encoder = ColumnTransformer(transformers=[
    ('numerical', 'passthrough', numerical_cols),
    ('ordinal', ordinal, ordinal_cols),
    ('onehot', onehot, onehot_cols)
])

def train_model(X_train, y_train, save_path="models/model.pkl", random_seed=42):
    """
    Trains an XGBoost classifier within a scikit-learn pipeline that includes
    column encoding using Ordinal and OneHot encoders. Uses the best hyperparameters
    obtained from cross-validation and saves the trained pipeline to disk.

    The pipeline includes:
    - ColumnTransformer for encoding numeric, ordinal, binary, and categorical features
    - XGBoost classifier with tuned hyperparameters

    Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        save_path (str): Path where the trained model will be saved (default: "models/model.pkl")
        random_seed (int): Random seed for model reproducibility (default: 42)

    Returns:
        sklearn.pipeline.Pipeline: Trained pipeline object
    """
    # Use best parameters from RandomizedSearch
    xgb = XGBClassifier(
        booster='gbtree',
        random_state=random_seed,
        learning_rate=0.3,
        max_depth=7,
        subsample=0.75,
        reg_lambda=0.1,
        reg_alpha=0
    )

    model_pipeline = Pipeline([
        ('encoder', encoder),
        ('xgb', xgb)
    ])

    # Train model
    start_time = time.time()
    model_pipeline.fit(X_train, y_train)
    end_time = time.time()
    runtime = end_time - start_time

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model_pipeline, save_path)

    print(f"✅ Model trained and saved to {save_path}")
    print(f"⏱️ Training time: {runtime:.2f} seconds")

    return model_pipeline
