import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

# Add config directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config_loader import get_config

config = get_config()
mlflow_config = config.get_mlflow_config()
hyperparams = config.get_hyperparameters()
xgb_params = config.get('model_training.xgboost_params')
feature_config = config.get_model_features_config()
artifacts_config = config.get_artifacts_config()
training_config = config.get('model_training.training')

tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or mlflow_config['tracking_uri']
mlflow.set_tracking_uri(tracking_uri)

print("="*60)
print("XGBOOST MODEL TRAINING WITH MLFLOW (CONFIG-DRIVEN)")
print("="*60)
print(f"MLFlow URI: {tracking_uri}")
print(f"Hyperparameters: {hyperparams}")
print(f"Model save path: {artifacts_config['model_save_path']}")

# Feature groups from configuration
numerical_cols = feature_config['numerical_cols']
ordinal_cols = feature_config['ordinal_cols']
binary_cols = feature_config['binary_cols']
categorical_cols = feature_config['categorical_cols']
onehot_cols = binary_cols + categorical_cols

# Create encoders using configuration
def create_encoders():
    """Create encoders using configuration."""
    ordinal_categories = []
    for col in ordinal_cols:
        if col in feature_config['ordinal_categories']:
            ordinal_categories.append(feature_config['ordinal_categories'][col])

    ordinal = OrdinalEncoder(
        categories=ordinal_categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    onehot = OneHotEncoder(handle_unknown="ignore")

    return ordinal, onehot

ordinal, onehot = create_encoders()

encoder = ColumnTransformer(
    transformers=[
        ("numerical", "passthrough", numerical_cols),
        ("ordinal", ordinal, ordinal_cols),
        ("onehot", onehot, onehot_cols),
    ]
)

class CustomMLModel(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow PyFunc model wrapper for XGBoost pipeline.
    Encapsulates preprocessing and prediction logic.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None

    def load_context(self, context):
        """Load model artifacts from MLflow context."""
        self.model = joblib.load(context.artifacts["model"])

        # Load preprocessor if exists
        if "preprocessor" in context.artifacts:
            self.preprocessor = joblib.load(context.artifacts["preprocessor"])

        # Load feature names
        if "feature_names" in context.artifacts:
            with open(context.artifacts["feature_names"], 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        # The model already includes preprocessing in the pipeline
        predictions = self.model.predict(model_input)
        return predictions

def train_model(X_train=None, y_train=None, X_train_path=None, y_train_path=None, use_existing_run=False):
    """
    Trains an XGBoost classifier with MLFlow tracking using configuration.
    - Log exactly 3 hyperparameters: n_estimators, max_depth, learning_rate
    - Use custom PyFunc wrapper
    - Save artifacts to mlflow/artifacts/

    Parameters:
        X_train (pd.DataFrame, optional): Training features
        y_train (pd.Series, optional): Training labels
        X_train_path (str, optional): Path to X_train.pkl (fallback)
        y_train_path (str, optional): Path to y_train.pkl (fallback)
        use_existing_run (bool): Whether to use existing MLflow run or create new one

    Returns:
        sklearn.pipeline.Pipeline: Trained model pipeline
    """
    # Check if we should use existing run or create new one
    if use_existing_run and mlflow.active_run() is not None:
        # Use existing run - don't create a new one
        return _train_model_core(X_train, y_train, X_train_path, y_train_path)
    else:
        # Set experiment before starting run
        if 'experiment_name' in mlflow_config:
            mlflow.set_experiment(mlflow_config['experiment_name'])

        with mlflow.start_run():
            return _train_model_core(X_train, y_train, X_train_path, y_train_path)

def _train_model_core(X_train=None, y_train=None, X_train_path=None, y_train_path=None):
    """
    Core training logic that works within an existing MLflow run.
    """
    active_run = mlflow.active_run()
    if not active_run:
        raise RuntimeError("No active MLflow run! Cannot log parameters. Make sure you're inside mlflow.start_run() context.")

    print(f"Active MLflow run confirmed: {active_run.info.run_id}")
    print(f"Run status: {active_run.info.status}")
    print(f"Experiment ID: {active_run.info.experiment_id}")

    # Load training data
    if X_train is not None and y_train is not None:
        print("Using provided training data")
    elif X_train_path and y_train_path:
        print("Loading training data from files")
        X_train = joblib.load(X_train_path)
        y_train = joblib.load(y_train_path)
    else:
        raise ValueError("Either provide X_train/y_train or X_train_path/y_train_path")

    # Log exactly 3 hyperparameters for classification
    n_estimators = hyperparams['n_estimators']
    max_depth = hyperparams['max_depth']
    learning_rate = hyperparams['learning_rate']

    print(f"About to log hyperparameters to run {active_run.info.run_id}:")
    print(f"n_estimators: {n_estimators} (type: {type(n_estimators)})")
    print(f"max_depth: {max_depth} (type: {type(max_depth)})")
    print(f"learning_rate: {learning_rate} (type: {type(learning_rate)})")

    try:
        # Convert to basic Python types to avoid MLflow serialization issues
        n_estimators_val = int(n_estimators) if hasattr(n_estimators, '__int__') else n_estimators
        max_depth_val = int(max_depth) if hasattr(max_depth, '__int__') else max_depth
        learning_rate_val = float(learning_rate) if hasattr(learning_rate, '__float__') else learning_rate

        print("Logging parameters...")
        mlflow.log_param("n_estimators", n_estimators_val)
        print(f"Successfully logged n_estimators: {n_estimators_val}")

        mlflow.log_param("max_depth", max_depth_val)
        print(f"Successfully logged max_depth: {max_depth_val}")

        mlflow.log_param("learning_rate", learning_rate_val)
        print(f"Successfully logged learning_rate: {learning_rate_val}")

        # ✅ Immediate verification that parameters were saved
        print("Verifying parameters were saved to MLflow...")
        current_run = mlflow.get_run(active_run.info.run_id)
        logged_params = current_run.data.params
        print(f"All parameters currently in MLflow: {logged_params}")

        # Check specifically for our required parameters
        required_params = ["n_estimators", "max_depth", "learning_rate"]
        missing_params = [p for p in required_params if p not in logged_params]

        if missing_params:
            print(f"WARNING: Missing parameters after logging: {missing_params}")
            print("This indicates an MLflow logging issue!")
        else:
            print(f"SUCCESS: All {len(required_params)} parameters confirmed in MLflow!")

    except Exception as param_error:
        print(f"CRITICAL ERROR logging parameters: {param_error}")
        print(f"MLflow run ID: {active_run.info.run_id}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        import traceback
        traceback.print_exc()
        # Continue with training even if parameter logging fails
        print("Continuing with model training despite parameter logging failure...")

    # Create XGBoost classifier
    print("Creating XGBoost classifier with parameters:")
    print(f"n_estimators: {n_estimators}")
    print(f"max_depth: {max_depth}")
    print(f"learning_rate: {learning_rate}")

    xgb = XGBClassifier(
        booster=xgb_params['booster'],
        random_state=xgb_params['random_state'],
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        subsample=xgb_params['subsample'],
        reg_lambda=xgb_params['reg_lambda'],
        reg_alpha=xgb_params['reg_alpha'],
    )

    # Create model pipeline
    model_pipeline = Pipeline([("encoder", encoder), ("xgb", xgb)])

    print("Model pipeline created successfully")
    print(f"Pipeline steps: {[step[0] for step in model_pipeline.steps]}")
    print(f"Training data shape: {X_train.shape}")

    print("Training XGBoost model with MLFlow tracking...")
    start = time.time()
    model_pipeline.fit(X_train, y_train)
    end = time.time()

    training_time = end - start
    print(f"Training completed in {training_time:.2f} seconds")

    # Save model artifacts to mlflow/artifacts/
    print("Saving model artifacts...")
    artifacts_dir = artifacts_config['mlflow_artifacts_dir']
    os.makedirs(artifacts_dir, exist_ok=True)
    print(f"Artifacts directory: {artifacts_dir}")

    model_artifact_path = os.path.join(artifacts_dir, artifacts_config['model_file'])
    joblib.dump(model_pipeline, model_artifact_path)
    print(f"Model saved to: {model_artifact_path}")

    feature_names_path = os.path.join(artifacts_dir, artifacts_config['feature_names_file'])
    with open(feature_names_path, 'w') as f:
        if hasattr(X_train, 'columns'):
            for feature in X_train.columns:
                f.write(f"{feature}\n")
        else:
            for i in range(X_train.shape[1]):
                f.write(f"feature_{i}\n")
    print(f"Feature names saved to: {feature_names_path}")

    artifacts = {
        "model": model_artifact_path,
        "feature_names": feature_names_path
    }

    try:
        print("Logging model to MLflow with custom PyFunc wrapper...")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CustomMLModel(),
            artifacts=artifacts,
            conda_env=None
        )
        print("Model successfully logged to MLflow with PyFunc wrapper")
    except Exception as model_log_error:
        print(f"Error logging model to MLflow: {model_log_error}")
        import traceback
        traceback.print_exc()
        print("Continuing despite model logging error...")

    # Save model to configured path
    if training_config.get('save_model', True):
        model_save_path = artifacts_config['model_save_path']
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(model_pipeline, model_save_path)
        print(f"Model saved to configured path: {model_save_path}")

    # Log training time as metric
    if training_config.get('log_training_time', True):
        try:
            mlflow.log_metric("training_time", training_time)
            print(f"Training time logged as metric: {training_time:.2f} seconds")
        except Exception as metric_error:
            print(f"Error logging training time metric: {metric_error}")

    # Final verification
    if mlflow.active_run():
        final_run_id = mlflow.active_run().info.run_id
        print("Model training completed successfully!")
        print(f"MLflow run ID: {final_run_id}")

        # Final parameter check
        try:
            final_run = mlflow.get_run(final_run_id)
            final_params = final_run.data.params
            final_metrics = final_run.data.metrics
            print("Final MLflow state:")
            print(f"Parameters ({len(final_params)}): {final_params}")
            print(f"Metrics ({len(final_metrics)}): {list(final_metrics.keys())}")
        except Exception as final_check_error:
            print(f"Could not perform final MLflow check: {final_check_error}")
    else:
        print("Warning: No active MLflow run at completion")

    return model_pipeline

def train_with_feature_engineering(use_existing_run=False):
    """
    Train model using the feature engineering pipeline.
    This should work with the drifted data pipeline.

    Args:
        use_existing_run (bool): Whether to use existing MLflow run or create new one
    """
    from src.feature_engineering import main as run_feature_engineering

    print("Running feature engineering pipeline...")
    engineered_data = run_feature_engineering()

    # Use original (non-drifted) data for training
    X_train_eng, X_test_eng, y_train_eng, y_test_eng = engineered_data['original']
    print("Feature engineering completed:")
    print(f"Original - Train: {X_train_eng.shape}, Test: {X_test_eng.shape}")

    # Also get drifted data for drift detection later
    X_train_drift_eng, X_test_drift_eng, y_train_drift_eng, y_test_drift_eng = engineered_data['drifted']
    print(f"Drifted  - Train: {X_train_drift_eng.shape}, Test: {X_test_drift_eng.shape}")

    print(f"Training with engineered features: {X_train_eng.shape}")

    model = train_model(X_train=X_train_eng, y_train=y_train_eng, use_existing_run=use_existing_run)

    return model, engineered_data

def main():
    try:
        model, data = train_with_feature_engineering()

        # Extract the original data for return
        X_train_eng, X_test_eng, y_train_eng, y_test_eng = data['original']

        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("✓ Model training successful!")
        print(f"✓ Training data shape: {X_train_eng.shape}")
        print(f"✓ Model type: {type(model)}")
        print(f"✓ MLflow tracking URI: {tracking_uri}")
        print("="*60)

        return model, data

    except Exception as e:
        print(f"\n  TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
