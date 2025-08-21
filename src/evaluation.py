import os
import sys
import json
import joblib
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# Add config directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config_loader import get_config

config = get_config()
mlflow_config = config.get_mlflow_config()

tracking_uri = mlflow_config.get('tracking_uri')
mlflow.set_tracking_uri(tracking_uri)

experiment_name = mlflow_config.get('experiment_name')
if experiment_name:
    try:
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment set to: {experiment_name}")
    except Exception as e:
        print(f"Could not set MLflow experiment: {e}")

def evaluate_model_with_mlflow(model, X_test, y_test, save_results=True, run_context=None):
    """
    Evaluate model and log exactly 2 metrics to MLFlow.
    - Classification: accuracy, f1_score

    Parameters:
        model: Trained model (sklearn pipeline or estimator)
        X_test (pd.DataFrame or np.array): Test features
        y_test (pd.Series or np.array): Test labels
        save_results (bool): Whether to save results to JSON file
        run_context (str): Additional context for logging

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print(f"Evaluating model{' (' + run_context + ')' if run_context else ''}...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Log exactly 2 evaluation metrics for classification
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics to MLFlow with context if provided
    metric_suffix = f"_{run_context}" if run_context else ""
    mlflow.log_metric(f"accuracy{metric_suffix}", accuracy)
    mlflow.log_metric(f"f1_score{metric_suffix}", f1)

    # Create results dictionary
    results = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "n_samples": len(y_test),
        "n_features": X_test.shape[1] if hasattr(X_test, 'shape') else len(X_test[0])
    }

    # Save evaluation results to reports/evaluation_results.json
    if save_results:
        os.makedirs("reports", exist_ok=True)
        results_path = "reports/evaluation_results.json"

        if run_context:
            context_path = f"reports/evaluation_results_{run_context}.json"
            with open(context_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"{run_context.title()} evaluation results saved to: {context_path}")
        else:
            # Save main results
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Evaluation results saved to: {results_path}")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return results

def check_performance_threshold(results, threshold_type="classification"):
    """
    Check if model meets performance threshold for registration.
    - Classification: accuracy > 0.8
    - Regression: MSE < 0.1
    - Clustering: silhouette_score > 0.5

    Parameters:
        results (dict): Evaluation results
        threshold_type (str): Type of model ("classification", "regression", "clustering")

    Returns:
        bool: Whether threshold is met
        str: Threshold description
    """
    if threshold_type == "classification":
        threshold_met = results.get("accuracy", 0) > 0.8
        threshold_desc = "accuracy > 0.8"
    elif threshold_type == "regression":
        threshold_met = results.get("mse", float('inf')) < 0.1
        threshold_desc = "MSE < 0.1"
    elif threshold_type == "clustering":
        threshold_met = results.get("silhouette_score", 0) > 0.5
        threshold_desc = "silhouette_score > 0.5"
    else:
        threshold_met = False
        threshold_desc = "unknown threshold"

    return threshold_met, threshold_desc

def register_model_if_threshold_met(results, model_name="obesity_risk_classifier"):
    """
    Register model if performance threshold is met.

    Parameters:
        results (dict): Evaluation results
        model_name (str): Name for model registration

    Returns:
        bool: Whether model was registered
    """
    threshold_met, threshold_desc = check_performance_threshold(results, "classification")

    if threshold_met:
        try:
            # Register model using specified format
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, model_name)

            print(f"Model registered as '{model_name}' (threshold met: {threshold_desc})")
            print(f"   Accuracy: {results['accuracy']:.4f}")
            return True
        except Exception as e:
            print(f"Model registration failed: {e}")
            return False
    else:
        print(f"Model not registered (threshold not met: {threshold_desc})")
        print(f"Current accuracy: {results.get('accuracy', 0):.4f}")
        return False

def evaluate_model(
    model_path=None,
    X_test_path=None,
    y_test_path=None,
    model=None,
    X_test=None,
    y_test=None,
    report_path="reports/metrics.txt",
    plot_path="reports/confusion_matrix.png",
    context=None
):
    """
    Enhanced evaluation function that works with both file paths and direct data.
    Integrates with MLFlow tracking.

    Parameters:
        model_path (str, optional): Path to trained model (.pkl)
        X_test_path (str, optional): Path to X_test.pkl
        y_test_path (str, optional): Path to y_test.pkl
        model (optional): Trained model object
        X_test (optional): Test features
        y_test (optional): Test labels
        report_path (str): Path to save classification report
        plot_path (str): Path to save confusion matrix plot
        context (str): Context for the evaluation (e.g., "original", "drifted")

    Returns:
        dict: Evaluation results
    """

    # Load data/model from files or use provided objects
    if model is not None and X_test is not None and y_test is not None:
        print(f"Using provided model and test data{' (' + context + ')' if context else ''}")
    elif model_path and X_test_path and y_test_path:
        print(f"Loading model and test data from files{' (' + context + ')' if context else ''}")
        model = joblib.load(model_path)
        X_test = joblib.load(X_test_path)
        y_test = joblib.load(y_test_path)
    else:
        raise ValueError("Either provide model/X_test/y_test or their file paths")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)

    # Class labels for confusion matrix
    class_labels_num = [0, 1, 2, 3]
    class_labels = ["Underweight", "Normal_weight", "Overweight", "Obesity"]
    cm = confusion_matrix(y_test, y_pred, labels=class_labels_num)

    # Adjust file paths if context is provided
    if context:
        base_report = os.path.splitext(report_path)[0]
        base_plot = os.path.splitext(plot_path)[0]
        report_path = f"{base_report}_{context}.txt"
        plot_path = f"{base_plot}_{context}.png"

    # Save detailed metrics report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-Score (Weighted): {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"Detailed metrics saved to: {report_path}")

    # Create and save confusion matrix plot
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap="Blues", xticks_rotation="vertical", values_format="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix{' (' + context.title() + ')' if context else ''}")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Confusion matrix saved to: {plot_path}")

    # Return evaluation results
    results = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "n_samples": len(y_test),
        "n_features": X_test.shape[1] if hasattr(X_test, 'shape') else len(X_test[0])
    }

    return results

def evaluate_with_feature_engineering():
    """
    Evaluate model using the complete pipeline.
    Includes model registration based on performance threshold.
    """
    try:
        # Import pipeline components
        from src.feature_engineering import main as run_feature_engineering
        from src.model_training import train_model

        print("="*60)
        print("COMPLETE PIPELINE EVALUATION WITH MLFLOW")
        print("="*60)

        # Clean up any existing MLflow runs first
        if mlflow.active_run():
            mlflow.end_run()
            print("Cleaned up existing MLflow run")

        # Set experiment before starting run
        if 'experiment_name' in mlflow_config:
            mlflow.set_experiment(mlflow_config['experiment_name'])
            print(f"MLflow experiment set: {mlflow_config['experiment_name']}")

        # Run feature engineering to get data
        print("Running feature engineering pipeline...")
        engineered_data = run_feature_engineering()

        if not engineered_data or 'original' not in engineered_data:
            print("Feature engineering did not return expected data structure")
            return None

        # Extract data
        X_train_eng, X_test_eng, y_train_eng, y_test_eng = engineered_data['original']
        X_train_drift, X_test_drift, y_train_drift, y_test_drift = engineered_data['drifted']

        print("Data extracted successfully")
        print(f"Original test data: {X_test_eng.shape}")
        print(f"Drifted test data: {X_test_drift.shape}")

        # Train and evaluate within the same MLFlow run
        with mlflow.start_run():
            run_id = mlflow.active_run().info.run_id
            print(f"MLflow run started: {run_id}")

            # Train model using existing run context
            print("Training model...")
            model = train_model(X_train=X_train_eng, y_train=y_train_eng, use_existing_run=True)

            if model is None:
                print("Model training returned None")
                return None

            print("Model training completed")

            # Evaluate on original test data
            print("Evaluating on original test data...")
            original_results = evaluate_model_with_mlflow(
                model, X_test_eng, y_test_eng, save_results=False, run_context="original"
            )

            if not original_results:
                print("Original evaluation failed")
                return None

            print(f"Original evaluation: accuracy={original_results['accuracy']:.4f}")

            # Evaluate on drifted test data
            print("Evaluating on drifted test data...")
            drifted_results = evaluate_model_with_mlflow(
                model, X_test_drift, y_test_drift, save_results=False, run_context="drifted"
            )

            if not drifted_results:
                print("Drifted evaluation failed")
                return None

            print(f"Drifted evaluation: accuracy={drifted_results['accuracy']:.4f}")

            # Check performance threshold and register model if met
            print("Checking performance threshold...")
            model_registered = register_model_if_threshold_met(
                original_results,
                model_name="obesity_risk_classifier"
            )

            print(f"Model registration status: {model_registered}")

            # Log model registration status
            try:
                mlflow.log_param("model_registered", model_registered)
            except Exception as e:
                print(f"Could not log model_registered param: {e}")

            # Calculate performance degradation due to drift
            performance_degradation = {
                "accuracy_drop": original_results["accuracy"] - drifted_results["accuracy"],
                "f1_score_drop": original_results["f1_score"] - drifted_results["f1_score"]
            }

            print("Performance degradation calculated")
            print(f"Accuracy drop: {performance_degradation['accuracy_drop']:.4f}")
            print(f"F1-score drop: {performance_degradation['f1_score_drop']:.4f}")

            # Log drift impact metrics
            try:
                mlflow.log_metric("accuracy_drop_due_to_drift", performance_degradation["accuracy_drop"])
                mlflow.log_metric("f1_score_drop_due_to_drift", performance_degradation["f1_score_drop"])
                print("Drift metrics logged to MLflow")
            except Exception as e:
                print(f"Could not log drift metrics: {e}")

            # Create comprehensive results
            comprehensive_results = {
                "original_data": original_results,
                "drifted_data": drifted_results,
                "performance_degradation": performance_degradation,
                "model_registered": model_registered,
                "mlflow_run_id": run_id
            }

            print("Comprehensive results created")

            # Save to reports/evaluation_results.json
            try:
                os.makedirs("reports", exist_ok=True)
                with open("reports/evaluation_results.json", 'w') as f:
                    json.dump(comprehensive_results, f, indent=2)

                print("Comprehensive evaluation results saved to: reports/evaluation_results.json")
            except Exception as e:
                print(f"Could not save results to file: {e}")

            # Also create detailed reports and plots
            try:
                print("Creating detailed reports and plots...")
                evaluate_model(
                    model=model,
                    X_test=X_test_eng,
                    y_test=y_test_eng,
                    report_path="reports/metrics.txt",
                    plot_path="reports/confusion_matrix.png",
                    context="original"
                )

                evaluate_model(
                    model=model,
                    X_test=X_test_drift,
                    y_test=y_test_drift,
                    report_path="reports/metrics.txt",
                    plot_path="reports/confusion_matrix.png",
                    context="drifted"
                )

                print("Detailed reports and plots created")
            except Exception as e:
                print(f"Could not create detailed reports: {e}")

            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            print(f"Original Accuracy: {original_results['accuracy']:.4f}")
            print(f"Original F1-Score: {original_results['f1_score']:.4f}")
            print(f"Drifted Accuracy: {drifted_results['accuracy']:.4f}")
            print(f"Drifted F1-Score: {drifted_results['f1_score']:.4f}")
            print(f"Accuracy Drop: {performance_degradation['accuracy_drop']:.4f}")
            print(f"F1-Score Drop: {performance_degradation['f1_score_drop']:.4f}")
            print(f"Model Registered: {'Yes' if model_registered else 'No'}")
            print(f"MLFlow Run ID: {run_id}")
            print("="*60)

            print("Evaluation completed successfully, returning results")
            return comprehensive_results

    except ImportError as e:
        print(f"Pipeline components not available: {e}")
        print("Make sure all pipeline components are working:")
        print("  - src/feature_engineering.py")
        print("  - src/model_training.py")
        return None
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_from_files(
    model_path="models/model.pkl",
    X_test_path="data/processed/X_test.pkl",
    y_test_path="data/processed/y_test.pkl"
):
    """
    Evaluate model from saved files with MLFlow integration.
    Fallback method for file-based evaluation.
    """
    print("Attempting file-based evaluation...")

    if not all(os.path.exists(p) for p in [model_path, X_test_path, y_test_path]):
        print("   Required files not found:")
        print(f"  Model: {model_path} ({'✓' if os.path.exists(model_path) else '❌'})")
        print(f"  X_test: {X_test_path} ({'✓' if os.path.exists(X_test_path) else '❌'})")
        print(f"  y_test: {y_test_path} ({'✓' if os.path.exists(y_test_path) else '❌'})")
        return None

    with mlflow.start_run():
        # Load model and data
        model = joblib.load(model_path)
        X_test = joblib.load(X_test_path)
        y_test = joblib.load(y_test_path)

        # Evaluate with detailed metrics
        evaluate_model(model=model, X_test=X_test, y_test=y_test)

        # Evaluate with MLFlow logging
        mlflow_results = evaluate_model_with_mlflow(model, X_test, y_test)

        # Check threshold and register if met
        model_registered = register_model_if_threshold_met(mlflow_results)

        # Log registration status
        mlflow.log_param("model_registered", model_registered)

        print("  File-based evaluation completed!")
        print(f"  Accuracy: {mlflow_results['accuracy']:.4f}")
        print(f"  F1-Score: {mlflow_results['f1_score']:.4f}")
        print(f"  Model Registered: {'Yes' if model_registered else 'No'}")

        return mlflow_results

def main():
    """
    Main function to test evaluation with MLFlow integration.
    """
    print("="*60)
    print("MODEL EVALUATION WITH MLFLOW INTEGRATION")
    print("="*60)
    print(f"MLFlow Tracking URI: {tracking_uri}")

    # Method 1: Try complete pipeline evaluation
    print("\n Attempting complete pipeline evaluation...")
    results = evaluate_with_feature_engineering()

    if results is not None:
        print("\n Complete pipeline evaluation successful!")
        return results

    # Method 2: Fallback to file-based evaluation
    print("\n Attempting file-based evaluation...")
    results = evaluate_from_files()

    if results is not None:
        print("\n File-based evaluation successful!")
        return results

    # Method 3: Guidance for setup
    print("\n No evaluation data available. Please run the pipeline:")
    print("   1. Start MLFlow server: mlflow server --host 127.0.0.1 --port 5000")
    print("   2. python src/data_preprocessing.py")
    print("   3. python src/feature_engineering.py")
    print("   4. python src/model_training.py")
    print("   5. python src/evaluation.py")

    return None

if __name__ == "__main__":
    main()
