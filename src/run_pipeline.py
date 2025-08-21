import os
import sys
import mlflow
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config_loader import get_config

config = get_config()
mlflow_config = config.get_mlflow_config()

tracking_uri = mlflow_config.get('tracking_uri', 'http://localhost:5000')
mlflow.set_tracking_uri(tracking_uri)

def run_complete_pipeline():
    print("="*60)
    print("RUNNING COMPLETE ML PIPELINE")
    print("="*60)
    print(f"MLFlow Tracking URI: {tracking_uri}")

    # Set experiment before starting run
    if 'experiment_name' in mlflow_config:
        mlflow.set_experiment(mlflow_config['experiment_name'])
        print(f"MLFlow Experiment: {mlflow_config['experiment_name']}")

    # Start MLFlow run for the entire pipeline
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        print(f"MLFlow Run ID: {run_id}")

        try:
            # Step 1: Data Preprocessing with Drift Generation
            print("\n" + "="*40)
            print("1. DATA PREPROCESSING WITH DRIFT GENERATION")
            print("="*40)

            from src.data_preprocessing import preprocess_data
            preprocessing_result = preprocess_data()
            X_train, X_test, y_train, y_test, X_train_drifted, y_train_drifted, X_test_drifted, y_test_drifted = preprocessing_result

            # Log data shapes
            mlflow.log_param("original_train_shape", f"{X_train.shape[0]}x{X_train.shape[1]}")
            mlflow.log_param("original_test_shape", f"{X_test.shape[0]}x{X_test.shape[1]}")
            mlflow.log_param("drifted_train_shape", f"{X_train_drifted.shape[0]}x{X_train_drifted.shape[1]}")
            mlflow.log_param("drifted_test_shape", f"{X_test_drifted.shape[0]}x{X_test_drifted.shape[1]}")

            print("Data preprocessed successfully")
            print(f"Original - Train: {X_train.shape}, Test: {X_test.shape}")
            print(f"Drifted  - Train: {X_train_drifted.shape}, Test: {X_test_drifted.shape}")

            # Step 2: Feature Engineering
            print("\n" + "="*40)
            print("2. FEATURE ENGINEERING")
            print("="*40)

            from src.feature_engineering import engineer_features_from_splits

            # Apply feature engineering to original data
            X_train_eng, X_test_eng, y_train_eng, y_test_eng = engineer_features_from_splits(
                X_train, X_test, y_train, y_test
            )

            # Apply feature engineering to drifted data
            X_train_drift_eng, X_test_drift_eng, y_train_drift_eng, y_test_drift_eng = engineer_features_from_splits(
                X_train_drifted, X_test_drifted, y_train_drifted, y_test_drifted
            )

            # Log feature engineering results
            mlflow.log_param("engineered_train_shape", f"{X_train_eng.shape[0]}x{X_train_eng.shape[1]}")
            mlflow.log_param("engineered_test_shape", f"{X_test_eng.shape[0]}x{X_test_eng.shape[1]}")
            mlflow.log_param("feature_engineering_applied", True)

            print("Feature engineering completed")
            print(f"Original Engineered - Train: {X_train_eng.shape}, Test: {X_test_eng.shape}")
            print(f"Drifted Engineered  - Train: {X_train_drift_eng.shape}, Test: {X_test_drift_eng.shape}")

            # Step 3: Model Training
            print("\n" + "="*40)
            print("3. MODEL TRAINING WITH MLFLOW")
            print("="*40)

            from src.model_training import train_model

            model = train_model(X_train=X_train_eng, y_train=y_train_eng, use_existing_run=True)

            # Step 4: Model Evaluation
            print("\n" + "="*40)
            print("4. MODEL EVALUATION")
            print("="*40)

            from src.evaluation import evaluate_model_with_mlflow, check_performance_threshold

            # Evaluate on original test data
            print("Evaluating on original test data...")
            original_results = evaluate_model_with_mlflow(
                model, X_test_eng, y_test_eng, save_results=False, run_context="original"
            )

            # Evaluate on drifted test data
            print("Evaluating on drifted test data...")
            drifted_results = evaluate_model_with_mlflow(
                model, X_test_drift_eng, y_test_drift_eng, save_results=False, run_context="drifted"
            )

            # Calculate performance degradation
            accuracy_drop = original_results['accuracy'] - drifted_results['accuracy']
            f1_drop = original_results['f1_score'] - drifted_results['f1_score']

            mlflow.log_metric("accuracy_drop_due_to_drift", accuracy_drop)
            mlflow.log_metric("f1_drop_due_to_drift", f1_drop)

            print("Model evaluation completed")
            print(f"Original - Accuracy: {original_results['accuracy']:.4f}, F1: {original_results['f1_score']:.4f}")
            print(f"Drifted  - Accuracy: {drifted_results['accuracy']:.4f}, F1: {drifted_results['f1_score']:.4f}")
            print(f"Performance drop - Accuracy: {accuracy_drop:.4f}, F1: {f1_drop:.4f}")

            # Step 5: Performance Threshold Check and Model Registration
            print("\n" + "="*40)
            print("5. PERFORMANCE THRESHOLD CHECK & MODEL REGISTRATION")
            print("="*40)

            # Classification threshold accuracy > 0.8
            threshold_met, threshold_desc = check_performance_threshold(original_results, "classification")

            try:
                mlflow.set_tag("performance_threshold", threshold_desc)
                mlflow.set_tag("threshold_met", str(threshold_met))
            except Exception as log_error:
                print(f"Warning: Could not log threshold info to MLflow: {log_error}")

            if threshold_met:
                # Register model using exact format
                model_name = "obesity_risk_classifier"  # Meaningful name as required
                model_uri = f"runs:/{run_id}/model"

                try:
                    mlflow.register_model(model_uri, model_name)
                    print("Model registered successfully!")
                    print(f"Model name: {model_name}")
                    print(f"Model URI: {model_uri}")
                    print(f"Threshold: {threshold_desc} ✓")
                    try:
                        mlflow.set_tag("model_registered", "true")
                        mlflow.set_tag("model_name", model_name)
                    except Exception as log_error:
                        print(f"Warning: Could not log registration success: {log_error}")
                    model_registered = True
                except Exception as e:
                    print(f"Model registration failed: {e}")
                    try:
                        mlflow.set_tag("model_registered", "false")
                        mlflow.set_tag("registration_error", str(e)[:100])
                    except Exception as log_error:
                        print(f"Warning: Could not log registration failure: {log_error}")
                    model_registered = False
            else:
                print("Model does not meet performance threshold")
                print(f"Current accuracy: {original_results['accuracy']:.4f}")
                print(f"Required: {threshold_desc}")
                try:
                    mlflow.set_tag("model_registered", "false")
                    mlflow.set_tag("threshold_not_met_reason", f"accuracy_{original_results['accuracy']:.4f}_required_{threshold_desc}")
                except Exception as log_error:
                    print(f"Warning: Could not log threshold failure: {log_error}")
                model_registered = False

            # Step 6: Drift Detection
            print("\n" + "="*40)
            print("6. DRIFT DETECTION")
            print("="*40)

            from src.drift_detection import detect_drift

            # Run drift detection twice as specified
            print("Running drift detection on test set...")
            test_drift_results = detect_drift('data/splits/test.csv', 'data/splits/drifted_test.csv')

            # Log drift status to MLFlow
            try:
                mlflow.set_tag("test_drift_detected", str(test_drift_results["drift_detected"]))
                mlflow.log_metric("test_overall_drift_score", test_drift_results["overall_drift_score"])
            except Exception as log_error:
                print(f"Warning: Could not log drift results to MLflow: {log_error}")

            # Log individual feature drift scores
            try:
                for feature, score in test_drift_results["feature_drifts"].items():
                    # Use metrics for drift scores to avoid parameter conflicts
                    safe_feature_name = feature.replace(" ", "_").replace("-", "_")
                    mlflow.log_metric(f"drift_score_{safe_feature_name}", score)
            except Exception as log_error:
                print(f"Warning: Could not log feature drift scores: {log_error}")

            # Save comprehensive evaluation results to JSON
            comprehensive_results = {
                "original_data_evaluation": original_results,
                "drifted_data_evaluation": drifted_results,
                "performance_degradation": {
                    "accuracy_drop": accuracy_drop,
                    "f1_score_drop": f1_drop
                },
                "model_registration": {
                    "threshold_met": threshold_met,
                    "threshold_description": threshold_desc,
                    "model_registered": model_registered
                },
                "drift_detection": test_drift_results,
                "mlflow_info": {
                    "run_id": run_id,
                    "tracking_uri": tracking_uri
                }
            }

            # Save to reports/evaluation_results.json
            os.makedirs("reports", exist_ok=True)
            with open("reports/evaluation_results.json", 'w') as f:
                json.dump(comprehensive_results, f, indent=2)

            print("Comprehensive results saved to: reports/evaluation_results.json")

            # Raise error if drift detected
            if test_drift_results["drift_detected"]:
                error_msg = "Data drift detected in test set! Model retraining required."
                print(f"{error_msg}")

                # Log the error to MLFlow using tags instead of params to avoid conflicts
                try:
                    mlflow.set_tag("pipeline_final_status", "drift_detected_error")
                    mlflow.set_tag("pipeline_outcome", "drift_error_raised")
                    mlflow.set_tag("drift_error_message", error_msg)
                except Exception as log_error:
                    print(f"Warning: Could not log drift error to MLflow: {log_error}")

                raise ValueError(error_msg)
            else:
                print("\nNo significant drift detected - pipeline completed successfully")
                try:
                    mlflow.set_tag("pipeline_final_status", "completed_successfully")
                    mlflow.set_tag("pipeline_outcome", "success")
                except Exception as log_error:
                    print(f"Warning: Could not log success to MLflow: {log_error}")

            pipeline_results = {
                "status": "success",
                "preprocessing": {
                    "original_train_shape": X_train.shape,
                    "original_test_shape": X_test.shape,
                    "drifted_train_shape": X_train_drifted.shape,
                    "drifted_test_shape": X_test_drifted.shape,
                    "drift_generated": True
                },
                "feature_engineering": {
                    "engineered_train_shape": X_train_eng.shape,
                    "engineered_test_shape": X_test_eng.shape,
                    "features_engineered": True
                },
                "model_training": {
                    "model_type": "XGBoost Pipeline",
                    "mlflow_run_id": run_id,
                    "model_trained": True
                },
                "evaluation": original_results,
                "model_registration": {
                    "threshold_met": threshold_met,
                    "model_registered": model_registered
                },
                "drift_detection": test_drift_results,
                "performance_comparison": {
                    "original_accuracy": original_results['accuracy'],
                    "drifted_accuracy": drifted_results['accuracy'],
                    "accuracy_drop": accuracy_drop,
                    "f1_drop": f1_drop
                }
            }

            return pipeline_results

        except Exception as e:
            try:
                mlflow.log_param("pipeline_error", str(e)[:250])
                mlflow.set_tag("pipeline_outcome", "error")
                mlflow.set_tag("error_type", type(e).__name__)
            except Exception as log_error:
                print(f"Warning: Could not log error to MLflow: {log_error}")
            raise

def run_pipeline_with_error_handling():
    """
    Run pipeline with comprehensive error handling.
    Handles the expected drift detection error.
    """
    try:
        results = run_complete_pipeline()

        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("All steps completed without drift detection")
        print(f"Model performance: {results['evaluation']['accuracy']:.4f} accuracy")
        print(f"Model registered: {results['model_registration']['model_registered']}")
        print("="*60)

        return results

    except ValueError as e:
        if "Data drift detected" in str(e):
            print("\n" + "="*60)
            print("DRIFT DETECTION ERROR")
            print("="*60)
            print(f"{e}")
            print("="*60)

            return {
                "status": "drift_detected",
                "message": str(e),
                "expected_behavior": True
            }
        else:
            print(f"\nUnexpected ValueError: {e}")
            raise

    except ImportError as e:
        print(f"\nImport error - missing pipeline component: {e}")
        print("Make sure all required modules are available:")
        print("  - src/data_preprocessing.py")
        print("  - src/feature_engineering.py")
        print("  - src/model_training.py")
        print("  - src/evaluation.py")
        print("  - src/drift_detection.py")
        raise

    except Exception as e:
        print(f"\nUnexpected pipeline error: {e}")
        import traceback
        traceback.print_exc()
        print("\nDebugging tips:")
        print("1. Check if MLFlow server is running: mlflow server --host 127.0.0.1 --port 5000")
        print("2. Verify all configuration files are present")
        print("3. Check if data files exist in the data/ directory")
        raise

def main():
    """
    Main function to run the complete pipeline.
    """
    if mlflow.active_run():
        mlflow.end_run()

    try:
        mlflow.search_experiments()
        print(f"MLFlow connection successful: {tracking_uri}")
    except Exception as e:
        print(f"MLFlow connection issue: {e}")
        print("Make sure MLFlow server is running:")
        print("mlflow server --host 127.0.0.1 --port 5000")

    results = run_pipeline_with_error_handling()

    print("\nPIPELINE EXECUTION SUMMARY:")
    print("="*50)

    if results and "status" in results:
        if results["status"] == "drift_detected":
            print("Status: Drift detected")
            print(f"Message: {results['message']}")
        elif results["status"] == "success":
            print("Status: Success (no drift detected)")
            print(f"Training accuracy: {results['evaluation']['accuracy']:.4f}")
            print(f"Model registered: {results['model_registration']['model_registered']}")
            print(f"Drift detected: {results['drift_detection']['drift_detected']}")
        else:
            print(f"Status: {results['status']}")
    else:
        print("Status: Unknown (check console output above)")

    print("="*50)

    return results

if __name__ == "__main__":
    main()
