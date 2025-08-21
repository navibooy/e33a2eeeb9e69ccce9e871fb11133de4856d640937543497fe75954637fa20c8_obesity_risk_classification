"""
HW3 ML Pipeline DAG with Drift Detection and Branching Logic
Using ConfigLoader for MLflow configuration
"""

import os
import sys
import json
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.models import Variable
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path for imports
sys.path.append('/opt/airflow/dags/repo')
sys.path.append('/opt/airflow/dags')
sys.path.append('.')

try:
    from config.config_loader import get_config as get_config_loader
    config_loader = get_config_loader(environment="production")
    CONFIG_LOADER_AVAILABLE = True
    logger.info("✓ ConfigLoader imported successfully")
except ImportError as e:
    logger.warning(f"ConfigLoader not available: {e}")
    CONFIG_LOADER_AVAILABLE = False

def get_airflow_config():
    """Get configuration from Airflow Variables with sensible defaults"""
    return {
        'model_accuracy_threshold': float(Variable.get("model_accuracy_threshold", default_var=0.8)),
        'drift_threshold': float(Variable.get("drift_threshold", default_var=0.5)),
        'mlflow_experiment_name': Variable.get("mlflow_experiment_name", default_var="obesity_risk_classification"),
        'max_retries': int(Variable.get("max_retries", default_var=2)),
        'retry_delay_minutes': int(Variable.get("retry_delay_minutes", default_var=5)),
    }

def setup_mlflow_from_config():
    """
    Setup MLflow using ConfigLoader instead of hardcoded values
    """
    if not CONFIG_LOADER_AVAILABLE:
        logger.warning("ConfigLoader not available, using fallback MLflow setup")
        mlflow_uri = "http://mlflow:5000"
        experiment_name = "obesity_risk_classification"
    else:
        try:
            mlflow_config = config_loader.get_mlflow_config()
            mlflow_uri = mlflow_config['tracking_uri']
            experiment_name = mlflow_config['experiment_name']

            logger.info("✓ MLflow config from ConfigLoader:")
            logger.info(f"  URI: {mlflow_uri}")
            logger.info(f"  Experiment: {experiment_name}")

        except Exception as e:
            logger.error(f"Failed to get MLflow config from ConfigLoader: {e}")
            # Fallback
            mlflow_uri = "http://localhost:5000"
            experiment_name = "obesity_risk_classification"

    try:
        mlflow.set_tracking_uri(mlflow_uri)
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri
        os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name

        logger.info(f"✓ MLflow URI set to: {mlflow_uri}")
        logger.info(f"✓ Environment variable MLFLOW_TRACKING_URI set to: {os.environ.get('MLFLOW_TRACKING_URI')}")

        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"✓ MLflow experiment set to: {experiment_name}")
            return True
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}")
            return False

    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        return False

airflow_config = get_airflow_config()

mlflow_setup_success = setup_mlflow_from_config()

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 8, 2),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': airflow_config['max_retries'],
    'retry_delay': timedelta(minutes=airflow_config['retry_delay_minutes']),
    'max_active_runs': 1
}

dag = DAG(
    'ml_pipeline_dag',
    default_args=default_args,
    description='ML Pipeline with Drift Detection',
    schedule=timedelta(days=1),
    catchup=False,
    tags=['ml', 'hw3', 'drift-detection', 'improved'],
    max_active_runs=1
)

def cleanup_mlflow_runs():
    """Helper function to clean up any hanging MLflow runs"""
    try:
        if mlflow.active_run():
            run_id = mlflow.active_run().info.run_id
            logger.warning(f"Cleaning up existing MLflow run: {run_id}")
            mlflow.end_run()
            return run_id
    except Exception as e:
        logger.warning(f"Warning: Error cleaning up MLflow runs: {e}")
    return None

def get_mlflow_config_in_task():
    """
    Helper function to get MLflow config within tasks
    """
    if CONFIG_LOADER_AVAILABLE:
        try:
            return config_loader.get_mlflow_config()
        except Exception as e:
            logger.warning(f"Failed to get MLflow config in task: {e}")

    # Fallback configuration
    return {
        'tracking_uri': os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
        'experiment_name': os.environ.get('MLFLOW_EXPERIMENT_NAME', 'obesity_risk_classification'),
        'use_file_tracking': False
    }

# Data persistence and XCom usage
def save_task_results(task_id: str, results: dict, **context):
    """Save task results both to file and XCom for better data flow"""
    os.makedirs("airflow_results", exist_ok=True)
    with open(f"airflow_results/{task_id}_results.json", 'w') as f:
        json.dump(results, f, default=str, indent=2)
    context['task_instance'].xcom_push(key='results', value=results)
    return results

def get_task_results(task_id: str, **context):
    """Get task results from XCom or file fallback"""
    try:
        results = context['task_instance'].xcom_pull(task_ids=task_id, key='results')
        if results:
            return results
    except Exception as e:
        logger.warning(f"Could not get XCom results for {task_id}: {e}")

    try:
        with open(f"airflow_results/{task_id}_results.json", 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Could not load results for {task_id}: {e}")
        return {}

def preprocess_data_task(**context):
    """
    Data preprocessing with drift generation
    """
    logger.info("="*50)
    logger.info("AIRFLOW TASK: DATA PREPROCESSING")
    logger.info("="*50)

    try:
        cleanup_mlflow_runs()
        from src.data_preprocessing import preprocess_data

        start_time = datetime.now()
        result = preprocess_data()
        X_train, X_test, y_train, y_test, X_train_drifted, y_train_drifted, X_test_drifted, y_test_drifted = result
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        logger.info(f"Data preprocessing completed in {processing_time:.2f} seconds")
        logger.info(f"Original - Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Drifted  - Train: {X_train_drifted.shape}, Test: {X_test_drifted.shape}")

        results = {
            "original_train_shape": X_train.shape,
            "original_test_shape": X_test.shape,
            "drifted_train_shape": X_train_drifted.shape,
            "drifted_test_shape": X_test_drifted.shape,
            "preprocessing_successful": True,
            "processing_time_seconds": processing_time,
            "timestamp": datetime.now().isoformat(),
            "data_quality_checks": {
                "original_null_count": X_train.isnull().sum().sum() if hasattr(X_train, 'isnull') else 0,
                "drifted_null_count": X_train_drifted.isnull().sum().sum() if hasattr(X_train_drifted, 'isnull') else 0,
            }
        }

        return save_task_results("preprocess_data", results, **context)

    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        cleanup_mlflow_runs()
        raise

def feature_engineering_task(**context):
    logger.info("="*50)
    logger.info("AIRFLOW TASK: FEATURE ENGINEERING")
    logger.info("="*50)

    try:
        cleanup_mlflow_runs()

        preprocessing_results = get_task_results("preprocess_data", **context)
        if preprocessing_results:
            logger.info(f"✓ Retrieved preprocessing results: {preprocessing_results.get('timestamp', 'N/A')}")

        from src.feature_engineering import main as run_feature_engineering

        start_time = datetime.now()
        engineered_data = run_feature_engineering()
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        X_train_eng, X_test_eng, y_train_eng, y_test_eng = engineered_data['original']
        X_train_drift_eng, X_test_drift_eng, y_train_drift_eng, y_test_drift_eng = engineered_data['drifted']

        logger.info(f"✓ Feature engineering completed in {processing_time:.2f} seconds")
        logger.info(f"  Original Engineered - Train: {X_train_eng.shape}, Test: {X_test_eng.shape}")
        logger.info(f"  Drifted Engineered  - Train: {X_train_drift_eng.shape}, Test: {X_test_drift_eng.shape}")

        results = {
            "original_engineered_train_shape": X_train_eng.shape,
            "original_engineered_test_shape": X_test_eng.shape,
            "drifted_engineered_train_shape": X_train_drift_eng.shape,
            "drifted_engineered_test_shape": X_test_drift_eng.shape,
            "feature_engineering_successful": True,
            "processing_time_seconds": processing_time,
            "timestamp": datetime.now().isoformat(),
            "feature_count": X_train_eng.shape[1],
        }

        return save_task_results("feature_engineering", results, **context)

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        cleanup_mlflow_runs()
        raise

def train_model_task(**context):
    """
    Model training using ConfigLoader for MLflow setup
    """
    logger.info("="*50)
    logger.info("AIRFLOW TASK: MODEL TRAINING")
    logger.info("="*50)

    try:
        mlflow_config = get_mlflow_config_in_task()
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])

        logger.info(f"Using MLflow URI from config: {mlflow_config['tracking_uri']}")

        cleanup_mlflow_runs()

        feature_results = get_task_results("feature_engineering", **context)
        if feature_results:
            logger.info(f"Retrieved feature engineering results: {feature_results.get('timestamp', 'N/A')}")

        from src.model_training import main as run_model_training

        logger.info("Running complete model training pipeline...")
        start_time = datetime.now()

        model, data = run_model_training()

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        cleanup_mlflow_runs()

        X_train_eng, X_test_eng, y_train_eng, y_test_eng = data['original']
        X_train_drift_eng, X_test_drift_eng, y_train_drift_eng, y_test_drift_eng = data['drifted']

        logger.info(f"Model training completed successfully in {training_time:.2f} seconds")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Original - Train: {X_train_eng.shape}, Test: {X_test_eng.shape}")
        logger.info(f"Drifted  - Train: {X_train_drift_eng.shape}, Test: {X_test_drift_eng.shape}")

        results = {
            "model_trained": True,
            "model_type": "XGBoost Pipeline",
            "training_data_shape": X_train_eng.shape,
            "engineered_data_available": True,
            "training_time_seconds": training_time,
            "timestamp": datetime.now().isoformat(),
            "mlflow_config_used": mlflow_config,
            "original_shapes": {
                "train": X_train_eng.shape,
                "test": X_test_eng.shape
            },
            "drifted_shapes": {
                "train": X_train_drift_eng.shape,
                "test": X_test_drift_eng.shape
            }
        }

        return save_task_results("train_model", results, **context)

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        import traceback
        traceback.print_exc()
        cleanup_mlflow_runs()
        raise

def evaluate_model_task(**context):
    """
    Model evaluation with ConfigLoader MLflow setup
    """
    logger.info("="*50)
    logger.info("AIRFLOW TASK: MODEL EVALUATION")
    logger.info("="*50)

    try:
        mlflow_config = get_mlflow_config_in_task()
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])

        cleanup_mlflow_runs()

        training_results = get_task_results("train_model", **context)
        if training_results:
            logger.info(f"Retrieved training results: {training_results.get('timestamp', 'N/A')}")

        # Check if model exists before retraining
        model_exists = os.path.exists("models/model.pkl")
        if not model_exists:
            logger.info("Model file not found, running training to get model and data...")
            from src.model_training import main as run_model_training
            model, data = run_model_training()
        else:
            logger.info("Model file found, loading existing model...")
            try:
                import pickle
                with open("models/model.pkl", 'rb') as f:
                    model = pickle.load(f)
                from src.model_training import main as run_model_training
                _, data = run_model_training()
                logger.info("Loaded existing model successfully")
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}, retraining...")
                from src.model_training import main as run_model_training
                model, data = run_model_training()

        from src.evaluation import evaluate_model_with_mlflow, register_model_if_threshold_met

        X_train_eng, X_test_eng, y_train_eng, y_test_eng = data['original']
        X_train_drift_eng, X_test_drift_eng, y_train_drift_eng, y_test_drift_eng = data['drifted']

        logger.info("Model and data loaded")
        logger.info(f"Original test data: {X_test_eng.shape}")
        logger.info(f"Drifted test data: {X_test_drift_eng.shape}")

        start_time = datetime.now()

        # Evaluate within an MLflow run
        with mlflow.start_run():
            mlflow.set_tag("pipeline_stage", "evaluation")

            logger.info("Evaluating on original test data...")
            original_results = evaluate_model_with_mlflow(
                model, X_test_eng, y_test_eng, save_results=False, run_context="original"
            )

            logger.info("Evaluating on drifted test data...")
            drifted_results = evaluate_model_with_mlflow(
                model, X_test_drift_eng, y_test_drift_eng, save_results=False, run_context="drifted"
            )

            logger.info(f"Checking performance threshold (configured: {airflow_config['model_accuracy_threshold']})...")  # ✅ FIXED
            model_registered = register_model_if_threshold_met(original_results)

            performance_degradation = {
                "accuracy_drop": original_results["accuracy"] - drifted_results["accuracy"],
                "f1_score_drop": original_results["f1_score"] - drifted_results["f1_score"]
            }

            mlflow.log_metric("accuracy_drop_due_to_drift", performance_degradation["accuracy_drop"])
            mlflow.log_metric("f1_score_drop_due_to_drift", performance_degradation["f1_score_drop"])

            run_id = mlflow.active_run().info.run_id

        end_time = datetime.now()
        evaluation_time = (end_time - start_time).total_seconds()

        results = {
            "original_data": original_results,
            "drifted_data": drifted_results,
            "performance_degradation": performance_degradation,
            "model_registered": model_registered,
            "mlflow_run_id": run_id,
            "evaluation_method": "optimized_evaluation",
            "evaluation_time_seconds": evaluation_time,
            "timestamp": datetime.now().isoformat(),
            "threshold_met": original_results["accuracy"] >= airflow_config['model_accuracy_threshold'],  # ✅ FIXED
            "mlflow_config_used": mlflow_config,
        }

        with open("airflow_results/evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        os.makedirs("reports", exist_ok=True)
        with open("reports/evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation completed successfully in {evaluation_time:.2f} seconds")
        logger.info(f"Original Accuracy: {original_results['accuracy']:.4f}")
        logger.info(f"Drifted Accuracy: {drifted_results['accuracy']:.4f}")
        logger.info(f"Model Registered: {model_registered}")
        logger.info(f"MLFlow Run ID: {run_id}")

        return save_task_results("evaluate_model", results, **context)

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        cleanup_mlflow_runs()
        raise

def drift_detection_task(**context):
    """
    Drift detection
    """
    logger.info("="*50)
    logger.info("AIRFLOW TASK: DRIFT DETECTION")
    logger.info("="*50)

    try:
        cleanup_mlflow_runs()
        from src.drift_detection import detect_drift

        start_time = datetime.now()
        logger.info("Running drift detection on test set...")
        drift_results = detect_drift('data/splits/test.csv', 'data/splits/drifted_test.csv')
        end_time = datetime.now()
        detection_time = (end_time - start_time).total_seconds()

        drift_detected = drift_results['drift_detected']
        overall_score = drift_results['overall_drift_score']
        threshold_exceeded = overall_score > airflow_config['drift_threshold']  # ✅ FIXED

        logger.info(f"Drift detection completed in {detection_time:.2f} seconds")
        logger.info(f"Drift detected: {drift_detected}")
        logger.info(f"Overall drift score: {overall_score:.4f} (threshold: {airflow_config['drift_threshold']})")  # ✅ FIXED
        logger.info(f"Threshold exceeded: {threshold_exceeded}")
        logger.info(f"Feature drifts: {drift_results.get('feature_drifts', {})}")

        enhanced_drift_results = {
            **drift_results,
            "detection_time_seconds": detection_time,
            "timestamp": datetime.now().isoformat(),
            "configured_threshold": airflow_config['drift_threshold'],  # ✅ FIXED
            "threshold_exceeded": threshold_exceeded,
        }

        return save_task_results("drift_detection", enhanced_drift_results, **context)

    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        cleanup_mlflow_runs()
        raise

def branch_on_drift(**context):
    """
    HW3 Requirement: BranchPythonOperator with task_id="branch_on_drift"
    """
    logger.info("="*50)
    logger.info("AIRFLOW TASK: BRANCHING ON DRIFT")
    logger.info("="*50)

    try:
        cleanup_mlflow_runs()

        drift_results = get_task_results("drift_detection", **context)
        evaluation_results = get_task_results("evaluate_model", **context)

        if drift_results:
            logger.info("Using drift results from XCom/previous task")
            drift_detected = drift_results.get("drift_detected", False)
            drift_score = drift_results.get("overall_drift_score", 0.0)
        else:
            logger.info("Falling back to reading drift results from file...")
            drift_report_path = "reports/drift_report.json"

            if os.path.exists(drift_report_path):
                with open(drift_report_path, 'r') as f:
                    file_drift_results = json.load(f)

                drift_detected = file_drift_results.get("drift_detected", False)
                drift_score = file_drift_results.get("overall_drift_score", 0.0)
            else:
                logger.error(f"Drift report not found at {drift_report_path}")
                return "retrain_model"

        performance_threshold_met = False
        if evaluation_results:
            performance_threshold_met = evaluation_results.get("threshold_met", False)
            accuracy = evaluation_results.get("original_data", {}).get("accuracy", 0.0)
            logger.info(f"Model performance - Accuracy: {accuracy:.4f}, Threshold met: {performance_threshold_met}")

        logger.info("Branching decision factors:")
        logger.info(f"  Drift detected: {drift_detected}")
        logger.info(f"  Overall drift score: {drift_score:.4f}")
        logger.info(f"  Performance threshold met: {performance_threshold_met}")

        if drift_detected or not performance_threshold_met:
            reason = []
            if drift_detected:
                reason.append("drift_detected")
            if not performance_threshold_met:
                reason.append("performance_threshold_not_met")

            logger.info(f"Branching to retrain_model - Reasons: {', '.join(reason)}")
            return "retrain_model"
        else:
            logger.info("No issues detected - branching to pipeline_complete")
            return "pipeline_complete"

    except Exception as e:
        logger.error(f"Branching logic failed: {e}")
        cleanup_mlflow_runs()
        return "retrain_model"

def retrain_model_task(**context):
    """
    Retrain model with original (non-drifted) data
    """
    logger.info("="*50)
    logger.info("AIRFLOW TASK: MODEL RETRAINING")
    logger.info("="*50)

    try:
        mlflow_config = get_mlflow_config_in_task()
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])

        cleanup_mlflow_runs()

        drift_results = get_task_results("drift_detection", **context)
        evaluation_results = get_task_results("evaluate_model", **context)

        retrain_reasons = []
        if drift_results and drift_results.get("drift_detected"):
            retrain_reasons.append("data_drift_detected")
        if evaluation_results and not evaluation_results.get("threshold_met", True):
            retrain_reasons.append("performance_threshold_not_met")

        logger.info(f"Retraining reasons: {retrain_reasons}")

        from src.model_training import train_with_feature_engineering

        logger.info("Retraining model with original (non-drifted) data...")
        start_time = datetime.now()

        with mlflow.start_run():
            mlflow.set_tag("model_retrained_due_to_drift", "true")
            mlflow.set_tag("retrain_reasons", ",".join(retrain_reasons))
            mlflow.set_tag("retrained_by", "airflow_dag")
            mlflow.set_tag("pipeline_stage", "retraining")
            mlflow.set_tag("original_run_timestamp", datetime.now().isoformat())

            logger.info("Calling train_with_feature_engineering with use_existing_run=True...")
            model, data = train_with_feature_engineering(use_existing_run=True)

            X_train_eng, X_test_eng, y_train_eng, y_test_eng = data['original']
            run_id = mlflow.active_run().info.run_id

            mlflow.log_metric("retrain_data_samples", X_train_eng.shape[0])
            mlflow.log_metric("retrain_features", X_train_eng.shape[1])
            mlflow.log_metric("retraining_timestamp", datetime.now().timestamp())

            if drift_results:
                mlflow.log_metric("drift_score_that_triggered_retrain", drift_results.get("overall_drift_score", 0.0))

            end_time = datetime.now()
            retrain_time = (end_time - start_time).total_seconds()
            mlflow.log_metric("retrain_time_seconds", retrain_time)

            logger.info(f"Model retrained successfully in {retrain_time:.2f} seconds")
            logger.info(f"New MLFlow Run ID: {run_id}")
            logger.info(f"Retrained on original data: {X_train_eng.shape}")

        results = {
            "model_retrained": True,
            "retrain_reasons": retrain_reasons,
            "new_mlflow_run_id": run_id,
            "retrain_timestamp": datetime.now().isoformat(),
            "retrained_data_shape": X_train_eng.shape,
            "retrain_time_seconds": retrain_time,
            "drift_score": drift_results.get("overall_drift_score", 0.0) if drift_results else 0.0,
            "mlflow_config_used": mlflow_config,
        }

        return save_task_results("retrain_model", results, **context)

    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        import traceback
        traceback.print_exc()
        cleanup_mlflow_runs()
        raise

def pipeline_complete_task(**context):
    """
    Simple completion task
    """
    logger.info("="*50)
    logger.info("AIRFLOW TASK: PIPELINE COMPLETE")
    logger.info("="*50)

    try:
        cleanup_mlflow_runs()

        pipeline_start_time = context['dag_run'].start_date
        pipeline_end_time = datetime.now()
        total_pipeline_time = (pipeline_end_time - pipeline_start_time).total_seconds()

        all_results = {
            "preprocessing": get_task_results("preprocess_data", **context),
            "feature_engineering": get_task_results("feature_engineering", **context),
            "training": get_task_results("train_model", **context),
            "evaluation": get_task_results("evaluate_model", **context),
            "drift_detection": get_task_results("drift_detection", **context),
        }

        logger.info("ML Pipeline completed successfully!")
        logger.info("No drift detected - no retraining needed")
        logger.info(f"Total pipeline execution time: {total_pipeline_time:.2f} seconds")

        summary = {
            "pipeline_status": "completed_successfully",
            "drift_detected": False,
            "retraining_required": False,
            "completion_timestamp": datetime.now().isoformat(),
            "total_execution_time_seconds": total_pipeline_time,
            "pipeline_start_time": pipeline_start_time.isoformat(),
            "pipeline_end_time": pipeline_end_time.isoformat(),
            "task_summary": {
                task: results.get("processing_time_seconds", 0) or results.get("training_time_seconds", 0) or results.get("evaluation_time_seconds", 0) or results.get("detection_time_seconds", 0)
                for task, results in all_results.items() if results
            },
            "final_model_metrics": all_results.get("evaluation", {}).get("original_data", {}),
            "airflow_config_used": airflow_config,
            "mlflow_config_used": get_mlflow_config_in_task(),
        }

        return save_task_results("pipeline_complete", summary, **context)

    except Exception as e:
        logger.error(f"❌ Pipeline completion failed: {e}")
        cleanup_mlflow_runs()
        raise


preprocess_data = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data_task,
    dag=dag,
)

feature_engineering = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering_task,
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    dag=dag,
)

evaluate_model = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model_task,
    dag=dag,
)

drift_detection = PythonOperator(
    task_id='drift_detection',
    python_callable=drift_detection_task,
    dag=dag,
)

branch_on_drift = BranchPythonOperator(
    task_id='branch_on_drift',
    python_callable=branch_on_drift,
    dag=dag,
)

retrain_model = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model_task,
    dag=dag,
    trigger_rule='none_failed',
)

pipeline_complete = PythonOperator(
    task_id='pipeline_complete',
    python_callable=pipeline_complete_task,
    dag=dag,
    trigger_rule='none_failed',
)

preprocess_data >> feature_engineering >> train_model >> evaluate_model >> drift_detection >> branch_on_drift >> [retrain_model, pipeline_complete]

dag.doc_md = """
# ML Pipeline DAG with ConfigLoader Integration

## Key Features:

### 1. **ConfigLoader Integration**
- Uses `config.config_loader.get_mlflow_config()` for dynamic MLflow URI detection
- Automatically detects Docker Compose environment vs local environment
- Falls back gracefully if ConfigLoader is not available

### 2. **Dynamic MLflow Configuration**
- `mlflow:5000` when Docker Compose is running
- `localhost:5000` when running locally
- Environment variable override support via `MLFLOW_TRACKING_URI`

### 3. **Enhanced Error Handling**
- Graceful fallback if MLflow server is unavailable
- Better logging and monitoring throughout pipeline
- Comprehensive task result tracking

### 4. **Configuration Management**
- Centralized configuration through ConfigLoader
- Environment-specific overrides
- Airflow Variables for runtime configuration

## Configuration Sources (Priority Order):
1. `MLFLOW_TRACKING_URI` environment variable
2. ConfigLoader Docker Compose detection
3. ConfigLoader fallback settings
4. Hardcoded fallback values

## MLflow Setup Process:
1. Import ConfigLoader from `config.config_loader`
2. Call `get_mlflow_config()` to get dynamic configuration
3. Set MLflow URI based on environment detection
4. Use configuration consistently across all tasks

## Benefits of ConfigLoader Integration:
- **Flexibility**: Works in both Docker and local environments
- **Maintainability**: Centralized configuration management
- **Reliability**: Graceful fallbacks prevent pipeline failures
- **Consistency**: Same configuration logic across all components
"""
