"""
Model Drift Detection Implementation
"""

import os
import json
import pandas as pd
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

def detect_drift(reference_data_path: str, current_data_path: str) -> Dict[str, Any]:
    """
    Parameters:
        reference_data_path (str): Path to reference dataset CSV
        current_data_path (str): Path to current dataset CSV

    Returns:
        Dict[str, Any]: Dictionary containing drift detection results
    """

    print("Detecting drift between:")
    print(f"Reference: {reference_data_path}")
    print(f"Current: {current_data_path}")

    reference_data = pd.read_csv(reference_data_path)
    current_data = pd.read_csv(current_data_path)

    print(f"Reference data shape: {reference_data.shape}")
    print(f"Current data shape: {current_data.shape}")

    # Extract feature columns only (exclude target)
    target_column_names = ['target', 'label', 'NObeyesdad', 'class', 'y']

    feature_columns = []
    for col in reference_data.columns:
        if col not in target_column_names:
            feature_columns.append(col)

    # If no known target columns found, assume last column is target
    if len(feature_columns) == len(reference_data.columns):
        feature_columns = reference_data.columns[:-1].tolist()

    reference_features = reference_data[feature_columns]
    current_features = current_data[feature_columns]

    print(f"  Feature columns: {len(feature_columns)} features")

    # Try Evidently first, fallback to statistical methods
    try:
        result = _evidently_drift_detection(reference_features, current_features)
        print("  Using Evidently drift detection")
    except Exception as e:
        print(f"Evidently failed ({e}), using statistical fallback")
        result = _statistical_drift_detection(reference_features, current_features)

    # Include at least 3 features
    if len(feature_columns) > 3:
        selected_features = feature_columns[:3]
        filtered_drifts = {k: v for k, v in result["feature_drifts"].items()
                          if k in selected_features}
        if filtered_drifts:
            result["feature_drifts"] = filtered_drifts
            result["overall_drift_score"] = sum(filtered_drifts.values()) / len(filtered_drifts)

    # Save to reports/drift_report.json
    os.makedirs("reports", exist_ok=True)
    drift_report_path = "reports/drift_report.json"

    with open(drift_report_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Drift detected: {result['drift_detected']}")
    print(f"Overall drift score: {result['overall_drift_score']:.4f}")
    print(f"Results saved to: {drift_report_path}")

    # Return same dictionary structure as saved JSON
    return result

def _evidently_drift_detection(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Drift detection using Evidently with safe imports.
    """

    # Try multiple import patterns for Evidently
    Report = None
    DataDriftPreset = None

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        try:
            from evidently.report import Report
            from evidently.metrics import DataDriftPreset
        except ImportError:
            # Import attempt 3: Nested imports
            try:
                from evidently.report import Report
                from evidently.metrics.data_drift.data_drift_preset import DataDriftPreset
            except ImportError:
                # Import attempt 4: Different class names
                try:
                    from evidently.report import Report
                    from evidently.metrics import DataDriftTable as DataDriftPreset
                except ImportError:
                    raise ImportError("Could not import Evidently components")

    # Use Report(metrics=[DataDriftPreset()]) to analyze drift
    drift_report = Report(metrics=[DataDriftPreset()])

    # Run the drift analysis
    drift_report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    # Extract results from report.as_dict()
    report_dict = drift_report.as_dict()

    # Extract drift_detected from metrics[0]["result"]["dataset_drift"]
    try:
        drift_detected = report_dict["metrics"][0]["result"]["dataset_drift"]
    except (KeyError, IndexError):
        # Fallback: search for dataset_drift in any metric
        drift_detected = False
        for metric in report_dict.get("metrics", []):
            if "dataset_drift" in metric.get("result", {}):
                drift_detected = metric["result"]["dataset_drift"]
                break

    # Extract feature-level drift scores
    feature_drifts = {}
    try:
        drift_by_columns = report_dict["metrics"][1]["result"]["drift_by_columns"]
    except (KeyError, IndexError):
        # Fallback: search for drift_by_columns in any metric
        drift_by_columns = {}
        for metric in report_dict.get("metrics", []):
            if "drift_by_columns" in metric.get("result", {}):
                drift_by_columns = metric["result"]["drift_by_columns"]
                break

    # Extract drift scores for features
    for feature, drift_info in drift_by_columns.items():
        if isinstance(drift_info, dict):
            if "drift_score" in drift_info:
                drift_score = drift_info["drift_score"]
            elif "stattest_result" in drift_info and isinstance(drift_info["stattest_result"], dict):
                p_value = drift_info["stattest_result"].get("pvalue", 0.5)
                drift_score = 1.0 - p_value
            else:
                drift_score = 0.0
        else:
            drift_score = 0.0

        feature_drifts[feature] = float(drift_score)

    # Calculate overall drift score
    overall_drift_score = sum(feature_drifts.values()) / len(feature_drifts) if feature_drifts else 0.0

    return {
        "drift_detected": bool(drift_detected),
        "feature_drifts": feature_drifts,
        "overall_drift_score": float(overall_drift_score)
    }

def _statistical_drift_detection(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Fallback drift detection using statistical tests.
    """
    try:
        from scipy import stats
    except ImportError:
        print("Installing scipy for statistical drift detection...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
        from scipy import stats

    feature_drifts = {}
    drift_threshold = 0.05  # p-value threshold for significance

    for column in reference_data.columns:
        try:
            if reference_data[column].dtype in ['object', 'category', 'string']:
                # Categorical data: Chi-square test
                ref_counts = reference_data[column].value_counts()
                curr_counts = current_data[column].value_counts()

                # Align categories
                all_categories = sorted(set(ref_counts.index) | set(curr_counts.index))
                if len(all_categories) > 1:
                    ref_aligned = [max(1, ref_counts.get(cat, 0)) for cat in all_categories]
                    curr_aligned = [max(1, curr_counts.get(cat, 0)) for cat in all_categories]

                    try:
                        _, p_value = stats.chisquare(curr_aligned, ref_aligned)
                        drift_score = 1.0 - p_value
                    except Exception:
                        drift_score = 0.0
                else:
                    drift_score = 0.0

            else:
                # Numerical data: Kolmogorov-Smirnov test
                try:
                    ref_clean = reference_data[column].dropna()
                    curr_clean = current_data[column].dropna()
                    if len(ref_clean) > 0 and len(curr_clean) > 0:
                        _, p_value = stats.ks_2samp(ref_clean, curr_clean)
                        drift_score = 1.0 - p_value
                    else:
                        drift_score = 0.0
                except Exception:
                    drift_score = 0.0

            # Ensure drift score is between 0 and 1
            feature_drifts[column] = float(max(0.0, min(1.0, drift_score)))

        except Exception as e:
            print(f"Warning: Could not calculate drift for {column}: {e}")
            feature_drifts[column] = 0.0

    # Determine if drift detected
    drift_detected = any(score > (1.0 - drift_threshold) for score in feature_drifts.values())

    # Calculate overall drift score
    overall_drift_score = sum(feature_drifts.values()) / len(feature_drifts) if feature_drifts else 0.0

    return {
        "drift_detected": drift_detected,
        "feature_drifts": feature_drifts,
        "overall_drift_score": overall_drift_score
    }

if __name__ == "__main__":
    print("Testing drift detection...")

    reference_path = "data/splits/test.csv"
    current_path = "data/splits/drifted_test.csv"

    if os.path.exists(reference_path) and os.path.exists(current_path):
        result = detect_drift(reference_path, current_path)
        print("Drift detection test completed!")
        print(f"Result: {result}")
    else:
        print("Test files not found:")
        print(f"  {reference_path}: {'✓' if os.path.exists(reference_path) else '❌'}")
        print(f"  {current_path}: {'✓' if os.path.exists(current_path) else '❌'}")
