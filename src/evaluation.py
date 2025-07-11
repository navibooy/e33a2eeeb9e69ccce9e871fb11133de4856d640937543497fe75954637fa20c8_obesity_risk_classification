import os

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(
    model_path,
    X_test,
    y_test,
    report_path="reports/metrics.txt",
    plot_path="reports/confusion_matrix.png",
):
    """
    Loads a trained model from disk, makes predictions on the test set, evaluates its performance,
    and saves the evaluation metrics and confusion matrix to the reports directory.

    Steps performed:
    - Loads the model using joblib
    - Generates predictions on the test set
    - Calculates accuracy and classification report
    - Saves metrics to a text file (metrics.txt)
    - Plots and saves a labeled confusion matrix as an image

    Parameters:
        model_path (str): File path to the trained model (.pkl format)
        X_test (pd.DataFrame or np.ndarray): Test feature set
        y_test (pd.Series or np.ndarray): True labels for the test set
        report_path (str): Path to save the textual evaluation metrics (default: "reports/metrics.txt")
        plot_path (str): Path to save the confusion matrix image (default: "reports/confusion_matrix.png")

    Returns:
        None
    """
    # Load model
    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    class_labels_num = [0, 1, 2, 3]
    cm = confusion_matrix(y_test, y_pred, labels=class_labels_num)

    # Save text report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"✅ Metrics saved to {report_path}")

    # Plot and save confusion matrix
    class_labels = ["Underweight", "Normal_weight", "Overweight", "Obesity"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap="Blues", xticks_rotation="vertical", values_format="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.savefig(plot_path)
    print(f"📊 Confusion matrix saved to {plot_path}")
    plt.close()
