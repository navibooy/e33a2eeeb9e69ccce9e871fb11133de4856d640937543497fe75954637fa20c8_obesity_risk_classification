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
    X_test_path,
    y_test_path,
    report_path="reports/metrics.txt",
    plot_path="reports/confusion_matrix.png",
):
    """
    Loads a trained model and test data from disk, evaluates its performance,
    and saves the metrics and confusion matrix to disk.

    Parameters:
        model_path (str): Path to trained model (.pkl)
        X_test_path (str): Path to X_test.pkl
        y_test_path (str): Path to y_test.pkl
        report_path (str): Path to save classification report (default: "reports/metrics.txt")
        plot_path (str): Path to save confusion matrix plot (default: "reports/confusion_matrix.png")

    Returns:
        None
    """

    model = joblib.load(model_path)
    X_test = joblib.load(X_test_path)
    y_test = joblib.load(y_test_path)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    class_labels_num = [0, 1, 2, 3]
    class_labels = ["Underweight", "Normal_weight", "Overweight", "Obesity"]
    cm = confusion_matrix(y_test, y_pred, labels=class_labels_num)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"Metrics saved to: {report_path}")

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap="Blues", xticks_rotation="vertical", values_format="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Confusion matrix saved to: {plot_path}")
