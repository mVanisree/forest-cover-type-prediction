"""Class-wise evaluation helpers."""
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_classwise(model, X_test, y_test):
    report = classification_report(y_test, model.predict(X_test), output_dict=True, zero_division=0)
    rows = []
    for label, metrics in report.items():
        if label.isdigit():
            rows.append({
                "Class": int(label),
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-score": metrics["f1-score"],
                "Support": int(metrics["support"]),
            })
    return pd.DataFrame(rows), confusion_matrix(y_test, model.predict(X_test))
