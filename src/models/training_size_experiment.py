"""Evaluate Random Forest at multiple fractions of the training partition."""
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from src.data.load_data import load_covtype

def run_training_size_experiment(fractions=(0.10, 0.25, 0.50, 0.75, 1.0),
                                 test_size=0.2, random_state=42, data_home=None,
                                 n_estimators=200, n_jobs=-1):
    X, y = load_covtype(data_home=data_home)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    rows = []
    for fraction in fractions:
        if fraction < 1:
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train, train_size=fraction, random_state=random_state, stratify=y_train
            )
        else:
            X_sub, y_sub = X_train, y_train
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
        start = time.perf_counter()
        model.fit(X_sub, y_sub)
        elapsed = time.perf_counter() - start
        pred = model.predict(X_test)
        rows.append({
            "Training Fraction": fraction,
            "Training Samples": len(y_sub),
            "Accuracy": accuracy_score(y_test, pred),
            "Macro F1": f1_score(y_test, pred, average="macro"),
            "Training Time (sec)": elapsed,
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    result = run_training_size_experiment()
    result.to_csv(root / "results" / "training_size_regenerated.csv", index=False)
    print(result.to_string(index=False))
