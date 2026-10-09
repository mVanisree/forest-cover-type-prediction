"""Train and compare ensemble classifiers."""
import time
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from src.data.load_data import load_covtype

def train_and_compare(test_size=0.2, random_state=42, data_home=None, n_jobs=-1):
    X, y = load_covtype(data_home=data_home)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    models = {
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=random_state, n_jobs=n_jobs),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=n_jobs),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, random_state=random_state
        ),
    }
    rows, fitted = [], {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Macro F1": f1_score(y_test, pred, average="macro"),
            "Weighted F1": f1_score(y_test, pred, average="weighted"),
            "Training Time (sec)": elapsed,
        })
        fitted[name] = model
    return X_train, X_test, y_train, y_test, pd.DataFrame(rows), fitted

if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    (root / "results").mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test, results, fitted = train_and_compare()
    results.to_csv(root / "results" / "model_comparison_regenerated.csv", index=False)
    import joblib
    for name, model in fitted.items():
        slug = name.lower().replace(" ", "_")
        joblib.dump(model, root / "models" / f"{slug}.joblib")
    print(results.to_string(index=False))
