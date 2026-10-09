"""Run the full reproducible Covertype experiments."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from src.models.train_models import train_and_compare
from src.features.feature_selection import fit_feature_selector
from src.models.classwise_evaluation import evaluate_classwise
from src.models.training_size_experiment import run_training_size_experiment
from src.visualization.plots import create_plots

def main():
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    models_dir = root / "models"
    results_dir.mkdir(exist_ok=True); models_dir.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test, comparison, fitted = train_and_compare()
    comparison.to_csv(results_dir / "model_comparison_regenerated.csv", index=False)

    # Save model artifacts; select Extra Trees by macro-F1 and keep Random Forest too.
    for name, model in fitted.items():
        joblib.dump(model, models_dir / (name.lower().replace(" ", "_") + ".joblib"))

    best_name = comparison.sort_values("Macro F1", ascending=False).iloc[0]["Model"]
    best_model = fitted[best_name]
    classwise, cm = evaluate_classwise(best_model, X_test, y_test)
    classwise.to_csv(results_dir / "classwise_evaluation_regenerated.csv", index=False)
    pd.DataFrame(cm).to_csv(results_dir / "confusion_matrix_regenerated.csv", index=False, header=False)

    selector = fit_feature_selector(X_train, y_train)
    selected_train = selector.transform(X_train)
    selected_test = selector.transform(X_test)
    selected_model = fitted["Extra Trees"].__class__(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    selected_model.fit(selected_train, y_train)
    from sklearn.metrics import accuracy_score, f1_score
    pred = selected_model.predict(selected_test)
    feature_rows = [
        {"Setup":"All 54 features", "Features":X_train.shape[1],
         "Accuracy":accuracy_score(y_test, fitted["Extra Trees"].predict(X_test)),
         "Macro F1":f1_score(y_test, fitted["Extra Trees"].predict(X_test), average="macro"),
         "Weighted F1":f1_score(y_test, fitted["Extra Trees"].predict(X_test), average="weighted")},
        {"Setup":"Selected features", "Features":selected_train.shape[1],
         "Accuracy":accuracy_score(y_test, pred),
         "Macro F1":f1_score(y_test, pred, average="macro"),
         "Weighted F1":f1_score(y_test, pred, average="weighted")},
    ]
    pd.DataFrame(feature_rows).to_csv(results_dir / "feature_selection_regenerated.csv", index=False)
    joblib.dump(selector, models_dir / "feature_selector.joblib")
    joblib.dump(selected_model, models_dir / "extra_trees_selected_features.joblib")

    # Tree impurity feature importance from Random Forest.
    import numpy as np
    rf = fitted["Random Forest"]
    names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    imp = pd.DataFrame({"Feature":names, "Importance":rf.feature_importances_}).sort_values("Importance", ascending=False)
    imp.head(10).to_csv(results_dir / "feature_importance_regenerated.csv", index=False)

    size_results = run_training_size_experiment()
    size_results.to_csv(results_dir / "training_size_regenerated.csv", index=False)
    create_plots(results_dir)
    print("\nMODEL COMPARISON\n", comparison.to_string(index=False))
    print("\nFEATURE SELECTION\n", pd.DataFrame(feature_rows).to_string(index=False))
    print("\nCLASS-WISE EVALUATION (best by macro F1:", best_name, ")\n", classwise.to_string(index=False))
    print("\nTRAINING DATA SIZE\n", size_results.to_string(index=False))
    print("\nSaved outputs in:", results_dir)
if __name__ == "__main__":
    main()
