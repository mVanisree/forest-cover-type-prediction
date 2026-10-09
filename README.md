# Forest Cover Type Prediction Using Large-Scale Machine Learning

A reproducible Python project for multiclass forest-cover prediction using the UCI Covertype dataset.

## Project objectives
- Compare Random Forest, Extra Trees, and HistGradientBoosting classifiers.
- Rank feature importance.
- Evaluate precision, recall, F1-score, and support by class.
- Compare all 54 features against a selected feature subset.
- Study performance and training time across different training-data fractions.

## Dataset
- Source: UCI Machine Learning Repository — Covertype
- URL: https://archive.ics.uci.edu/dataset/31/covertype
- Expected data: 581,012 rows, 54 input features, 7 target classes.
- The data is downloaded using `sklearn.datasets.fetch_covtype`; no dataset file is bundled in this repository.

## Environment
The project was developed for Google Colab. It can also run locally with Python 3.10–3.12.

### Install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the pipeline
```bash
python -m src.run_all
```

The pipeline downloads the Covertype data if necessary, performs a stratified train/test split, trains three models, exports metrics, runs feature selection, evaluates class-wise metrics, runs training-size experiments, and writes plots to `results/`.

## Repository structure
```text
Forest-Cover-Type-Prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                 # Optional local dataset files
│   └── processed/           # Optional derived data
├── models/                  # Saved models and selector (generated)
├── reports/
│   └── results_summary.md
├── results/
│   ├── model_comparison.csv
│   ├── feature_selection.csv
│   ├── training_size_results.csv
│   ├── feature_importance_top10.csv
│   └── classwise_evaluation.csv
├── src/
│   ├── run_all.py
│   ├── data/load_data.py
│   ├── features/feature_selection.py
│   ├── models/train_models.py
│   ├── models/classwise_evaluation.py
│   ├── models/training_size_experiment.py
│   └── visualization/plots.py
└── tests/
```

## Existing results versus regenerated results
The CSVs initially included in `results/` are the actual metrics supplied from the user's Colab run. Running the pipeline will regenerate metrics based on the installed scikit-learn version, random seed, hardware, and implementation details; small differences are possible. The supplied class-wise table is the user's pasted report and should be checked against the exact estimator that generated it before making claims about its model name.

## Main metrics
- **Accuracy:** fraction of all predictions that are correct.
- **Macro F1:** F1 averaged equally across classes.
- **Weighted F1:** F1 averaged with class support as weights.
- **Training time:** wall-clock seconds for model fitting; hardware-dependent.


## Generated charts
The `results/` folder includes PNG charts generated from the supplied result CSVs. See `results/CHARTS_README.md` for a list and explanation. These figures visualize the provided metrics and do not imply the models were retrained during chart creation.
