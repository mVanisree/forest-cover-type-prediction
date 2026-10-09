# Generated results and charts

These PNG charts were generated from the supplied result CSV files:
- `model_comparison.png`: accuracy, macro F1, and weighted F1 for the three classifiers.
- `feature_importance_top10.png`: top 10 feature importance ranking.
- `training_size_performance.png`: accuracy and macro F1 versus training sample count.
- `training_size_runtime.png`: training time versus training sample count.
- `classwise_f1_score.png`: class-wise F1-scores.
- `classwise_recall.png`: class-wise recall.
- `feature_selection_comparison.png`: comparison of all 54 features and selected 27 features.

The class-wise CSV is preserved exactly as supplied by the project owner. Its source model should be verified in the original Colab notebook before labelling it as the best model. These charts visualize the CSV values; they do not retrain any models.
