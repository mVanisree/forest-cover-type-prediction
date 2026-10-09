"""Feature selection helpers."""
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

def fit_feature_selector(X_train, y_train, random_state=42, threshold="median", n_estimators=200):
    """Fit selector on training data only to avoid test-data leakage."""
    estimator = ExtraTreesClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1,
        class_weight=None
    )
    selector = SelectFromModel(estimator, threshold=threshold)
    selector.fit(X_train, y_train)
    return selector
