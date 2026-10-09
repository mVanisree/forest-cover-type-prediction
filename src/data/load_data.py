"""Load the UCI Covertype dataset using scikit-learn."""
from sklearn.datasets import fetch_covtype

def load_covtype(data_home=None):
    """Return X, y for UCI Covertype. Download happens on first call."""
    bunch = fetch_covtype(data_home=data_home, as_frame=False, download_if_missing=True)
    return bunch.data, bunch.target
