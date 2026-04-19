from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.utils import resample
import numpy as np

def preprocess_fold(X_train, X_test, mode, n_components=None):
    """
    Applies preprocessing to a single cross-validation fold.
    The scaler/PCA is always fit on X_train only and then applied to X_test,
    preventing any information from the test set leaking into the training process.

    Modes:
        'raw'    — no preprocessing, data passed through unchanged
        'scaled' — min-max scaling to [0, 1] range
        'pca'    — min-max scaling followed by PCA dimensionality reduction
    """
    if mode == "raw":
        return X_train, X_test

    elif mode == "scaled":
        # fit scaler on training data only, then apply the same transform to test
        scaler = MinMaxScaler()
        return scaler.fit_transform(X_train), scaler.transform(X_test)

    elif mode == "pca":
        if n_components is None:
            raise ValueError("n_components is not set")
        # scale first so all features contribute equally to PCA
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # fit PCA on training data only, project both splits into the same subspace
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        return X_train_pca, X_test_pca
