from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.utils import resample
import numpy as np

def preprocess(X, mode, n_components=None, sigma=None):
    if mode == "raw":
        return X
    elif mode == "scaled":
        return MinMaxScaler().fit_transform(X)
    elif mode == "pca":
        if n_components is None:
            raise ValueError("n_components is not set")
        X_scaled = MinMaxScaler().fit_transform(X)
        return PCA(n_components=n_components).fit_transform(X_scaled)
    elif mode == "gaussian":
        X_scaled = MinMaxScaler().fit_transform(X)
        noise = np.random.normal(0, sigma, size=X_scaled.shape)
        return X + noise
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")

def preprocess_fold(X_train, X_test, mode, n_components=None, sigma=None):
    if mode == "raw":
        return X_train, X_test
    elif mode == "scaled":
        scaler = MinMaxScaler()
        return scaler.fit_transform(X_train), scaler.transform(X_test)
    elif mode == "pca":
        if n_components is None:
            raise ValueError("n_components is not set")
        scaler = MinMaxScaler()                        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)       
        
        pca = PCA(n_components=n_components)          
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)     
        return X_train_pca, X_test_pca

        
def preprocess_mnist_digits(digit1, digit2, n_samples):
    print("Loading MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)

    # filter to use only two digits 
    mask = (y == digit1) | (y == digit2)
    X_filtered = X[mask]
    y_filtered = y[mask]

    y_binary = (y_filtered == digit2).astype(int)

    # resampling
    X, y = resample(X_filtered, y_binary, n_samples=n_samples, random_state=42, stratify=y_binary)

    X_reduced = preprocess(X, 'pca', n_components=3)

    print(f"Digits {digit1} vs {digit2}")
    print(f"Samples: {X_reduced.shape[0]}")
    print(f"Features after PCA: {X_reduced.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X_reduced, y
    
    




    
            
    