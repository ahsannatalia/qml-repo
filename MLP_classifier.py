from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.utils import algorithm_globals
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import time
from preprocess_data import preprocess_fold

# comparable model parameter are passed as default
class MLP_classifier:
    def __init__(self, k=5, seed=1, maxiter=8000, hidden=(4,), activation="tanh", solver="lbfgs"):
        self.k = k
        self.seed = seed
        # k-fold cross-validation: splits data into k equal parts, shuffled before splitting
        self.kf = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)

        self.Accuracies = []
        self.Precisions = []
        self.F1_scores = []
        self.Recalls = []
        self.TrainTimes = []
        self.maxiter = maxiter
        self.hidden = hidden
        self.activation = activation
        self.solver = solver

    def reset(self):
        # clear stored results so the same instance can be reused across experiments
        self.Accuracies.clear()
        self.Precisions.clear()
        self.Recalls.clear()
        self.F1_scores.clear()
        self.TrainTimes.clear()

    def fit_predict_evaluate(self, X, y, preprocess_mode, n_components=None):
        self.reset()

        for fold, (train_idx, test_idx) in enumerate(self.kf.split(X), start=1):
            # each fold uses a different 1/k slice as the test set
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # preprocessing is fit on train only to prevent data leakage into the test set
            X_train, X_test = preprocess_fold(X_train, X_test,
                                           mode=preprocess_mode,
                                           n_components=n_components)

            print(f"num of features: ", X_train.shape[1])

            # create a fresh classifier for each fold
            clf = MLPClassifier(hidden_layer_sizes=self.hidden, activation=self.activation, solver=self.solver, random_state=self.seed, max_iter=self.maxiter)

            # train and time the model
            start = time.perf_counter()
            clf.fit(X_train, y_train)
            end = time.perf_counter()
            self.TrainTimes.append(end - start)

            y_pred = clf.predict(X_test)

            # sanity check: confirm both classes appear in predictions and ground truth
            print(f"Fold {fold} predicted classes: {np.unique(y_pred)}")
            print(f"Fold {fold} actual classes:    {np.unique(y_test)}")

            # evaluate on the held-out test fold
            score = clf.score(X_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            F1_score = f1_score(y_test, y_pred)

            self.Accuracies.append(score)
            self.Precisions.append(precision)
            self.Recalls.append(recall)
            self.F1_scores.append(F1_score)

            print(f"Fold {fold} accuracy: {score:.4f}")
            print(f"Fold {fold} precision: {precision:.4f}")
            print(f"Fold {fold} recall: {recall:.4f}")
            print(f"Fold {fold} f1-score: {F1_score:.4f}")

    def get_train_times(self):
        return self.TrainTimes

    def get_accuracies(self):
        return self.Accuracies

    def get_precisions(self):
        return self.Precisions

    def get_recalls(self):
        return self.Recalls

    def get_F1_Scores(self):
        return self.F1_scores
