from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from qiskit.primitives import StatevectorSampler as Sampler
import time
from preprocess_data import preprocess_fold

# the parameters from the sweep are set as default
class VQC_classifier:
    def __init__(self, k=5, seed=1, maxiter=100, fm_rep=1, ansatz_rep=2, ansatz_type='EfficientSU2', max_folds=None):
        self.k = k
        self.seed = seed
        self.kf = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)
        
        self.Accuracies = []
        self.Precisions = []
        self.F1_scores = []
        self.Recalls = []
        self.maxiter = maxiter
        self.TrainTimes = []
        self.objective_values = []
        self.fm_rep = fm_rep
        self.ansatz_rep = ansatz_rep
        self.ansatz_type = ansatz_type
        self.max_folds = max_folds

    def reset(self):
        self.Accuracies.clear()
        self.Precisions.clear()
        self.Recalls.clear()
        self.F1_scores.clear()
        self.TrainTimes.clear()
        self.objective_values.clear()

    def fit_predict_evaluate(self, X, y, preprocess_mode, n_components=None):
        self.reset()
        sampler = Sampler()

        for fold, (train_idx, test_idx) in enumerate(self.kf.split(X), start=1):
            if self.max_folds is not None and fold > self.max_folds:
                break
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            fold_objectives = []

            X_train, X_test = preprocess_fold(X_train, X_test, mode=preprocess_mode, n_components=n_components)
            
            num_features = X_train.shape[1]
            print(f"num of features: ", num_features)

            def callback(weights, obj_value):
                fold_objectives.append(obj_value)
        
            feature_map = ZZFeatureMap(feature_dimension=num_features, reps=self.fm_rep)

            if self.ansatz_type == 'EfficientSU2':
                ansatz = EfficientSU2(num_qubits=num_features, reps=self.ansatz_rep)
            else:
                ansatz = RealAmplitudes(num_qubits=num_features, reps=self.ansatz_rep)
        
            # using an optimiser
            optimizer = COBYLA(maxiter=self.maxiter)
            vqc = VQC(
                sampler=sampler,
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=optimizer,
                callback=callback
            )

            # training
            start = time.perf_counter()
            vqc.fit(X_train, y_train)
            end = time.perf_counter()
            self.TrainTimes.append(end - start)

            # predicting
            y_pred = vqc.predict(X_test)

            # check convergence
            print(f"Fold {fold} predicted classes: {np.unique(y_pred)}")
            print(f"Fold {fold} actual classes:    {np.unique(y_test)}")

            # evaluating on unseen data
            score = vqc.score(X_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            F1_score = f1_score(y_test, y_pred)
            
            # appending the score to the arrays
            self.Accuracies.append(score)
            self.Precisions.append(precision)
            self.Recalls.append(recall)
            self.F1_scores.append(F1_score)
        
            print(f"Fold {fold} accuracy: {score:.4f}")
            print(f"Fold {fold} precision: {precision:.4f}")
            print(f"Fold {fold} recall: {recall:.4f}")
            print(f"Fold {fold} f1-score: {F1_score:.4f}")
            self.objective_values.append(fold_objectives)

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

    def get_objective_values(self):
        return self.objective_values
