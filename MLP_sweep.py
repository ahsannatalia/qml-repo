from MLP_classifier import MLP_classifier
from sklearn.datasets import make_classification
import pandas as pd
import itertools
import numpy as np

# Synthetic binary classification dataset — same across all configs for a fair comparison
X, y = make_classification(n_samples=1000, n_features=6, n_informative=6, n_redundant=0, n_classes=2, random_state=42)

# Comparable MLP sweep — small architectures to match VQC parameter count                                                                                                                                      
hidden_layers_comparable = [(2,), (3,), (4,)]                                                                                                                                                                  
                                                                                                                                                                                                                
# Benchmark MLP sweep — larger architectures for upper-bound reference                                                                                                                                         
hidden_layers_benchmark  = [(2,), (3,), (4,), (8, 4), (32, 16)]    

# Set MODE to 'comparable' or 'benchmark' to switch between sweeps
MODE = 'benchmark'
hidden_layers = hidden_layers_comparable if MODE == 'comparable' else hidden_layers_benchmark
activations   = ['tanh', 'relu']
solvers       = ['lbfgs', 'adam']
max_iters     = [1000, 5000, 8000]

results = []
total = len(hidden_layers) * len(activations) * len(solvers) * len(max_iters)
count = 0

# exhaustive grid search over all combinations
for hidden, activation, solver, max_iter in itertools.product(
        hidden_layers, activations, solvers, max_iters):
    count += 1
    print(f"Config {count}/{total}: hidden={hidden}, activation={activation}, solver={solver}, max_iter={max_iter}")

    clf = MLP_classifier(hidden=hidden, activation=activation,
                        solver=solver, maxiter=max_iter)
    clf.fit_predict_evaluate(X, y, preprocess_mode='scaled')

    results.append({
        'hidden': hidden, 'activation': activation,
        'solver': solver, 'max_iter': max_iter,
        'accuracy': np.mean(clf.get_accuracies()),
        'std':      np.std(clf.get_accuracies()),
    })
    print(f"  Accuracy: {results[-1]['accuracy']:.4f}")

results_df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
print("\n=== TOP 10 CONFIGS ===")
print(results_df.head(10))
# save full sweep results for inspection
results_df.to_csv(f'mlp_sweep_results_{MODE}.csv', index=False)
