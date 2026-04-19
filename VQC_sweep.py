from sklearn.datasets import make_classification
from VQC_classifier import VQC_classifier
import numpy as np
import pandas as pd
import itertools

X, y = make_classification(n_samples=1000, n_features=6, n_informative=6,
                            n_redundant=0, n_classes=2, random_state=42)

featuremap_reps = [1, 2, 3]
ansatz_reps     = [1, 2, 3]
maxiters        = [100, 200, 300] 
ansatz_types    = ['RealAmplitudes', 'EfficientSU2']

# #Phase 1 — screen full grid with k=1:

phase1_results = []
total = len(featuremap_reps) * len(ansatz_reps) * len(maxiters) * len(ansatz_types)
count = 0

for ansatz_type, fm_rep, ansatz_rep, maxiter in itertools.product(
        ansatz_types, featuremap_reps, ansatz_reps, maxiters):
    count += 1
    print(f"\nPhase 1 - Config {count}/{total}")

    qclf = VQC_classifier(k=5, fm_rep=fm_rep, ansatz_rep=ansatz_rep, maxiter=maxiter, ansatz_type=ansatz_type, max_folds=1)
    qclf.fit_predict_evaluate(X, y, preprocess_mode='scaled')

    phase1_results.append({
        'ansatz_type': ansatz_type, 'fm_rep': fm_rep,
        'ansatz_rep': ansatz_rep, 'maxiter': maxiter,
        'accuracy': np.mean(qclf.get_accuracies()),
    })
    pd.DataFrame(phase1_results).to_csv('results/vqc_phase1_progress.csv', index=False)

phase1_df = pd.DataFrame(phase1_results).sort_values('accuracy', ascending=False)
print("\n=== PHASE 1 TOP 5 ===")
print(phase1_df.head(5))

# Phase 2 — full 5-fold on top 5:
top5 = phase1_df.head(5)
print("\n=== PHASE 1 TOP 5 ===")
print(top5)
phase2_results = []

for _, cfg in top5.iterrows():
    print(f"\nPhase 2 - ansatz={cfg['ansatz_type']}, fm_rep={cfg['fm_rep']}, ansatz_rep={cfg['ansatz_rep']}, maxiter={cfg['maxiter']}")

    qclf = VQC_classifier(fm_rep=int(cfg['fm_rep']),
                        ansatz_rep=int(cfg['ansatz_rep']),
                        maxiter=int(cfg['maxiter']),
                        ansatz_type=cfg['ansatz_type'])
    qclf.fit_predict_evaluate(X, y, preprocess_mode='scaled')

    phase2_results.append({
        'ansatz_type': cfg['ansatz_type'], 'fm_rep': cfg['fm_rep'],
        'ansatz_rep': cfg['ansatz_rep'], 'maxiter': cfg['maxiter'],
        'accuracy': np.mean(qclf.get_accuracies()),
        'std':      np.std(qclf.get_accuracies()),
        'time':     np.mean(qclf.get_train_times()),
    })

phase2_df = pd.DataFrame(phase2_results).sort_values('accuracy', ascending=False)
print("\n=== PHASE 2 FINAL RESULTS ===")
print(phase2_df)
phase2_df.to_csv('results/vqc_phase2_results.csv', index=False)