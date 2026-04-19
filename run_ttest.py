import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import sys
import os
from datetime import datetime

# ── Usage ──────────────────────────────────────────────────────────────────
# python run_ttest.py results/your_fold_file.csv
# ---------------------------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python run_ttest.py <path_to_fold_csv>")
    sys.exit(1)

fold_csv = sys.argv[1]
fold_df  = pd.read_csv(fold_csv)
print(f"Loaded {fold_csv} — {len(fold_df)} rows, columns: {list(fold_df.columns)}")

# ── Auto-detect models and metrics from column names ───────────────────────
# Columns look like:  mlp_accuracy, vqc_f1, benchmark_time, etc.
# We extract unique prefixes (everything before the last underscore)
skip_cols = {'seed', 'fold'}
# Find model prefixes: prefixes (everything before last '_') shared by >1 columns
_non_skip = [c for c in fold_df.columns if c not in skip_cols]
_prefix_counts = {}
for _c in _non_skip:
    if '_' in _c:
        _p = '_'.join(_c.split('_')[:-1])
        _prefix_counts[_p] = _prefix_counts.get(_p, 0) + 1
_model_prefixes = {p for p, n in _prefix_counts.items() if n > 1}
param_col = [c for c in _non_skip
             if '_'.join(c.split('_')[:-1]) not in _model_prefixes
             and c not in _model_prefixes][0]   # the experiment param column (e.g. class_sep)

metric_cols = [c for c in fold_df.columns if c not in skip_cols and c != param_col]
models  = sorted(set('_'.join(c.split('_')[:-1]) for c in metric_cols))
metrics = sorted(set(c.split('_')[-1] for c in metric_cols))

print(f"\nDetected param:   {param_col}")
print(f"Detected models:  {models}")
print(f"Detected metrics: {metrics}")

comparisons = [(m1, m2) for i, m1 in enumerate(models) for m2 in models[i+1:]]

ttest_results = []
print("\n=== T-TEST RESULTS (paired, per fold across parameter values) ===")

for m1, m2 in comparisons:
    for metric in metrics:
        m1_col = f'{m1}_{metric}'
        m2_col = f'{m2}_{metric}'

        if m1_col not in fold_df.columns or m2_col not in fold_df.columns:
            continue

        m1_scores = fold_df[m1_col].values
        m2_scores = fold_df[m2_col].values

        if len(m1_scores) < 2:
            print(f"\n{metric.upper()} — {m1.upper()} vs {m2.upper()}: not enough data points")
            continue

        t_stat, p_value = ttest_rel(m1_scores, m2_scores)

        print(f"\n{metric.upper()} — {m1.upper()} vs {m2.upper()}")
        print(f"  {m1.upper()} mean: {np.mean(m1_scores):.4f}")
        print(f"  {m2.upper()} mean: {np.mean(m2_scores):.4f}")
        print(f"  t-stat:   {t_stat:.4f}")
        print(f"  p-value:  {p_value:.4f}")
        print(f"  {'Significant (p < 0.05)' if p_value < 0.05 else 'Not significant (p >= 0.05)'}")

        ttest_results.append({
            'comparison':  f'{m1}_vs_{m2}',
            'metric':      metric,
            f'{m1}_mean':  np.mean(m1_scores),
            f'{m2}_mean':  np.mean(m2_scores),
            't_stat':      t_stat,
            'p_value':     p_value,
            'significant': p_value < 0.05,
        })

ttest_df = pd.DataFrame(ttest_results)
timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
base_name  = os.path.splitext(os.path.basename(fold_csv))[0]
out_path   = os.path.join('ttest_results', f"{base_name}_ttest_{timestamp}.csv")
ttest_df.to_csv(out_path, index=False)
print(f"\nT-test results saved to {out_path}")
