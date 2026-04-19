import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import sys
import os
from datetime import datetime

# Runs a paired t-test comparing each pair of models across all metrics.
# Input must be a fold-level CSV (one row per fold per seed per param value),
# with columns like: mlp_accuracy, vqc_f1, benchmark_time, class_sep, fold, seed
# Output is saved to ttest_results/

def process_file(fold_csv):
    fold_df  = pd.read_csv(fold_csv)
    print(f"\nLoaded {fold_csv} — {len(fold_df)} rows, columns: {list(fold_df.columns)}")

    # Columns follow the pattern: <model>_<metric> (e.g. mlp_accuracy, vqc_f1)
    # We identify model prefixes as those shared by more than one column,
    # then infer the experiment parameter as the remaining non-metric column.
    skip_cols = {'seed', 'fold'}
    _non_skip = [c for c in fold_df.columns if c not in skip_cols]
    _prefix_counts = {}
    for _c in _non_skip:
        if '_' in _c:
            _p = '_'.join(_c.split('_')[:-1])
            _prefix_counts[_p] = _prefix_counts.get(_p, 0) + 1
    _model_prefixes = {p for p, n in _prefix_counts.items() if n > 1}
    # the experiment param column is whichever column is not a model metric
    param_col = [c for c in _non_skip
                 if '_'.join(c.split('_')[:-1]) not in _model_prefixes
                 and c not in _model_prefixes][0]

    metric_cols = [c for c in fold_df.columns if c not in skip_cols and c != param_col]
    models  = sorted(set('_'.join(c.split('_')[:-1]) for c in metric_cols))
    metrics = sorted(set(c.split('_')[-1] for c in metric_cols))

    print(f"\nDetected param:   {param_col}")
    print(f"Detected models:  {models}")
    print(f"Detected metrics: {metrics}")

    # test all unique pairs of models
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

            # paired t-test: each pair of scores comes from the same fold/seed/param condition
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

    ttest_df   = pd.DataFrame(ttest_results)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name  = os.path.splitext(os.path.basename(fold_csv))[0]
    out_path   = os.path.join('ttest_results', f"{base_name}_ttest_{timestamp}.csv")
    ttest_df.to_csv(out_path, index=False)
    print(f"\nT-test results saved to {out_path}")

# checks if the whether a file is given 
if len(sys.argv) < 2:
    print("Usage: python run_ttest.py <file1.csv> [file2.csv ...]")
    sys.exit(1)

csv_files = sys.argv[1:]

for csv_path in csv_files:
    process_file(csv_path)
