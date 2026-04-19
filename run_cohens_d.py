import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Computes Cohen's d effect size for each model pair across all metrics.
# --time-only restricts analysis to training time columns only.
# Input must be a fold-level CSV (same format as run_ttest.py).
# Output is saved to cohens_d_results/
#
# Cohen's d (paired): d = mean(x1 - x2) / std(x1 - x2)
# Effect size guide:  negligible < 0.2, small ≥ 0.2, medium ≥ 0.5, large ≥ 0.8

def cohens_d_paired(x1, x2):
    diff = x1 - x2
    return np.mean(diff) / np.std(diff, ddof=1)

def effect_label(d):
    a = abs(d)
    if a < 0.2:
        return "Negligible"
    elif a < 0.5:
        return "Small"
    elif a < 0.8:
        return "Medium"
    else:
        return "Large"

def process_file(fold_csv, time_only=False):
    fold_df = pd.read_csv(fold_csv)
    print(f"\nLoaded {fold_csv} — {len(fold_df)} rows, columns: {list(fold_df.columns)}")

    # Same logic as run_ttest.py: identifies model prefixes and the experiment param.
    # std columns are excluded since they are derived, not raw fold scores.
    skip_std  = {c for c in fold_df.columns if 'std' in c.split('_')}
    skip_cols = {'seed', 'fold'} | skip_std
    _non_skip = [c for c in fold_df.columns if c not in skip_cols]
    _prefix_counts = {}
    for _c in _non_skip:
        if '_' in _c:
            _p = '_'.join(_c.split('_')[:-1])
            _prefix_counts[_p] = _prefix_counts.get(_p, 0) + 1
    _model_prefixes = {p for p, n in _prefix_counts.items() if n > 1}
    param_col = [c for c in _non_skip
                 if '_'.join(c.split('_')[:-1]) not in _model_prefixes
                 and c not in _model_prefixes][0]

    metric_cols = [c for c in fold_df.columns if c not in skip_cols and c != param_col]
    if time_only:
        # restrict to training time when --time-only is passed in the command
        metric_cols = [c for c in metric_cols if c.endswith('_time')]
    models  = sorted(set('_'.join(c.split('_')[:-1]) for c in metric_cols))
    metrics = sorted(set(c.split('_')[-1] for c in metric_cols))

    print(f"\nDetected param:   {param_col}")
    print(f"Detected models:  {models}")
    print(f"Detected metrics: {metrics}")

    # test all unique pairs of models
    comparisons = [(m1, m2) for i, m1 in enumerate(models) for m2 in models[i+1:]]

    cohens_d_results = []
    print("\n=== COHEN'S D RESULTS ===")

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

            d = cohens_d_paired(m1_scores, m2_scores)
            label = effect_label(d)

            print(f"\n{metric.upper()} — {m1.upper()} vs {m2.upper()}")
            print(f"  {m1.upper()} mean: {np.mean(m1_scores):.4f}")
            print(f"  {m2.upper()} mean: {np.mean(m2_scores):.4f}")
            print(f"  Cohen's d: {d:.4f}")
            print(f"  Effect size: {label}")

            cohens_d_results.append({
                'comparison':   f'{m1}_vs_{m2}',
                'metric':       metric,
                f'{m1}_mean':   np.mean(m1_scores),
                f'{m2}_mean':   np.mean(m2_scores),
                'cohens_d':     d,
                'effect_size':  label,
            })

    out_df    = pd.DataFrame(cohens_d_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(fold_csv))[0]
    suffix    = '_time' if time_only else ''
    out_dir   = 'cohens_d_results'
    os.makedirs(out_dir, exist_ok=True)
    out_path  = os.path.join(out_dir, f"{base_name}{suffix}_cohens_d_{timestamp}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nCohen's d results saved to {out_path}")

# checks if no files are given
if len(sys.argv) < 2:
    print("Usage: python run_cohens_d.py [--time-only] <file1.csv> [file2.csv ...]")
    sys.exit(1)

time_only = '--time-only' in sys.argv
csv_files = [a for a in sys.argv[1:] if not a.startswith('--')]

for csv_path in csv_files:
    process_file(csv_path, time_only=time_only)
