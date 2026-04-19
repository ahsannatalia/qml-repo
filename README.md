# Advantages of Quantum Machine Learning - Dissertation Project
Variant 4 - Comparing the training time and performance of quantum and classical neural networks


Implementation code for the dissertation:
**"Do quantum neural networks offer a practical advantage over classical neural networks in performance and training efficiency across varying data conditions?"**
Natalia Ahsan — King's College London - k23080184, 2026

---

## Overview

This repository contains the full implementation used to benchmark a Variational Quantum Classifier (VQC) against a Multi-Layer Perceptron (MLP) on binary classification tasks. Experiments vary one dataset difficulty parameter at a time to isolate the conditions under which each model performs better or worse. Statistical significance and effect sizes are computed using paired t-tests and Cohen's d.

---

## Repository Structure

```
.
├── VQC_classifier.py    # VQC using ZZFeatureMap + EfficientSU2/RealAmplitudes + COBYLA
├── MLP_classifier.py    # MLP wrapper (sklearn MLPClassifier) with k-fold CV
├── preprocess_data.py   # Preprocessing utilities
├── VQC_sweep.py         # Two-phase hyperparameter sweep for VQC
├── MLP_sweep.py         # Hyperparameter sweep for MLP
├── experiments.ipynb    # Main experiment notebook (configure and run all experiments)
├── plot_results.py      # Generate metric plots from result CSVs
├── run_ttest.py         # Paired t-test on fold-level result CSVs
├── run_cohens_d.py      # Cohen's d effect size analysis
├── requirements.txt     # Python dependency versions
├── final results 2.0/   # Final experiment result CSVs (averaged + fold-level) used in dissertation
├── final images/        # Final publication-quality plots used in dissertation
├── ttest_results/       # Output from run_ttest.py
└── cohens_d_results/    # Output from run_cohens_d.py
```

---

## Models
After a hyperparameter sweep.

### Variational Quantum Classifier (VQC)

- **Feature map:** ZZFeatureMap (Qiskit), `reps=1`
- **Ansatz:** EfficientSU2, `reps=2`
- **Optimiser:** COBYLA, `maxiter=100`
- **Simulator:** Qiskit `StatevectorSampler`
- **Input:** 4 features (after PCA reduction where applicable)

### Classical MLP (Comparable)

- **Architecture:** 1 hidden layer, 4 units
- **Activation:** tanh
- **Solver:** lbfgs
- **Max iterations:** 8000
- Architecture chosen to match the VQC's parameter count for a fair comparison.

### Classical MLP (Benchmark)

- **Architecture:** 2 hidden layer, 32 and 16 units
- **Activation:** relu
- **Solver:** lbfgs
- **Max iterations:** 8000
- Architecture chosen to match the VQC's parameter count for a fair comparison.

### Evaluation Protocol

Both models use 5-fold stratified cross-validation with the same random seeds. Metrics recorded per fold: accuracy, precision, recall, F1-score, and training time.

---

## Experiments

All experiments use `sklearn.datasets.make_classification` (1000 samples, 4 informative features, binary classification) and vary one parameter at a time. Three random seeds are used per condition; results are averaged across seeds and folds.

The notebook saves fresh output to a local `results/` directory (created automatically, not included in this repository). The pre-computed results used in the dissertation runs are provided in `final results 2.0/`.

---

## Setup

### Requirements

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

---

## Reproducing the Experiments

### 1. Hyperparameter sweeps (optional — best configs already identified)

```bash
python VQC_sweep.py   # two-phase grid search: fm_reps, ansatz_reps, maxiter, ansatz_type
python MLP_sweep.py   # grid search: hidden layers, activation, solver, max_iter
```

Results are saved as CSVs in the current directory.

### 2. Run experiments

Open `experiments.ipynb` in Jupyter and edit the **Experiment Configuration** cell at the top. Set the `EXPERIMENT` parameter (e.g. `class_sep`) and optionally change `RESULTS_DIR`. Run all cells — results are saved automatically as timestamped CSVs.

```bash
jupyter lab experiments.ipynb
```

### 3. Statistical analysis

Fold-level CSVs (named `*_folds_*.csv`) are required as input.

```bash
# Paired t-test
python run_ttest.py <file1.csv> [file2.csv ...]
```

Results are saved to `ttest_results/`.

```bash
# Cohen's d effect size
# --time-only - runs the analysis for only time columns in the csv file
python run_cohens_d.py [--time-only] <file1.csv> [file2.csv ...]
```

Results are saved to `cohens_d_results/`.

### 4. Generate plots

Edit the `experiments` list in `plot_results.py` to point to the desired result CSV, then run:

```bash
python plot_results.py
```

Plots are saved to `images/` (created automatically if not present). The plots used in the dissertation are provided in `final images/`.

---

## Data

All classification datasets are synthetic and generated programmatically via `sklearn.datasets.make_classification`. No external data download is required to reproduce the main results.

---

## Citation

If you use this code, please cite:

```
[Your Name]. ([Year]). [Dissertation Title]. [University].
Zenodo. https://doi.org/[your-doi-here]
```

---

## License

[MIT / CC BY 4.0 — choose one]

This code was written as part of an undergraduate dissertation and is made available for reproducibility purposes.
