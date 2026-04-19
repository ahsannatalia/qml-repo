import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'no-latex'])

# Each entry is (path_to_csv, experiment_param_column, output_file_prefix)
# To plot a different experiment, add or edit entries here
experiments = [
    ('/Users/nataliaahsan/qml/results/final results /final_flip_y_exp.csv', 'flip_y', 'flip_y'),
]

metrics = ['accuracy', 'precision', 'recall', 'f1', 'time']
models  = ['mlp', 'benchmark', 'vqc']
labels  = {'mlp': 'MLP', 'benchmark': 'Benchmark MLP', 'vqc': 'VQC'}

for path, param, name in experiments:
    df = pd.read_csv(path)
    print(df.columns.tolist())

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))

        for model in models:
            col     = f'{model}_{metric}'
            std_col = f'{model}_std_{metric}'
            if col in df.columns:
                # use std as error bars if available
                yerr = df[std_col] if std_col in df.columns else None
                ax.errorbar(df[param], df[col], yerr=yerr, label=labels[model], capsize=3)

        ax.set_xlabel(param)
        # invert x-axis for class_sep so harder conditions appear on the right
        if param == 'class_sep':
            ax.invert_xaxis()
        ax.set_ylabel(metric.capitalize())
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig(f'images/{name}_{metric}_2.png', dpi=300, bbox_inches='tight')
        plt.close()
