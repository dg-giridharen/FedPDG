"""
experiments/statistical_tests.py — Statistical Significance Analysis
Wilcoxon signed-rank tests, LaTeX table generation, and convergence plots.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config


class StatisticalAnalyzer:
    """Complete statistical analysis and visualization pipeline."""

    def __init__(self, results_dir='./results', alpha=0.05):
        self.results_dir = results_dir
        self.alpha = alpha
        self.tables_dir = os.path.join(results_dir, 'tables')
        self.plots_dir = os.path.join(results_dir, 'plots')
        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def load_all_results(self, datasets, seeds):
        """Load results from JSON files across all datasets and seeds."""
        all_results = {}
        for ds in datasets:
            all_results[ds] = {}
            for seed in seeds:
                path = os.path.join(
                    self.results_dir, f'results_{ds}_seed{seed}.json')
                if os.path.exists(path):
                    with open(path) as f:
                        data = json.load(f)
                    for method, metrics in data.items():
                        if method not in all_results[ds]:
                            all_results[ds][method] = {
                                'accuracy': [], 'f1_macro': [],
                                'detection_rate': [], 'false_alarm_rate': [],
                            }
                        for m in ['accuracy', 'f1_macro',
                                  'detection_rate', 'false_alarm_rate']:
                            val = metrics.get(m, 0)
                            if isinstance(val, dict):
                                continue
                            all_results[ds][method][m].append(float(val))
        return all_results

    def run_wilcoxon_tests(self, all_results, metric='f1_macro'):
        """Wilcoxon signed-rank test: FedPDG vs each baseline."""
        print(f"\n{'='*60}")
        print(f"Wilcoxon Signed-Rank Tests (α={self.alpha}) — {metric}")
        print(f"{'='*60}")

        comparisons = {}
        for ds, methods in all_results.items():
            if 'FedPDG' not in methods:
                continue
            fedpdg_scores = methods['FedPDG'].get(metric, [])
            if len(fedpdg_scores) < 3:
                print(f"  {ds}: Not enough seeds for Wilcoxon (need ≥3)")
                continue

            print(f"\n  Dataset: {ds}")
            for method, scores_dict in methods.items():
                if method == 'FedPDG':
                    continue
                baseline_scores = scores_dict.get(metric, [])
                if len(baseline_scores) != len(fedpdg_scores):
                    continue

                try:
                    stat, p_val = wilcoxon(fedpdg_scores, baseline_scores)
                    sig = '**' if p_val < 0.01 else (
                        '*' if p_val < 0.05 else 'ns')
                    print(f"    FedPDG vs {method:12s}: "
                          f"p={p_val:.4f} {sig}")
                    comparisons[f"{ds}_{method}"] = {
                        'statistic': stat, 'p_value': p_val,
                        'significant': p_val < self.alpha,
                    }
                except Exception as e:
                    print(f"    FedPDG vs {method:12s}: FAILED ({e})")

        return comparisons

    def build_results_table(self, all_results, methods_order=None):
        """Build formatted results table with mean ± std."""
        if methods_order is None:
            methods_order = ['FedAvg', 'FedProx', 'Krum', 'FLAME',
                             'FLTrust', 'FedPDG']

        rows = []
        for ds, methods in all_results.items():
            for method in methods_order:
                if method not in methods:
                    continue
                scores = methods[method]
                row = {'Dataset': ds, 'Method': method}
                for m in ['accuracy', 'f1_macro',
                          'detection_rate', 'false_alarm_rate']:
                    vals = scores.get(m, [0])
                    mean_v = np.mean(vals)
                    std_v = np.std(vals)
                    row[m] = f"{mean_v:.4f} ± {std_v:.4f}"
                rows.append(row)

        df = pd.DataFrame(rows)
        print("\n" + "="*80)
        print("RESULTS TABLE")
        print("="*80)
        print(df.to_string(index=False))

        # Save CSV
        csv_path = os.path.join(self.tables_dir, 'main_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSaved → {csv_path}")

        # Save LaTeX
        tex_path = os.path.join(self.tables_dir, 'main_results.tex')
        df.to_latex(tex_path, index=False, caption='Main Results',
                    label='tab:main_results')
        print(f"Saved → {tex_path}")

        return df

    def plot_convergence(self, convergence_data, save_name='convergence.pdf'):
        """Plot F1 convergence curves with std shading."""
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {
            'FedPDG': '#e74c3c', 'FedAvg': '#3498db',
            'FedProx': '#2ecc71', 'Krum': '#9b59b6',
            'FLAME': '#f39c12', 'FLTrust': '#1abc9c',
        }

        for method, data in convergence_data.items():
            rounds = data.get('round', [])
            means = np.array(data.get('f1_macro', []))
            c = colors.get(method, '#888888')
            lw = 2.5 if method == 'FedPDG' else 1.5
            ax.plot(rounds, means, label=method, color=c, linewidth=lw)

        ax.set_xlabel('FL Round', fontsize=12)
        ax.set_ylabel('F1-Macro Score', fontsize=12)
        ax.set_title('Convergence Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)

        path = os.path.join(self.plots_dir, save_name)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Convergence plot → {path}")

    def plot_byzantine_robustness(self, byz_results,
                                  save_name='byzantine_robustness.pdf'):
        """Plot F1 vs Byzantine ratio."""
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {
            'FedPDG': '#e74c3c', 'FedAvg': '#3498db',
            'FedProx': '#2ecc71', 'Krum': '#9b59b6',
            'FLAME': '#f39c12', 'FLTrust': '#1abc9c',
        }

        for method, data in byz_results.items():
            ratios = sorted(data.keys())
            means = [np.mean(data[r]) for r in ratios]
            stds = [np.std(data[r]) for r in ratios]
            c = colors.get(method, '#888888')
            lw = 2.5 if method == 'FedPDG' else 1.5

            ax.errorbar(ratios, means, yerr=stds, label=method,
                        color=c, linewidth=lw, capsize=3, marker='o')

        ax.set_xlabel('Byzantine Client Ratio', fontsize=12)
        ax.set_ylabel('F1-Macro Score', fontsize=12)
        ax.set_title('Byzantine Robustness', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        path = os.path.join(self.plots_dir, save_name)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Byzantine plot → {path}")

    def plot_pds_heatmap(self, pds_data, save_name='pds_heatmap.pdf'):
        """Heatmap of PDS scores per client per round."""
        fig, ax = plt.subplots(figsize=(12, 5))

        # pds_data: list of dicts {client_id: score} per round
        rounds = list(range(len(pds_data)))
        clients = sorted(set().union(*[d.keys() for d in pds_data]))

        matrix = np.zeros((len(clients), len(rounds)))
        for r, rd in enumerate(pds_data):
            for i, cid in enumerate(clients):
                matrix[i, r] = rd.get(cid, 0)

        sns.heatmap(matrix, ax=ax, cmap='YlOrRd',
                    xticklabels=[f'R{r}' for r in rounds],
                    yticklabels=[f'C{c}' for c in clients])
        ax.set_xlabel('FL Round')
        ax.set_ylabel('Client ID')
        ax.set_title('PDS Scores Across Rounds', fontweight='bold')

        path = os.path.join(self.plots_dir, save_name)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PDS heatmap → {path}")


def main():
    config = Config()
    analyzer = StatisticalAnalyzer(config.RESULTS_DIR, config.SIGNIFICANCE_LEVEL)

    datasets = config.DATASETS
    seeds = config.RANDOM_SEEDS

    print("Loading results from all experiments...")
    all_results = analyzer.load_all_results(datasets, seeds)

    if not all_results:
        print("No result files found. Run main_experiment.py first.")
        return

    # 1. Build & Save Tables
    analyzer.build_results_table(all_results)

    # 2. Wilcoxon tests
    analyzer.run_wilcoxon_tests(all_results, 'f1_macro')
    analyzer.run_wilcoxon_tests(all_results, 'accuracy')

    # 3. Plot Convergence (using first dataset's FedPDG history)
    # Note: Baselines in current JSON don't have history, so this plots FedPDG
    ds_example = datasets[0]
    seed_example = seeds[0]
    path = os.path.join(config.RESULTS_DIR, f'results_{ds_example}_seed{seed_example}.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        conv_data = {}
        for method, metrics in data.items():
            if isinstance(metrics, dict) and 'convergence' in metrics:
                conv_data[method] = metrics['convergence']
        if conv_data:
            analyzer.plot_convergence(conv_data, save_name=f'convergence_{ds_example}.pdf')
            # Also plot PDS heatmap if available
            if 'FedPDG' in conv_data and 'pds_scores' in conv_data['FedPDG']:
                analyzer.plot_pds_heatmap(conv_data['FedPDG']['pds_scores'], 
                                        save_name=f'pds_heatmap_{ds_example}.pdf')

    # 4. Plot Byzantine Robustness (using CICIDS2017)
    byz_path = os.path.join(config.RESULTS_DIR, f'byzantine_CICIDS2017_seed{seeds[0]}.json')
    if os.path.exists(byz_path):
        with open(byz_path) as f:
            byz_data_raw = json.load(f)
        # Reshape: {method: {ratio: [scores]}}
        byz_results = {}
        for ratio_str, methods in byz_data_raw.items():
            ratio = float(ratio_str)
            for method, metrics in methods.items():
                if method not in byz_results:
                    byz_results[method] = {}
                if ratio not in byz_results[method]:
                    byz_results[method][ratio] = []
                byz_results[method][ratio].append(metrics.get('f1_macro', 0))
        
        analyzer.plot_byzantine_robustness(byz_results, save_name='byzantine_robustness_CICIDS2017.pdf')

    print("\nVisual analysis complete. Check ./results/plots/ and ./results/tables/")


if __name__ == '__main__':
    main()
