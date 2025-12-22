#!/usr/bin/env python3
"""Generate publication-ready figures for personalization in non-IID conditions.

Combined figures (averaged across all datasets) highlighting Evidential Trust advantages:
1. Non-IID robustness - accuracy across heterogeneity levels
2. Performance degradation from IID to non-IID
3. Personalization quality with uncertainty
4. Convergence speed comparison
5. Ablation study - robustness to hyperparameters

Usage:
    python experiments/paper/generate_figures.py
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Publication-ready IEEE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
})

# Color palette (colorblind-friendly)
COLORS = {
    'evidential_trust': '#E64B35',  # Red - highlight
    'fedavg': '#4DBBD5',            # Cyan
    'krum': '#00A087',              # Teal
    'balance': '#3C5488',           # Blue
    'sketchguard': '#F39B7F',       # Orange
    'ubar': '#8491B4',              # Purple
}

ALGO_NAMES = {
    'evidential_trust': 'Evidential Trust (Ours)',
    'fedavg': 'FedAvg',
    'krum': 'Krum',
    'balance': 'BALANCE',
    'sketchguard': 'Sketchguard',
    'ubar': 'UBAR',
}

# Short names for bar labels
ALGO_SHORT = {
    'evidential_trust': 'Evidential\nTrust',
    'fedavg': 'FedAvg',
    'krum': 'Krum',
    'balance': 'BALANCE',
    'sketchguard': 'Sketch-\nguard',
    'ubar': 'UBAR',
}

DATASETS = ['uci_har', 'pamap2', 'ppg_dalia']
ALGORITHMS = ['fedavg', 'balance', 'sketchguard', 'ubar', 'evidential_trust']

PAPER_DIR = Path(__file__).parent


def load_results(filename: str) -> Dict:
    """Load results from JSON file."""
    filepath = PAPER_DIR / filename
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return {}


def get_accuracy(results: Dict, key: str) -> Tuple[float, float]:
    """Get accuracy and std from results."""
    result = results.get('results', {}).get(key, {})
    if 'error' in result or not result:
        return None, None
    return result.get('final_accuracy'), result.get('final_std')


def get_avg_across_datasets(results: Dict, algo: str, alpha: str) -> Tuple[float, float, float]:
    """Get average accuracy and std across all datasets."""
    accs = []
    stds = []
    for dataset in DATASETS:
        key = f"heterogeneity__{dataset}__{algo}_alpha{alpha}"
        acc, std = get_accuracy(results, key)
        if acc is not None:
            accs.append(acc * 100)
            stds.append(std * 100 if std else 0)

    if accs:
        return np.mean(accs), np.std(accs), np.mean(stds)
    return 0, 0, 0


def fig1_noniid_robustness(results: Dict, output_dir: Path):
    """Figure 1: Non-IID Robustness - Combined accuracy across heterogeneity levels."""

    alphas = ['01', '05', '10']
    alpha_labels = ['α=0.1\n(High Heterogeneity)', 'α=0.5\n(Medium)', 'α=1.0\n(Low/IID)']

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(alphas))
    width = 0.14

    for i, algo in enumerate(ALGORITHMS):
        means = []
        errs = []
        for alpha in alphas:
            mean_acc, std_acc, _ = get_avg_across_datasets(results, algo, alpha)
            means.append(mean_acc)
            errs.append(std_acc)

        offset = (i - len(ALGORITHMS)/2 + 0.5) * width
        ax.bar(x + offset, means, width,
               yerr=errs, capsize=2,
               label=ALGO_NAMES[algo],
               color=COLORS[algo],
               edgecolor='black' if algo == 'evidential_trust' else 'none',
               linewidth=2 if algo == 'evidential_trust' else 0,
               error_kw={'linewidth': 1})

    ax.set_xlabel('Data Heterogeneity Level (Dirichlet α)')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Model Performance Across Data Heterogeneity Levels\n(Averaged Across UCI HAR, PAMAP2, PPG-DaLiA)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(alpha_labels)
    ax.set_ylim(0, 105)
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.legend(loc='lower right', frameon=True, fancybox=False)

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        plt.savefig(output_dir / f'fig1_noniid_robustness.{fmt}', format=fmt, bbox_inches='tight')
    print("Saved: fig1_noniid_robustness.pdf/png")
    plt.close()


def fig2_performance_degradation(results: Dict, output_dir: Path):
    """Figure 2: Performance degradation from IID to Non-IID (combined)."""

    fig, ax = plt.subplots(figsize=(7, 5))

    degradations = []
    colors = []
    labels = []

    for algo in ALGORITHMS:
        # Average degradation across datasets
        degs = []
        for dataset in DATASETS:
            key_01 = f"heterogeneity__{dataset}__{algo}_alpha01"
            key_10 = f"heterogeneity__{dataset}__{algo}_alpha10"
            acc_01, _ = get_accuracy(results, key_01)
            acc_10, _ = get_accuracy(results, key_10)

            if acc_01 and acc_10:
                deg = (acc_10 - acc_01) * 100
                degs.append(deg)

        if degs:
            degradations.append(np.mean(degs))
            colors.append(COLORS[algo])
            labels.append(ALGO_SHORT[algo])

    x_pos = np.arange(len(ALGORITHMS))
    bars = ax.bar(x_pos, degradations, color=colors,
                  edgecolor=['black' if a == 'evidential_trust' else 'none' for a in ALGORITHMS],
                  linewidth=[2 if a == 'evidential_trust' else 0 for a in ALGORITHMS])

    # Add value labels on bars
    for i, (bar, deg) in enumerate(zip(bars, degradations)):
        height = bar.get_height()
        ax.annotate(f'{deg:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Performance Degradation (%)\n(IID α=1.0 → Non-IID α=0.1)')
    ax.set_title('Robustness to Data Heterogeneity\n(Lower is Better)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.axhline(y=0, color='black', linewidth=0.8)

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        plt.savefig(output_dir / f'fig2_degradation.{fmt}', format=fmt, bbox_inches='tight')
    print("Saved: fig2_degradation.pdf/png")
    plt.close()


def fig3_personalization_quality(results: Dict, output_dir: Path):
    """Figure 3: Personalization quality at high heterogeneity (α=0.1)."""

    fig, ax = plt.subplots(figsize=(7, 5))

    # At α=0.1 (most challenging non-IID setting)
    accs = []
    stds = []
    colors = []
    labels = []

    for algo in ALGORITHMS:
        mean_acc, _, mean_std = get_avg_across_datasets(results, algo, '01')
        accs.append(mean_acc)
        stds.append(mean_std)
        colors.append(COLORS[algo])
        labels.append(ALGO_SHORT[algo])

    x_pos = np.arange(len(ALGORITHMS))
    bars = ax.bar(x_pos, accs, color=colors,
                  edgecolor=['black' if a == 'evidential_trust' else 'none' for a in ALGORITHMS],
                  linewidth=[2 if a == 'evidential_trust' else 0 for a in ALGORITHMS],
                  yerr=stds, capsize=4, error_kw={'linewidth': 1.5, 'capthick': 1.5})

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Personalization in Highly Non-IID Settings (α=0.1)\n(Error bars show std dev across nodes)',
                 fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    # Add accuracy annotations
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, acc + stds[i] + 2),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        plt.savefig(output_dir / f'fig3_personalization.{fmt}', format=fmt, bbox_inches='tight')
    print("Saved: fig3_personalization.pdf/png")
    plt.close()


def fig4_convergence_speed(results: Dict, output_dir: Path):
    """Figure 4: Convergence speed comparison."""

    fig, ax = plt.subplots(figsize=(7, 5))

    conv_rounds = []
    colors = []
    labels = []

    for algo in ALGORITHMS:
        rounds = []
        for dataset in DATASETS:
            key = f"heterogeneity__{dataset}__{algo}_alpha01"
            result = results.get('results', {}).get(key, {})
            if 'convergence_round' in result and result['convergence_round']:
                rounds.append(result['convergence_round'])

        if rounds:
            conv_rounds.append(np.mean(rounds))
            colors.append(COLORS[algo])
            labels.append(ALGO_SHORT[algo])
        else:
            conv_rounds.append(0)
            colors.append(COLORS[algo])
            labels.append(ALGO_SHORT[algo])

    x_pos = np.arange(len(ALGORITHMS))
    bars = ax.bar(x_pos, conv_rounds, color=colors,
                  edgecolor=['black' if a == 'evidential_trust' else 'none' for a in ALGORITHMS],
                  linewidth=[2 if a == 'evidential_trust' else 0 for a in ALGORITHMS])

    # Add value labels on bars
    for bar, rounds in zip(bars, conv_rounds):
        height = bar.get_height()
        ax.annotate(f'{rounds:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Rounds to Convergence')
    ax.set_title('Convergence Speed Comparison\n(Lower is Better)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)

    # Highlight fastest
    min_idx = np.argmin([r if r > 0 else float('inf') for r in conv_rounds])
    ax.annotate('Fastest', xy=(min_idx, conv_rounds[min_idx] + 3),
                ha='center', fontsize=9, color='#E64B35', fontweight='bold')

    # Add speedup annotation
    fedavg_rounds = conv_rounds[0]
    et_rounds = conv_rounds[-1]
    if fedavg_rounds > 0 and et_rounds > 0:
        speedup = fedavg_rounds / et_rounds
        ax.annotate(f'{speedup:.1f}x faster\nthan FedAvg',
                    xy=(len(ALGORITHMS)-1, et_rounds + 5),
                    ha='center', fontsize=9, color='#E64B35',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='#E64B35', alpha=0.8))

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        plt.savefig(output_dir / f'fig4_convergence.{fmt}', format=fmt, bbox_inches='tight')
    print("Saved: fig4_convergence.pdf/png")
    plt.close()


def fig5_ablation_study(results_ablation: Dict, output_dir: Path):
    """Figure 5: Ablation study - robustness to hyperparameters."""

    fig, ax = plt.subplots(figsize=(8, 5))

    # Group by hyperparameter type, averaging across datasets
    param_types = {
        'accuracy_weight': 'Accuracy\nWeight (λ)',
        'self_weight': 'Self\nWeight (ω)',
        'trust_threshold': 'Trust\nThreshold (τ)',
        'vacuity_threshold': 'Vacuity\nThreshold (ν)',
    }

    means = []
    stds = []
    mins = []
    maxs = []
    labels = []

    for param_type, label in param_types.items():
        accs = []
        for key, val in results_ablation.get('results', {}).items():
            if 'error' in val:
                continue
            if param_type in key:
                accs.append(val.get('final_accuracy', 0) * 100)

        if accs:
            means.append(np.mean(accs))
            stds.append(np.std(accs))
            mins.append(min(accs))
            maxs.append(max(accs))
            labels.append(label)

    x_pos = np.arange(len(labels))

    # Plot bars with error bars showing full range
    bars = ax.bar(x_pos, means, color=COLORS['evidential_trust'],
                  edgecolor='black', linewidth=1.5,
                  yerr=stds, capsize=5, error_kw={'linewidth': 1.5, 'capthick': 1.5})

    # Add annotations showing the range
    for i, (mean, mn, mx) in enumerate(zip(means, mins, maxs)):
        ax.annotate(f'{mn:.0f}-{mx:.0f}%',
                    xy=(i, mean + stds[i] + 1),
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Hyperparameter')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Ablation Study: Hyperparameter Sensitivity\n(Averaged across all datasets and parameter values)',
                 fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(75, 105)
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        plt.savefig(output_dir / f'fig5_ablation.{fmt}', format=fmt, bbox_inches='tight')
    print("Saved: fig5_ablation.pdf/png")
    plt.close()


def fig6_combined_summary(results: Dict, output_dir: Path):
    """Figure 6: Combined 2x2 summary figure."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    alphas = ['01', '05', '10']
    alpha_labels = ['α=0.1', 'α=0.5', 'α=1.0']

    # (a) Bar chart: Accuracy at each heterogeneity level
    ax = axes[0, 0]
    x = np.arange(len(alphas))
    width = 0.14

    for i, algo in enumerate(ALGORITHMS):
        means = []
        for alpha in alphas:
            mean_acc, _, _ = get_avg_across_datasets(results, algo, alpha)
            means.append(mean_acc)

        offset = (i - len(ALGORITHMS)/2 + 0.5) * width
        ax.bar(x + offset, means, width,
               label=ALGO_NAMES[algo],
               color=COLORS[algo],
               edgecolor='black' if algo == 'evidential_trust' else 'none',
               linewidth=1.5 if algo == 'evidential_trust' else 0)

    ax.set_xlabel('Heterogeneity Level')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(a) Accuracy vs. Heterogeneity', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(alpha_labels)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=8, frameon=True)

    # (b) Degradation bar chart
    ax = axes[0, 1]
    degradations = []
    for algo in ALGORITHMS:
        degs = []
        for dataset in DATASETS:
            key_01 = f"heterogeneity__{dataset}__{algo}_alpha01"
            key_10 = f"heterogeneity__{dataset}__{algo}_alpha10"
            acc_01, _ = get_accuracy(results, key_01)
            acc_10, _ = get_accuracy(results, key_10)
            if acc_01 and acc_10:
                degs.append((acc_10 - acc_01) * 100)
        degradations.append(np.mean(degs) if degs else 0)

    x_pos = np.arange(len(ALGORITHMS))
    bars = ax.bar(x_pos, degradations,
                  color=[COLORS[a] for a in ALGORITHMS],
                  edgecolor=['black' if a == 'evidential_trust' else 'none' for a in ALGORITHMS],
                  linewidth=[1.5 if a == 'evidential_trust' else 0 for a in ALGORITHMS])

    for bar, deg in zip(bars, degradations):
        ax.annotate(f'{deg:.1f}%', xy=(bar.get_x() + bar.get_width()/2, deg + 0.5),
                    ha='center', fontsize=8, fontweight='bold')

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Degradation (%)')
    ax.set_title('(b) IID→Non-IID Degradation (Lower=Better)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ALGO_SHORT[a] for a in ALGORITHMS], fontsize=8)
    ax.axhline(y=0, color='black', linewidth=0.8)

    # (c) Personalization at α=0.1 with std
    ax = axes[1, 0]
    accs = []
    stds = []
    for algo in ALGORITHMS:
        mean_acc, _, mean_std = get_avg_across_datasets(results, algo, '01')
        accs.append(mean_acc)
        stds.append(mean_std)

    bars = ax.bar(x_pos, accs,
                  color=[COLORS[a] for a in ALGORITHMS],
                  edgecolor=['black' if a == 'evidential_trust' else 'none' for a in ALGORITHMS],
                  linewidth=[1.5 if a == 'evidential_trust' else 0 for a in ALGORITHMS],
                  yerr=stds, capsize=3)

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(c) Personalization at α=0.1 (Error=Std Dev)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ALGO_SHORT[a] for a in ALGORITHMS], fontsize=8)
    ax.set_ylim(0, 110)

    # (d) Convergence speed
    ax = axes[1, 1]
    conv_rounds = []
    for algo in ALGORITHMS:
        rounds = []
        for dataset in DATASETS:
            key = f"heterogeneity__{dataset}__{algo}_alpha01"
            result = results.get('results', {}).get(key, {})
            if 'convergence_round' in result and result['convergence_round']:
                rounds.append(result['convergence_round'])
        conv_rounds.append(np.mean(rounds) if rounds else 0)

    bars = ax.bar(x_pos, conv_rounds,
                  color=[COLORS[a] for a in ALGORITHMS],
                  edgecolor=['black' if a == 'evidential_trust' else 'none' for a in ALGORITHMS],
                  linewidth=[1.5 if a == 'evidential_trust' else 0 for a in ALGORITHMS])

    for bar, rounds in zip(bars, conv_rounds):
        ax.annotate(f'{rounds:.0f}', xy=(bar.get_x() + bar.get_width()/2, rounds + 0.5),
                    ha='center', fontsize=8, fontweight='bold')

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Rounds')
    ax.set_title('(d) Convergence Speed (Lower=Better)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ALGO_SHORT[a] for a in ALGORITHMS], fontsize=8)

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        plt.savefig(output_dir / f'fig6_combined_summary.{fmt}', format=fmt, bbox_inches='tight')
    print("Saved: fig6_combined_summary.pdf/png")
    plt.close()


def main():
    output_dir = PAPER_DIR / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experiment results...")
    results_hetero = load_results('results_heterogeneity.json')
    results_ablation = load_results('results_ablation.json')

    if not results_hetero:
        print("ERROR: No heterogeneity results found!")
        return

    print(f"\nGenerating figures in: {output_dir}")
    print("=" * 60)

    print("\n[1/6] Non-IID robustness...")
    fig1_noniid_robustness(results_hetero, output_dir)

    print("\n[2/6] Performance degradation...")
    fig2_performance_degradation(results_hetero, output_dir)

    print("\n[3/6] Personalization quality...")
    fig3_personalization_quality(results_hetero, output_dir)

    print("\n[4/6] Convergence speed...")
    fig4_convergence_speed(results_hetero, output_dir)

    if results_ablation:
        print("\n[5/6] Ablation study...")
        fig5_ablation_study(results_ablation, output_dir)
    else:
        print("\n[5/6] Skipping ablation (no data)")

    print("\n[6/6] Combined summary...")
    fig6_combined_summary(results_hetero, output_dir)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}/")
    print("\nGenerated figures:")
    for f in sorted(output_dir.glob('fig*.pdf')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
