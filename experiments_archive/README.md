# Murmura Experiments Archive

This directory contains all experimental scripts, results, and analyses for the Murmura federated learning framework research project.

## Directory Structure

```
experiments_archive/
├── scripts/                    # All experimental scripts
│   ├── attack_experiments/     # Privacy attack experiments
│   │   ├── paper_experiments.py
│   │   └── rerun_attacks.py
│   ├── scalability_experiments/
│   │   └── scalability_experiments.py
│   ├── figure_generation/      # Paper figure generation scripts
│   │   ├── generate_correct_figures.py
│   │   ├── generate_figure1_heatmap.py
│   │   ├── generate_topology_vulnerability_figure.py
│   │   ├── generate_updated_figures.py
│   │   ├── figure_generation_summary.py
│   │   └── validate_figures.py
│   └── data_processing/        # Data analysis scripts
│       ├── combine_scalability_data.py
│       └── extract_scalability_metrics.py
│
├── results/                    # All experimental results
│   ├── attack_results/         # Privacy attack results
│   │   ├── results_phase1/     # Phase 1 attack results
│   │   └── results_phase2/     # Phase 2 attack results
│   ├── scalability_results/    # Scalability experiment data
│   │   ├── experiment_checkpoint.json
│   │   ├── scalability_analysis.json
│   │   ├── scalability_results.json
│   │   ├── scalability_summary.txt
│   │   ├── extracted_metrics.json
│   │   └── test_scalability_parallel/
│   └── training_visualizations/
│       ├── visualizations_phase1/  # 520 experiments
│       └── visualizations_phase2/  # 288 experiments
│
├── figures/                    # Generated figures for papers
│   └── analysis/              # All paper figures (PDF & PNG)
│       ├── fig1_*.pdf/png     # Attack effectiveness figures
│       ├── fig2_*.pdf/png     # Privacy mechanism figures
│       ├── fig3_*.pdf/png     # Topology vulnerability figures
│       ├── fig4_*.pdf/png     # Dataset comparison figures
│       ├── fig5_*.pdf/png     # Network scaling figures
│       └── ...
│
└── docs/                      # Documentation and papers
    ├── README_figure_generation.md
    ├── updated_results_section.tex
    ├── paper_updated_experimental_results.tex
    ├── ATTACK_IMPLEMENTATION_COMPARISON.md
    ├── SCALABILITY_ANALYSIS.md
    └── TOPOLOGY_PRIVACY_LEAKAGE_PAPER.md
```

## Experiment Types

### 1. Privacy Attack Experiments
- **Scripts**: `paper_experiments.py`, `rerun_attacks.py`
- **Results**: Located in `results/attack_results/`
- **Purpose**: Evaluate privacy vulnerabilities in different federated learning topologies

### 2. Scalability Experiments
- **Scripts**: `scalability_experiments.py`
- **Results**: JSON files in `results/scalability_results/`
- **Purpose**: Test framework performance with varying network sizes and configurations

### 3. Training Visualizations
- **Results**: CSV data and summary plots in `results/training_visualizations/`
- **Purpose**: Detailed training metrics and network communication patterns
- **Experiments**: 808 total experiments across two phases

## Key Files

### Configuration Results
- `scalability_results.json`: Complete scalability experiment data
- `scalability_analysis.json`: Processed metrics and analysis
- `experiment_checkpoint.json`: Experiment state for resumption
- `rerun_attack_results.json`: Attack effectiveness metrics

### Generated Figures
All publication-ready figures are in `figures/analysis/`, including:
- Attack effectiveness heatmaps
- Privacy degradation matrices
- Topology vulnerability comparisons
- Network scaling analyses
- Deployment risk assessments

### Documentation
- Research papers and analysis documents in `docs/`
- Figure generation instructions and methodology
- Attack implementation comparisons
- Scalability analysis reports

## Usage

To regenerate figures:
```bash
cd scripts/figure_generation
python generate_updated_figures.py
```

To run new experiments:
```bash
# Attack experiments
python scripts/attack_experiments/paper_experiments.py

# Scalability tests
python scripts/scalability_experiments/scalability_experiments.py
```

To analyze results:
```bash
python scripts/data_processing/extract_scalability_metrics.py
python scripts/data_processing/combine_scalability_data.py
```

## Archive Creation Date
December 6, 2025

## Notes
- All experiments use the Murmura framework with Ray for distributed computing
- Differential privacy experiments use Opacus for privacy guarantees
- Visualizations include network topology, communication patterns, and training metrics