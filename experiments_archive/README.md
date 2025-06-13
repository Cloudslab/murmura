# Experiments Archive - Four-Phase Experimental Design

This repository contains the complete experimental infrastructure for the paper "Network Structures as an Attack Surface: Topology-Based Privacy Leakage in Federated Learning" following the four-phase experimental design outlined in Section 4.5.

## Phase Overview

Our evaluation employs a comprehensive four-phase approach that captures idealized deployment scenarios, realistic adversarial knowledge constraints, operational deployment conditions, and enterprise-scale implications.

### Phase 1: Baseline Attack Effectiveness (Complete Knowledge)
**Purpose**: Establish baseline attack effectiveness through exhaustive evaluation across 520 unique configurations
- **Coverage**: Systematic evaluation across datasets, attack strategies, FL paradigms, network topologies, network sizes, and differential privacy protection levels
- **Objective**: Provides theoretical upper bounds on privacy leakage under complete topology knowledge without sampling effects
- **Results**: 84.1%, 65.0%, and 47.2% success rates for communication pattern, parameter magnitude, and topology structure attacks respectively

### Phase 2: Realistic Adversarial Knowledge Analysis
**Purpose**: Evaluate attack robustness across realistic adversarial knowledge constraints
- **Coverage**: Analysis of 2,100 attack instances spanning six knowledge scenarios
- **Scenarios**: Complete knowledge, statistical topology awareness, local neighborhood visibility (1-hop and 2-hop), and organizational structure knowledge (coarse and fine-grained)
- **Key Finding**: 80% of realistic partial knowledge scenarios maintain effectiveness above security thresholds

### Phase 3: Deployment Scenarios with Subsampling
**Purpose**: Evaluate realistic deployment scenarios incorporating client and data subsampling
- **Coverage**: 288 targeted configurations across moderate, strong, and very strong subsampling scenarios
- **Subsampling Levels**: 
  - Moderate: 50% clients, 80% data
  - Strong: 30% clients, 60% data  
  - Very Strong: 20% clients, 50% data
- **Objective**: Quantify privacy amplification effects and validate findings under practical deployment constraints

### Phase 4: Enterprise-Scale Analysis
**Purpose**: Address enterprise-scale implications through synthetic simulation methodology
- **Scale**: Networks with 50-500 nodes
- **Methodology**: Synthetic simulation framework calibrated against empirical data from smaller-scale experiments
- **Objective**: Provide critical insights into scalability patterns and extrapolate findings to production deployment scenarios

## Directory Structure

```
experiments_archive/
├── phase1_baseline_analysis/          # Phase 1: Complete Knowledge Baseline
│   ├── results_phase1/                # 520 configuration results
│   ├── results_phase2/                # Extended baseline results  
│   ├── scripts/                       # Phase 1 execution scripts
│   └── README_phase1.md              # Phase 1 specific documentation
├── phase2_realistic_knowledge/        # Phase 2: Partial Knowledge Analysis
│   ├── realistic_knowledge_full_analysis/  # 2,100 attack evaluations
│   ├── scripts/                       # Phase 2 execution scripts
│   └── README_phase2.md              # Phase 2 specific documentation
├── phase3_deployment_scenarios/       # Phase 3: Subsampling Analysis
│   ├── subsampling_analysis/          # Results with sampling effects
│   ├── scripts/                       # Phase 3 execution scripts
│   └── README_phase3.md              # Phase 3 specific documentation
├── phase4_enterprise_scalability/     # Phase 4: Large-Scale Analysis
│   ├── scalability_results/           # 50-500 node synthetic analysis
│   ├── scripts/                       # Phase 4 execution scripts
│   └── README_phase4.md              # Phase 4 specific documentation
├── figures/                           # Generated figures for all phases
│   ├── phase1_figures/                # Baseline analysis figures
│   ├── phase2_figures/                # Realistic knowledge figures
│   ├── phase3_figures/                # Deployment scenario figures
│   └── phase4_figures/                # Scalability analysis figures
├── scripts/                          # Cross-phase utilities
│   ├── analysis/                     # Analysis utilities
│   ├── figure_generation/            # Figure generation scripts
│   └── data_processing/              # Data processing utilities
└── docs/                             # Comprehensive documentation
    ├── EXPERIMENTAL_METHODOLOGY.md   # Complete experimental design
    ├── ATTACK_IMPLEMENTATION.md      # Attack vector details
    └── STATISTICAL_ANALYSIS.md       # Statistical validation methods
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