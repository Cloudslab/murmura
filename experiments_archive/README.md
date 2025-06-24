# Experiments Archive - Five-Phase Experimental Design

This repository contains the complete experimental infrastructure for the paper "Network Structures as an Attack Surface: Topology-Based Privacy Leakage in Federated Learning" following the comprehensive experimental design across attack evaluation and defense mechanism validation.

## Phase Overview

Our evaluation employs a comprehensive five-phase approach that captures idealized deployment scenarios, realistic adversarial knowledge constraints, operational deployment conditions, enterprise-scale implications, and empirical defense mechanism validation.

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

### Phase 5: Defense Mechanism Evaluation
**Purpose**: Empirically validate topology-aware defense mechanisms against identified privacy attacks
- **Coverage**: 6,240 evaluations across 520 experiments with 12 defense configurations
- **Defense Mechanisms**: 
  - Structural noise injection (edge perturbation, node disguise)
  - Topology-aware differential privacy (adaptive noise scaling)
  - Dynamic topology reconfiguration (periodic reshuffling)
  - Combined layered strategies
- **Key Findings**:
  - 15.44% average attack reduction with structural noise injection
  - Layered defenses achieve up to 89.4% protection against communication pattern attacks
  - Dynamic reconfiguration reduces topology structure attacks by 31.7%
- **Objective**: Provide practical defense implementations with measurable protection against topology-based privacy attacks

## Directory Structure

```
experiments_archive/
├── phase1_baseline_analysis/          # Phase 1: Complete Knowledge Baseline
│   ├── results_phase1/                # 520 configuration results
│   ├── results_phase2/                # Extended baseline results  
│   ├── training_data/                 # Original 5-node training data
│   ├── scripts/                       # Phase 1 execution scripts
│   └── README_phase1.md              # Phase 1 specific documentation
├── phase2_realistic_knowledge/        # Phase 2: Partial Knowledge Analysis
│   ├── realistic_knowledge_full_analysis/  # 2,100 attack evaluations
│   ├── scripts/                       # Phase 2 execution scripts
│   └── README_phase2.md              # Phase 2 specific documentation
├── phase3_deployment_scenarios/       # Phase 3: Subsampling Analysis
│   ├── subsampling_analysis/          # Results with sampling effects
│   │   └── training_data_extended/    # 10-node subsampling experiments
│   ├── scripts/                       # Phase 3 execution scripts
│   └── README_phase3.md              # Phase 3 specific documentation
├── phase4_enterprise_scalability/     # Phase 4: Large-Scale Analysis
│   ├── scalability_results/           # 50-500 node synthetic analysis
│   ├── scripts/                       # Phase 4 execution scripts
│   └── README_phase4.md              # Phase 4 specific documentation
├── phase5_defense_evaluation/         # Phase 5: Defense Mechanism Validation
│   ├── defense_evaluation/            # Defense evaluation results
│   │   ├── comprehensive_phase1_results/    # 6,240 evaluation results
│   │   ├── comprehensive_defense_results/   # Defense analysis & visualizations
│   │   ├── defense_evaluation_results/      # Additional evaluation data
│   │   ├── defense_test_results/           # Simple test results
│   │   ├── defense_concerns_analysis.json   # Defense concerns analysis
│   │   ├── fair_defense_comparison_results.json  # Fair defense comparison
│   │   ├── scripts/                        # Core defense evaluation scripts
│   │   ├── documentation/                  # Complete defense evaluation report
│   │   └── README.md                      # Defense evaluation documentation
│   ├── comprehensive_layered_privacy_evaluation/  # Layered privacy results
│   ├── layered_privacy_evaluation/    # Additional privacy evaluation
│   ├── scripts/                       # Additional defense analysis scripts
│   │   ├── analyze_defense_concerns.py
│   │   ├── dynamic_reconfig_analysis.py
│   │   ├── fair_defense_comparison.py
│   │   └── topology_dp_analysis.py
│   ├── figures/                       # Phase 5 defense figures
│   └── *.py                          # Top-level defense scripts
├── figures/                           # Generated figures for all phases
│   ├── phase1_figures/                # Baseline analysis figures
│   ├── phase2_figures/                # Realistic knowledge figures
│   ├── phase3_figures/                # Deployment scenario figures
│   ├── phase4_figures/                # Scalability analysis figures
│   └── phase5_figures/                # Defense evaluation figures
├── scripts/                          # Cross-phase utilities
│   ├── analysis/                     # Analysis utilities
│   ├── attack_experiments/           # Attack experiment scripts
│   ├── figure_generation/            # Figure generation scripts
│   ├── data_processing/              # Data processing utilities
│   └── scalability_experiments/      # Scalability experiment scripts
└── docs/                             # Comprehensive documentation
    ├── EXPERIMENTAL_METHODOLOGY.md   # Complete experimental design
    ├── ATTACK_IMPLEMENTATION_COMPARISON.md  # Attack vector details
    ├── REALISTIC_PARTIAL_TOPOLOGY_KNOWLEDGE_ANALYSIS.md
    ├── SCALABILITY_ANALYSIS.md       # Scalability analysis report
    ├── TOPOLOGY_PRIVACY_LEAKAGE_PAPER.md  # Main paper
    └── README_figure_generation.md   # Figure generation guide
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

## Latest Updates

### Phase 5 Defense Evaluation (December 2024)
- Completed comprehensive evaluation of topology-aware defense mechanisms
- Added layered privacy protection analysis with multiple defense combinations
- Introduced fair defense comparison framework for unbiased evaluation
- Generated defense effectiveness visualizations and deployment recommendations
- Validated statistical significance of defense improvements

### Recent Additions
- `analyze_defense_concerns.py`: Critical analysis of defense implementation challenges
- `fair_defense_comparison.py`: Unbiased comparison framework for defense mechanisms
- `topology_dp_analysis.py`: Topology-aware differential privacy implementation
- `dynamic_reconfig_analysis.py`: Dynamic topology reconfiguration strategies

## Archive Creation Date
December 6, 2024

## Notes
- All experiments use the Murmura framework with Ray for distributed computing
- Differential privacy experiments use Opacus for privacy guarantees
- Visualizations include network topology, communication patterns, and training metrics