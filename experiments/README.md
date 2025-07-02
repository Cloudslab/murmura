# Experiment Suite Documentation

This directory contains all experiment-related code and documentation for the trust monitoring system evaluation.

## Directory Structure

```
experiments/
├── test_scripts/           # Testing and validation scripts
│   ├── comprehensive_trust_test.py
│   ├── quick_threshold_test.py
│   └── test_beta_integration.py
├── analysis_tools/         # Analysis and debugging tools
│   ├── quick_trust_analysis.py
│   ├── trust_analysis_report.py
│   ├── trust_diagnostic_tool.py
│   └── trust_exchange_debugger.py
├── documentation/          # Technical documentation and findings
│   ├── trust_analysis_findings.md
│   ├── trust_monitor_implementation_details.md
│   └── trust_monitor_root_cause_analysis.md
└── README.md              # This file
```

## Main Experiment Suite

The primary experiment suite for paper results is located in the project root:

- `paper_experiment_suite.py` - Main experiment runner using subprocess
- `experiment_suite.py` - Direct implementation runner
- `experiment_metrics_collector.py` - Automated metrics collection
- `experiment_analysis.py` - Results analysis and visualization

## Running Experiments

### Quick Test (5 experiments)
```bash
python paper_experiment_suite.py --study_type quick_test --output_dir paper_experiments/quick_test
```

### Comprehensive Study
```bash
python paper_experiment_suite.py --study_type comprehensive --output_dir paper_experiments/full_study --max_workers 6
```

### Focused Studies
- `topology_comparison` - Compare ring, line, and complete topologies
- `attack_intensity` - Analyze different attack intensities
- `scalability` - Test with different network sizes
- `cross_dataset` - Compare MNIST vs CIFAR-10

## Test Scripts

### comprehensive_trust_test.py
Comprehensive testing of trust monitoring system with various attack scenarios.

### quick_threshold_test.py
Quick validation of beta threshold functionality.

### test_beta_integration.py
Integration tests for beta distribution-based adaptive thresholding.

## Analysis Tools

### trust_diagnostic_tool.py
Interactive diagnostic tool for debugging trust monitoring issues.

### trust_exchange_debugger.py
Debug tool for trust exchange protocol issues.

### trust_analysis_report.py
Generate comprehensive analysis reports from experiment results.

### quick_trust_analysis.py
Quick analysis tool for rapid insights from experiment data.

## Documentation

Detailed technical documentation and findings from trust monitor development and debugging are available in the `documentation/` subdirectory.