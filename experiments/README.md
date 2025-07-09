# Experiment Suite Documentation

This directory contains all experiment-related code and documentation for the trust monitoring system evaluation.

## Directory Structure

```
experiments/
├── trust_test_suite.py        # Unified comprehensive test suite
├── TRUST_TESTING.md          # Testing documentation and usage guide
├── documentation/             # Technical documentation and findings
│   ├── trust_analysis_findings.md
│   ├── trust_monitor_implementation_details.md
│   └── trust_monitor_root_cause_analysis.md
└── README.md                 # This file
```

## Main Experiment Suite

The primary experiment suite for paper results is located in the project root:

- `paper_experiment_suite.py` - Main experiment runner using subprocess
- `experiment_suite.py` - Direct implementation runner
- `experiment_metrics_collector.py` - Automated metrics collection
- `experiment_analysis.py` - Results analysis and visualization

## Unified Test Suite

The **trust_test_suite.py** is the consolidated testing framework that replaces all previous scattered test scripts. It provides:

### Test Types
- **Unit Tests** - Component testing (HSIC, Beta thresholds, Adaptive trust)
- **Integration Tests** - Trust monitor integration with Ray actors
- **End-to-End Tests** - Full FL scenarios (baseline, attacks, stealth, multiple attackers)
- **Quick Tests** - 3-round rapid iteration tests
- **Benchmark Tests** - Comprehensive performance testing across datasets/topologies
- **Comparison Tests** - Side-by-side scenario comparison

### Usage Examples

```bash
# Quick 3-round test for development
python experiments/trust_test_suite.py quick

# Run only unit tests
python experiments/trust_test_suite.py unit

# Test specific attack scenario
python experiments/trust_test_suite.py e2e --scenario attack --intensity moderate

# Full benchmark suite
python experiments/trust_test_suite.py benchmark

# Compare all scenarios
python experiments/trust_test_suite.py compare

# Run everything
python experiments/trust_test_suite.py all
```

### Output
All results are saved to timestamped directories with:
- Detailed test results (JSON)
- Complete test logs
- Comparison summaries

See `TRUST_TESTING.md` for complete usage documentation.

## Running Paper Experiments

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

## Documentation

Detailed technical documentation and findings from trust monitor development and debugging are available in the `documentation/` subdirectory.

## Migration Notes

The following files have been consolidated into `trust_test_suite.py`:
- Previous test scripts from `test_scripts/` directory
- Analysis tools from `analysis_tools/` directory
- Standalone test scripts from `scripts/` directory

This provides a unified, comprehensive testing framework with better organization and functionality.