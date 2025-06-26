# Trust Monitoring Baseline Experiments

This directory contains scripts for running comprehensive baseline experiments to evaluate the adaptive trust monitoring system performance before implementing attack scenarios.

## Scripts Overview

### 1. `quick_trust_test.py` - Quick Verification
Fast test to verify the trust system is working correctly.

```bash
# Run quick verification tests
python scripts/quick_trust_test.py
```

**What it does:**
- Runs minimal MNIST test (4 actors, 8 rounds)
- Runs minimal CIFAR-10 test (4 actors, 10 rounds, simple model)
- Verifies zero false positives and basic functionality
- Takes ~10-15 minutes total

**Use when:**
- First time setup verification
- After code changes
- Before running full experiments

### 2. `run_trust_baseline_experiments.py` - Comprehensive Baseline
Complete baseline experiment suite across multiple configurations.

```bash
# Run all baseline experiments
python scripts/run_trust_baseline_experiments.py

# List available experiments
python scripts/run_trust_baseline_experiments.py --list_experiments

# Run only MNIST experiments
python scripts/run_trust_baseline_experiments.py --skip_cifar10

# Run only CIFAR-10 experiments  
python scripts/run_trust_baseline_experiments.py --skip_mnist

# Run specific experiments
python scripts/run_trust_baseline_experiments.py --experiments mnist_ring_baseline cifar10_standard_baseline

# Custom output directory
python scripts/run_trust_baseline_experiments.py --output_dir my_baseline_results
```

## Experiment Configurations

### MNIST Experiments (9 total)
1. **Topology Comparisons:**
   - `mnist_ring_baseline` - Ring topology (default)
   - `mnist_complete_baseline` - Complete/all-to-all topology
   - `mnist_line_baseline` - Line topology

2. **Trust Profile Comparisons:**
   - `mnist_permissive_trust` - Lenient trust thresholds
   - `mnist_strict_trust` - Strict trust thresholds

3. **Thresholding Methods:**
   - `mnist_manual_threshold` - Fixed thresholds (no Beta adaptation)

4. **Scalability Tests:**
   - `mnist_scale_small` - 4 actors, 12 rounds
   - `mnist_scale_large` - 10 actors, 18 rounds

5. **Extended Training:**
   - `mnist_extended_training` - 25 rounds

### CIFAR-10 Experiments (8 total)
1. **Model Architecture Comparisons:**
   - `cifar10_simple_baseline` - Simple CNN model
   - `cifar10_standard_baseline` - Standard CNN model
   - `cifar10_resnet_baseline` - ResNet architecture

2. **Topology with Complex Models:**
   - `cifar10_complete_topology` - Complete topology

3. **Trust Profiles with Complex Models:**
   - `cifar10_permissive_resnet` - ResNet with permissive trust
   - `cifar10_strict_standard` - Standard model with strict trust

4. **Thresholding for Complex Models:**
   - `cifar10_manual_threshold` - Manual thresholds

5. **Scalability:**
   - `cifar10_scale_large` - 8 actors

## Expected Results

### Trust Monitoring Effectiveness
- **False Positive Rate:** 0% (zero false positives expected)
- **Average Trust Score:** ~0.95-1.0 for honest nodes
- **Exclusions/Downgrades:** 0 for honest-only scenarios

### Federated Learning Performance
- **MNIST Final Accuracy:** ~98-99%
- **CIFAR-10 Final Accuracy:** ~70-85% (depending on model complexity)
- **Convergence:** Smooth improvement over rounds

### Performance Characteristics
- **MNIST Execution Time:** ~2-5 minutes per experiment
- **CIFAR-10 Execution Time:** ~5-15 minutes per experiment (model dependent)
- **Memory Usage:** Scales with number of actors and model size

## Output Structure

Results are saved to timestamped directories:
```
trust_baseline_experiments/
└── baseline_YYYYMMDD_HHMMSS/
    ├── experiment_results.json          # Comprehensive results
    ├── mnist_ring_baseline/             # Individual experiment directories
    │   └── *results*.json
    ├── mnist_complete_baseline/
    ├── cifar10_standard_baseline/
    └── ...
```

### Key Metrics Tracked
- **FL Performance:** Final accuracy, accuracy improvement, convergence
- **Trust Monitoring:** False positive rate, trust scores, exclusions/downgrades
- **System Performance:** Execution time, resource usage
- **Configuration Impact:** Topology, trust profile, model complexity effects

## Usage Workflow

### 1. Initial Verification
```bash
# Quick test to verify setup
python scripts/quick_trust_test.py
```

### 2. Full Baseline Collection
```bash
# Run all experiments (takes 2-4 hours)
python scripts/run_trust_baseline_experiments.py
```

### 3. Selective Testing
```bash
# Test specific configurations
python scripts/run_trust_baseline_experiments.py \
  --experiments mnist_ring_baseline cifar10_standard_baseline \
  --output_dir focused_test
```

### 4. Analysis
Results include comprehensive statistics:
- Cross-experiment comparisons
- Dataset-specific performance
- Configuration impact analysis
- Summary statistics and charts

## Before Running Attacks

These baseline experiments establish:

1. **Trust System Reliability:** Zero false positives with honest nodes
2. **Performance Baselines:** FL accuracy without adversarial interference  
3. **Configuration Sensitivity:** How different settings affect performance
4. **System Scalability:** Performance across different network sizes
5. **Model Complexity Impact:** Trust monitoring effectiveness with different architectures

Use these results to:
- Validate trust system correctness
- Choose optimal configurations for attack experiments
- Establish performance baselines for comparison
- Identify potential issues before introducing adversarial scenarios

## Notes

- Each experiment runs independently (failures don't affect others)
- Results are saved incrementally (safe to interrupt and resume)
- Experiments use deterministic seeds for reproducibility
- Ray clusters are automatically managed (initialized/cleaned up per experiment)
- Failed experiments are logged with detailed error information