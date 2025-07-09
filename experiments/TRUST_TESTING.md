# Trust System Testing Guide

The trust system testing has been consolidated into a single comprehensive test suite: `trust_test_suite.py`

## Quick Start

```bash
# Run quick 3-round test
python trust_test_suite.py quick

# Run only unit tests
python trust_test_suite.py unit

# Run integration tests
python trust_test_suite.py integration

# Run specific end-to-end scenario
python trust_test_suite.py e2e --scenario attack --intensity moderate

# Run full benchmark suite
python trust_test_suite.py benchmark

# Compare all scenarios
python trust_test_suite.py compare

# Run everything
python trust_test_suite.py all
```

## Test Types

### 1. Unit Tests
- HSIC threshold integration
- Beta threshold configuration
- Adaptive trust system components

### 2. Integration Tests
- Trust monitor with beta threshold integration
- Full trust assessment pipeline

### 3. End-to-End Tests
- **Baseline**: Honest nodes only (should have zero false positives)
- **Attack**: Single malicious node with configurable intensity
- **Stealth**: Hard-to-detect attacks with high stealth
- **Multiple**: Multiple attackers scenario

### 4. Quick Tests
- 3-round rapid iteration tests
- Baseline + moderate attack

### 5. Benchmark Tests
- Multiple datasets (MNIST, CIFAR-10)
- Different topologies (ring, fully connected)
- Various attack intensities

### 6. Comparison Tests
- Side-by-side comparison of all scenarios
- Generates comparison summary with best metrics

## Output

All test results are saved to `trust_test_results/test_run_TIMESTAMP/` with:
- `test_results.json` - Detailed test results
- `test_suite.log` - Complete test logs
- `comparison_results.json` - Comparison summary (if running compare)

## Command Options

```bash
python trust_test_suite.py [command] [options]

Commands:
  quick         Run quick 3-round test
  unit          Run unit tests only
  integration   Run integration tests
  e2e           Run end-to-end tests
  benchmark     Run comprehensive benchmarks
  compare       Compare all scenarios
  all           Run all tests

Options:
  --scenario    E2E test scenario (baseline/attack/stealth/multiple/all)
  --intensity   Attack intensity (low/moderate/high)
  --rounds      Number of FL rounds (default: 5)
  --output-dir  Output directory (default: trust_test_results)
```

## Examples

```bash
# Test specific attack intensity
python trust_test_suite.py e2e --scenario attack --intensity high --rounds 10

# Run benchmarks with custom output directory
python trust_test_suite.py benchmark --output-dir my_benchmark_results

# Quick test for development
python trust_test_suite.py quick
```

## Old Test Scripts (Deprecated)

The following test scripts have been consolidated into `trust_test_suite.py`:
- `scripts/quick_trust_test.py`
- `scripts/run_trust_baseline_experiments.py`
- `experiments/test_scripts/test_trust_system.py`
- `experiments/test_scripts/quick_threshold_test.py`
- `experiments/test_scripts/comprehensive_trust_test.py`
- `experiments/test_scripts/test_beta_integration.py`

These can be safely removed as their functionality is now available in the unified test suite.