# Scalability Analysis Framework

This document describes the scalability simulation framework for testing topology-based attacks on large federated learning networks (50-500+ nodes) without the computational overhead of actual training.

## Overview

The scalability framework addresses reviewer concerns about the original paper's testing limitation of 30 nodes by providing:

1. **Synthetic Data Generation**: Creates realistic training traces for large networks
2. **Attack Simulation**: Runs the same topology attacks on synthetic data  
3. **Complexity Analysis**: Provides theoretical bounds and extrapolation
4. **Scaling Trends**: Shows how attack effectiveness changes with network size

## Framework Components

### Core Files

- `murmura/attacks/scalability_simulator.py`: Core simulation framework
- `scalability_experiments.py`: Experiment runner script
- Output files: Results, analysis, and summary reports

### Key Classes

#### `NetworkConfig`
Configuration for simulated networks:
```python
config = NetworkConfig(
    num_nodes=500,
    topology="star",  # star, ring, complete, line
    attack_strategy="sensitive_groups",
    dp_enabled=True,
    dp_epsilon=4.0,
    num_rounds=10
)
```

#### `SyntheticDataGenerator`
Generates realistic federated learning traces:
- Communication patterns based on topology
- Parameter updates with realistic scaling
- Node characteristics reflecting attack strategies
- Differential privacy noise simulation

#### `LargeScaleAttackSimulator`
Runs actual topology attacks on synthetic data:
- Uses the same attack implementations as the original paper
- Provides complexity analysis and theoretical bounds
- Scales to 500+ nodes efficiently

## Usage

### Quick Test (4 network sizes)
```bash
python scalability_experiments.py --quick_test
```

### Comprehensive Analysis (50-500 nodes)
```bash
python scalability_experiments.py --max_nodes 500 --step_size 50
```

### With Differential Privacy
```bash
python scalability_experiments.py --include_dp --max_nodes 300
```

### Custom Configuration
```bash
python scalability_experiments.py \
  --min_nodes 100 \
  --max_nodes 1000 \
  --step_size 100 \
  --topologies star ring complete \
  --attack_strategies sensitive_groups topology_correlated \
  --include_dp \
  --output_dir ./large_scale_results
```

## Validation Against Real Data

The synthetic data generator is calibrated using patterns from the original 5-30 node experiments:

1. **Parameter Magnitude Scaling**: Based on observed norm distributions
2. **Communication Patterns**: Reflects topology-specific behaviors  
3. **DP Noise Models**: Uses the same Gaussian mechanism as real experiments
4. **Attack Strategy Patterns**: Mirrors the three attack scenarios tested

## Key Results Format

### Scalability Trends
Shows attack effectiveness vs network size:
```
STAR TOPOLOGY:
  Attack Success Rate by Network Size:
     50 nodes: 85.0% success, 0.650 signal strength
    100 nodes: 82.0% success, 0.620 signal strength  
    200 nodes: 78.0% success, 0.580 signal strength
    500 nodes: 70.0% success, 0.520 signal strength
  Trend: DECREASING (attacks become less effective with scale)
```

### Complexity Bounds
Provides theoretical analysis:
```
Tested network size range: 50-500 nodes
Extrapolation validity: VALID
Recommended max extrapolation: 1000 nodes
Information-theoretic limit: 0.825
Network effect bound: 0.942
```

### Topology Vulnerability Ranking
```
1. STAR        : 75.2% success, 0.582 avg signal
2. COMPLETE    : 68.8% success, 0.534 avg signal  
3. RING        : 62.1% success, 0.487 avg signal
4. LINE        : 58.3% success, 0.445 avg signal
```

## Addressing Reviewer Concerns

### Scalability Beyond 30 Nodes ✅
- Tests 50-500+ nodes efficiently
- Shows scaling trends and complexity bounds
- Provides extrapolation to larger networks

### Computational Feasibility ✅
- Synthetic simulation runs in minutes vs hours for real training
- No Ray cluster resource constraints
- Parallel experiment execution

### Scientific Rigor ✅
- Uses same attack implementations as original paper
- Validated against real experimental patterns
- Provides theoretical complexity analysis
- Clear extrapolation bounds and limitations

### Real-World Relevance ✅
- Network sizes relevant to production FL deployments
- Includes realistic differential privacy settings
- Analysis of practical deployment implications

## Integration with Paper

### New Section: "Large-Scale Scalability Analysis"

1. **Methodology**: Describe synthetic simulation approach
2. **Validation**: Show calibration against 5-30 node real data
3. **Results**: Present scaling trends for 50-500+ nodes
4. **Theoretical Analysis**: Complexity bounds and extrapolation limits
5. **Implications**: Discuss real-world deployment considerations

### Key Figures to Generate

1. **Figure X**: Attack effectiveness vs network size (50-500 nodes)
2. **Figure Y**: Topology vulnerability comparison at scale
3. **Figure Z**: DP effectiveness scaling analysis

### Experimental Details for Methods Section

```
Large-scale experiments (50-500 nodes) were conducted using synthetic 
simulation to overcome computational constraints. The simulation framework 
generates realistic federated learning traces calibrated against our 
empirical results from 5-30 node experiments. Synthetic data includes:
(1) communication patterns reflecting network topology, (2) parameter 
updates with realistic magnitude distributions, and (3) differential 
privacy noise matching our DP mechanisms. The same topology attack 
implementations are executed on synthetic traces, providing scaling 
analysis that would be computationally prohibitive with actual training.
```

## Running Full Scale Experiments

For the paper revision, run comprehensive experiments:

```bash
# Generate results for paper
python scalability_experiments.py \
  --min_nodes 50 \
  --max_nodes 500 \
  --step_size 25 \
  --include_dp \
  --output_dir ./paper_scalability_results

# This will generate:
# - scalability_results.json (raw data)  
# - scalability_analysis.json (processed analysis)
# - scalability_summary.txt (human-readable report)
```

The framework provides rigorous scalability analysis to address reviewer concerns while maintaining computational feasibility and scientific validity.