# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research codebase for decentralized federated learning experiments, implementing Byzantine-resilient aggregation algorithms (BALANCE, Sketchguard, UBAR) over peer-to-peer graph topologies. The project uses the LEAF benchmark datasets (FEMNIST, CelebA) and includes experiment orchestration, data extraction, and visualization tools.

## Core Architecture

### Main Components

1. **`decentralized_fl_sim.py`** (1939 lines) - Core simulation engine
   - Implements decentralized federated learning with multiple aggregation strategies
   - Supports 5 aggregation algorithms: d-fedavg, krum, balance, sketchguard, ubar
   - Includes attack simulation (Gaussian, directed deviation)
   - Key classes:
     - `Graph`: Network topology structures (ring, fully-connected, Erdos-Renyi, k-regular)
     - `BALANCE`: Original Byzantine-resilient algorithm with distance-based filtering
     - `Sketchguard`: Count-Sketch compression for lightweight robust aggregation
     - `UBAR`: Two-stage Byzantine-resilient (distance filtering + loss evaluation)
     - `LocalModelPoisoningAttacker`: Adversarial node simulation

2. **`leaf_datasets.py`** (414 lines) - LEAF dataset loaders
   - PyTorch implementations of LEAF benchmark datasets
   - Classes: `LEAFFEMNISTDataset`, `LEAFCelebADataset`
   - Models: `LEAFFEMNISTModel`, `LEAFCelebAModel`
   - Function: `create_leaf_client_partitions()` - creates federated data partitions from LEAF user data

3. **`model_variants.py`** (271 lines) - Model scaling variants
   - Different sized FEMNIST architectures for scaling experiments
   - `FEMNISTTiny` (~200K params), `FEMNISTSmall` (~800K), `FEMNISTBaseline` (~6.5M)
   - `FEMNISTLarge` (~26M), `FEMNISTXLarge` (~52M)
   - CelebA variants: `CelebATiny`, `CelebASmall`, `CelebABaseline`

4. **`run_experiments.py`** - Batch experiment orchestration
   - Systematically runs experiments across parameter combinations
   - Generates log files in format: `{dataset}_{rounds}_{local_epochs}_{graph}_{agg}_{attack}.log`

5. **`leaf/`** - LEAF benchmark framework (original implementation)
   - Data preprocessing scripts in `leaf/data/{dataset}/preprocess/`
   - Original TensorFlow models in `leaf/models/`
   - Note: Main simulations use PyTorch (not these TensorFlow models)

### Aggregation Algorithms

The simulator supports 5 aggregation methods (specified via `--agg`):

- **d-fedavg**: Decentralized FedAvg (baseline)
- **krum**: Multi-Krum Byzantine-resilient aggregation
- **balance**: Distance-based filtering with adaptive thresholds
- **sketchguard**: Count-Sketch compression for filtering + full model aggregation
- **ubar**: Two-stage approach (distance filtering → loss evaluation)

## Running Experiments

### Basic Simulation

```bash
python decentralized_fl_sim.py \
    --dataset femnist \
    --num-nodes 20 \
    --rounds 10 \
    --local-epochs 3 \
    --graph fully \
    --batch-size 64 \
    --lr 0.001 \
    --agg ubar \
    --attack-percentage 0.2 \
    --attack-type gaussian \
    --verbose
```

### Common Parameters

- `--dataset`: femnist, celeba
- `--graph`: ring, fully, erdos (with `--p`), k-regular (with `--k`)
- `--agg`: d-fedavg, krum, balance, sketchguard, ubar
- `--attack-type`: directed_deviation, gaussian
- `--attack-percentage`: Fraction of nodes compromised (0.0-1.0)
- `--max-samples`: Limit samples per client (useful for testing)

### Algorithm-Specific Parameters

**Krum:**
- `--pct-compromised`: Expected fraction of Byzantine nodes

**BALANCE:**
- `--balance-gamma`: Threshold multiplier (default: 2)
- `--balance-kappa`: Time decay rate (default: 1)
- `--balance-alpha`: Aggregation weight (default: 0.5)

**Sketchguard:**
- `--sketch-size`: Count-Sketch compression size (default: 10000)
- `--sketch-cols`: Hash functions count (default: 5)

**UBAR:**
- `--ubar-rho`: Fraction of neighbors to accept in stage 1 (typically 1.0 - attack_percentage)

### Batch Experiments

```bash
# Run all experiment combinations
python run_experiments.py

# Dry run (print commands without executing)
python run_experiments.py --dry-run

# Skip existing results
python run_experiments.py --skip-existing
```

This systematically varies:
- Datasets: femnist, celeba
- Graphs: ring, fully, erdos (p=0.6, 0.45, 0.2)
- Aggregation methods: all 5
- Attack percentages: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8

Results are saved to `results/` directory as `.log` files.

### Model Scaling Experiments

```bash
# Run scaling experiments with different model sizes
python run_model_scaling_experiments.py

# Run scalability experiments (varying number of nodes)
python run_scalability_experiments.py
```

## Data Extraction and Visualization

### Extract Data from Logs

```bash
# Extract accuracy metrics
python scripts/data_extraction/extract_accuracies.py

# Extract timing data
python scripts/data_extraction/extract_timing_data.py
```

### Generate Figures

```bash
# Main paper-style accuracy comparisons
python scripts/visualization/visualize_paper_style.py

# Topology-based analysis
python scripts/visualization/visualize_topology_style.py

# Scaling analysis
python scripts/visualization/visualize_scalability.py
```

All visualization scripts output PDF and PNG files to the project root.

## LEAF Dataset Preparation

### FEMNIST

```bash
cd leaf/data/femnist
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample
```

This creates `leaf/data/femnist/data/train/` and `test/` directories with JSON files.

### CelebA

```bash
cd leaf/data/celeba
# Download CelebA dataset first, then:
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample
```

### Dataset Structure

LEAF datasets are stored as JSON files with structure:
```json
{
  "users": ["user1", "user2", ...],
  "num_samples": [100, 150, ...],
  "user_data": {
    "user1": {"x": [...], "y": [...]},
    ...
  }
}
```

The PyTorch dataset loaders (`LEAFFEMNISTDataset`, `LEAFCelebADataset`) automatically load all JSON files from train/test directories and create federated partitions.

## Device Selection

The codebase automatically selects the best available device:
1. CUDA (if available)
2. MPS (Apple Silicon)
3. CPU (fallback)

Device is selected via `device()` function in decentralized_fl_sim.py:69.

## Log File Format

Experiment logs follow the naming convention:
```
{dataset}_{rounds}_{local_epochs}_{clients}_{graph}_{batch_size}_{max_samples}_{agg}_{attack_pct}attack[_attack_type]_{lambda}.log
```

Example:
```
femnist_20_10_3_erdos_06_64_10000_ubar_20attack_gaussian_1lambda.log
```

## Key Implementation Details

### Graph Topologies

Graph construction in `make_graph()` (decentralized_fl_sim.py:87):
- **ring**: Each node connects to k neighbors (k must be even)
- **fully**: Complete graph (all-to-all)
- **erdos**: Erdos-Renyi with probability p
- **k-regular**: Each node has exactly k neighbors

Graphs are bidirectional (edges added in both directions).

### Model Aggregation Flow

1. **Local Training**: Each node trains on its local data for `--local-epochs` epochs
2. **Model Exchange**: Nodes share model parameters with graph neighbors
3. **Aggregation**: Each node applies aggregation algorithm to received models
4. **Update**: Local model updated with aggregated parameters

For UBAR and Sketchguard, additional filtering steps occur before aggregation.

### Attack Mechanisms

- **gaussian**: Adds Gaussian noise to model parameters (mean=0, std=10)
- **directed_deviation**: Scales model parameters by -5 to create opposite gradients

Attackers do NOT aggregate honestly - they broadcast malicious models while keeping their own frozen.

## Dependencies

Main dependencies (PyTorch-based simulation):
- torch
- numpy
- pandas
- matplotlib
- seaborn
- Pillow

LEAF framework dependencies (in `leaf/requirements.txt`):
- tensorflow==1.13.1 (only for original LEAF models, not main simulations)
- numpy==1.16.4
- scipy
- matplotlib
- pandas

## File Organization

```
edgedrift/
├── decentralized_fl_sim.py          # Main simulator
├── leaf_datasets.py                  # Dataset loaders
├── model_variants.py                 # Model architectures
├── run_experiments.py                # Batch runner
├── run_model_scaling_experiments.py
├── run_scalability_experiments.py
├── leaf/                             # LEAF benchmark framework
│   ├── data/{dataset}/              # Dataset preprocessing
│   └── models/                       # Original TensorFlow models
├── results/                          # Experiment logs
├── scripts/
│   ├── data_extraction/             # Log parsing scripts
│   └── visualization/               # Figure generation
└── figures/                          # Generated visualizations
```

## Testing and Debugging

For quick testing with minimal runtime:
```bash
python decentralized_fl_sim.py \
    --dataset femnist \
    --num-nodes 4 \
    --rounds 2 \
    --local-epochs 1 \
    --max-samples 500 \
    --agg d-fedavg \
    --verbose
```

Use `--verbose` flag for detailed per-round statistics including:
- Individual node accuracies
- Compromised vs honest node performance
- Algorithm-specific metrics (acceptance rates, filtering stats)