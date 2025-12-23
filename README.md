# Murmura

A modular, config-driven framework for **Evidential Trust-Aware Decentralized Federated Learning** with Byzantine-resilient aggregation.

> **Paper**: *Evidential Trust-Aware Model Personalization in Decentralized Federated Learning for Wearable IoT*
> by Rangwala, Sinnott, Buyya - University of Melbourne

> **Quick Start**: See [QUICKSTART.md](QUICKSTART.md) to get running in 5 minutes!

## Key Contributions

- **Evidential Trust-Aware Aggregation**: Novel algorithm leveraging Dirichlet-based uncertainty decomposition (epistemic vs. aleatoric) to identify and filter Byzantine peers
- **Uncertainty-Driven Personalization**: Adaptive self-weighting based on local model confidence
- **BALANCE-Style Threshold Dynamics**: Progressive trust threshold tightening as models converge

## Features

- **Aggregation Algorithms**: FedAvg, Krum, BALANCE, Sketchguard, UBAR, **Evidential Trust**
- **Flexible Topologies**: Ring, fully-connected, Erdos-Renyi, k-regular
- **Byzantine Attack Simulation**: Gaussian noise, directed deviation
- **Wearable IoT Datasets**: UCI HAR, PAMAP2, PPG-DaLiA with natural user heterogeneity
- **Evidential Deep Learning**: Built-in Dirichlet-based models with uncertainty quantification
- **Config-Driven**: YAML/JSON configuration for reproducible experiments
- **CLI & Python API**: Run experiments from command line or programmatically

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/murmura.git
cd murmura

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install murmura in editable mode
uv pip install -e .

# Install with development dependencies
uv sync --group dev

# Install with wearable examples
uv pip install -e ".[examples]"
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### CLI Usage

Run an experiment from a config file:

```bash
# Run evidential trust experiment on UCI HAR dataset
murmura run experiments/paper/uci_har/evidential_trust.yaml

# Run with attacks (30% Byzantine nodes)
murmura run experiments/paper/attacks/uci_har/evidential_trust_directed_deviation_30pct.yaml

# Or with uv run
uv run murmura run murmura/examples/configs/basic_fedavg.yaml
```

List available components:

```bash
murmura list-components topologies
murmura list-components aggregators
murmura list-components attacks
```

### Python API Usage

```python
from murmura import Network, Config
from murmura.topology import create_topology
from murmura.aggregation import EvidentialTrustAggregator
from murmura.utils import set_seed, get_device

# Load configuration
config = Config.from_yaml("config.yaml")

# Create and run network
network = Network.from_config(
    config=config,
    model_factory=your_model_factory,
    dataset_adapter=your_dataset_adapter,
    aggregator_factory=your_aggregator_factory,
    device=get_device()
)

results = network.train(rounds=50, local_epochs=2, lr=0.01)
```

## Evidential Trust-Aware Aggregation

The core contribution of this framework is the **Evidential Trust-Aware Aggregator**, which leverages Dirichlet-based evidential deep learning to evaluate peer trustworthiness:

### Key Insight

Epistemic-aleatoric uncertainty decomposition directly indicates peer reliability:
- **High epistemic uncertainty (vacuity)** -> Insufficient learning, possibly Byzantine
- **High aleatoric uncertainty (entropy)** -> Inherent data ambiguity, still trustworthy

### Algorithm Overview

1. **Cross-Evaluation**: Evaluate each neighbor's model on local validation data
2. **Trust Scoring**: Compute trust from uncertainty profile: `trust = (1 - vacuity) * (w_a * accuracy + (1 - w_a))`
3. **Threshold Filtering**: Apply BALANCE-style tightening threshold to filter unreliable peers
4. **Weighted Aggregation**: Combine filtered neighbors with personalization self-weight

### Configuration

```yaml
aggregation:
  algorithm: evidential_trust
  params:
    vacuity_threshold: 0.5     # Threshold for vacuity penalty
    accuracy_weight: 0.7       # Weight of accuracy in trust score
    trust_threshold: 0.1       # Minimum trust to include peer
    self_weight: 0.6           # Weight for own model (personalization)
```

### Python Usage

```python
from murmura.aggregation import EvidentialTrustAggregator

aggregator = EvidentialTrustAggregator(
    vacuity_threshold=0.5,      # tau_u - epistemic uncertainty threshold
    accuracy_weight=0.7,         # w_a - accuracy contribution to trust
    trust_threshold=0.1,         # tau_min - minimum trust threshold
    self_weight=0.6,             # alpha - personalization weight
    use_adaptive_trust=True,     # EMA smoothing on trust scores
    use_tightening_threshold=True,  # BALANCE-style threshold dynamics
)
```

## Wearable IoT Datasets

Murmura includes built-in support for wearable sensor datasets with natural user heterogeneity:

### UCI HAR (Human Activity Recognition)
- **Source**: Smartphone accelerometer/gyroscope data
- **Nodes**: 10 subjects
- **Activities**: 6 classes (walking, sitting, standing, etc.)
- **Features**: 561 hand-crafted features

### PAMAP2 (Physical Activity Monitoring)
- **Source**: Body-worn IMU sensors
- **Nodes**: 9 subjects
- **Activities**: 12 classes (cycling, ironing, rope jumping, etc.)
- **Features**: 4000 (window-based from 3 IMUs)

### PPG-DaLiA (Heart Rate Monitoring)
- **Source**: Wrist-worn PPG, EDA, accelerometer, temperature
- **Nodes**: 15 subjects
- **Activities**: 7 classes (sitting, walking, cycling, driving, etc.)
- **Features**: 192 (window-based physiological signals)

### Configuration Example

```yaml
data:
  adapter: wearables.uci_har  # or wearables.pamap2, wearables.ppg_dalia
  params:
    data_path: wearables_datasets/UCI HAR Dataset
    partition_method: dirichlet
    alpha: 0.5  # Data heterogeneity (lower = more heterogeneous)

model:
  factory: examples.wearables.uci_har
  params:
    input_dim: 561
    hidden_dims: [256, 128]
    num_classes: 6
    dropout: 0.3
```

## Byzantine Attack Simulation

### Gaussian Noise Attack
Adds Gaussian noise to model parameters:

```yaml
attack:
  enabled: true
  type: gaussian
  percentage: 0.2  # 20% of nodes compromised
  params:
    noise_std: 10.0
```

### Directed Deviation Attack
Scales parameters with negative factor:

```yaml
attack:
  enabled: true
  type: directed_deviation
  percentage: 0.3  # 30% compromised
  params:
    lambda_param: -5.0
```

## Other Aggregation Algorithms

### FedAvg
Simple averaging of all neighbor models. Baseline for comparison.

```python
from murmura.aggregation import FedAvgAggregator
aggregator = FedAvgAggregator()
```

### Krum
Byzantine-resilient aggregation using distance-based selection.

```python
from murmura.aggregation import KrumAggregator
aggregator = KrumAggregator(num_compromised=2)
```

### BALANCE
Distance-based filtering with adaptive thresholds.

```python
from murmura.aggregation import BALANCEAggregator
aggregator = BALANCEAggregator(
    gamma=2.0,
    kappa=1.0,
    alpha=0.5,
    total_rounds=50
)
```

### Sketchguard
Count-Sketch compression for lightweight Byzantine-resilience.

```python
from murmura.aggregation import SketchguardAggregator
aggregator = SketchguardAggregator(
    model_dim=1000000,
    sketch_size=10000,
    gamma=2.0,
    total_rounds=50
)
```

### UBAR
Two-stage Byzantine-resilient (distance + loss evaluation).

```python
from murmura.aggregation import UBARAggregator
aggregator = UBARAggregator(
    rho=0.6,  # Expected fraction of honest neighbors
    alpha=0.5
)
```

## Network Topologies

Create different network topologies:

```python
from murmura.topology import create_topology

# Ring topology
topology = create_topology("ring", num_nodes=10)

# Fully connected
topology = create_topology("fully", num_nodes=10)

# Erdos-Renyi random graph
topology = create_topology("erdos", num_nodes=10, p=0.3)

# k-regular graph
topology = create_topology("k-regular", num_nodes=10, k=4)
```

## Running Paper Experiments

The `experiments/paper/` directory contains comprehensive experiment configurations:

```bash
# Generate all experiment configs (279 total)
python experiments/paper/generate_all_configs.py

# Run all experiments by category
python experiments/paper/run_comprehensive.py

# Run specific category
python experiments/paper/run_comprehensive.py --category attacks

# Run specific dataset within category
python experiments/paper/run_comprehensive.py --category attacks --dataset uci_har

# Generate summary reports
python experiments/paper/run_comprehensive.py --summary-only
```

### Experiment Categories

| Category | Description | Configs |
|----------|-------------|---------|
| **baseline** | No attacks, fully connected, alpha=0.5 | 18 |
| **heterogeneity** | Varying Dirichlet alpha in {0.1, 0.5, 1.0} | 54 |
| **attacks** | Byzantine attacks at 10%, 20%, 30% | 108 |
| **topologies** | Ring, fully, Erdos-Renyi, k-regular | 48 |
| **ablation** | Hyperparameter sensitivity study | 51 |

## Project Structure

```
murmura/
├── murmura/                      # Main package
│   ├── core/                     # Core components (Node, Network)
│   ├── topology/                 # Network topologies
│   ├── aggregation/              # Aggregation algorithms
│   │   ├── fedavg.py
│   │   ├── krum.py
│   │   ├── balance.py
│   │   ├── sketchguard.py
│   │   ├── ubar.py
│   │   └── evidential_trust.py   # Main contribution
│   ├── attacks/                  # Byzantine attacks
│   ├── data/                     # Data adapters & partitioners
│   ├── config/                   # Configuration system
│   ├── utils/                    # Utilities
│   ├── examples/
│   │   ├── configs/              # Example configs
│   │   ├── leaf/                 # LEAF benchmark integration
│   │   └── wearables/            # Wearable IoT datasets
│   │       ├── datasets.py       # UCI HAR, PAMAP2, PPG-DaLiA
│   │       ├── models.py         # Evidential deep learning models
│   │       └── adapter.py        # Dataset adapters
│   └── cli.py                    # CLI interface
├── experiments/
│   └── paper/                    # Paper experiment configs
│       ├── generate_all_configs.py
│       ├── run_comprehensive.py
│       ├── uci_har/
│       ├── pamap2/
│       ├── ppg_dalia/
│       ├── attacks/
│       ├── heterogeneity/
│       ├── topologies/
│       └── ablation/
├── wearables_datasets/           # Downloaded datasets
└── pyproject.toml
```

## Development

### Install development dependencies

```bash
uv sync --group dev
```

### Run tests

```bash
pytest tests/
```

### Format code

```bash
black murmura/
isort murmura/
```

### Type checking

```bash
mypy murmura/
```

## Citation

If you use Murmura in your research, please cite:

```bibtex
@misc{rangwala2025evidentialtrustawaremodelpersonalization,
      title={Evidential Trust-Aware Model Personalization in Decentralized Federated Learning for Wearable IoT}, 
      author={Murtaza Rangwala and Richard O. Sinnott and Rajkumar Buyya},
      year={2025},
      eprint={2512.19131},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2512.19131}, 
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- LEAF benchmark framework: https://leaf.cmu.edu
- UCI HAR Dataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
- PAMAP2 Dataset: https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring
- PPG-DaLiA Dataset: https://ubicomp.eti.uni-siegen.de/home/datasets/sensors19/
