# Murmura

A modular, config-driven framework for decentralized federated learning with Byzantine-resilient aggregation.

> **Quick Start**: See [QUICKSTART.md](QUICKSTART.md) to get running in 5 minutes!

## Features

- **Multiple Aggregation Algorithms**: FedAvg, Krum, BALANCE, Sketchguard, UBAR
- **Flexible Topologies**: Ring, fully-connected, Erdős-Rényi, k-regular
- **Byzantine Attack Simulation**: Gaussian noise, directed deviation
- **Config-Driven**: YAML/JSON configuration for reproducible experiments
- **Modular Design**: Easy to extend with custom datasets, models, and aggregators
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

# Install with example dependencies (for LEAF)
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
murmura run murmura/examples/configs/basic_fedavg.yaml

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
from murmura.aggregation import UBARAggregator
from murmura.utils import set_seed, get_device

# Load configuration
config = Config.from_yaml("config.yaml")

# Or create programmatically
from murmura.config import ExperimentConfig, TopologyConfig, AggregationConfig

config = Config(
    experiment=ExperimentConfig(name="my-experiment", rounds=20),
    topology=TopologyConfig(type="ring", num_nodes=10),
    aggregation=AggregationConfig(algorithm="ubar", params={"rho": 0.6}),
    # ... other configs
)

# Create and run network
network = Network.from_config(
    config=config,
    model_factory=your_model_factory,
    dataset_adapter=your_dataset_adapter,
    aggregator_factory=your_aggregator_factory,
    device=get_device()
)

results = network.train(rounds=20, local_epochs=3, lr=0.001)
```

## Configuration

Example YAML configuration:

```yaml
experiment:
  name: "ubar-experiment"
  seed: 42
  rounds: 20
  verbose: true

topology:
  type: "ring"  # ring, fully, erdos, k-regular
  num_nodes: 20

aggregation:
  algorithm: "ubar"  # fedavg, krum, balance, sketchguard, ubar
  params:
    rho: 0.6
    alpha: 0.5

attack:
  enabled: true
  type: "gaussian"  # gaussian, directed_deviation
  percentage: 0.2
  params:
    noise_std: 10.0

training:
  local_epochs: 3
  batch_size: 64
  lr: 0.001

data:
  adapter: "leaf.femnist"
  params:
    data_path: "leaf/data/femnist/data"

model:
  factory: "examples.leaf.models.LEAFFEMNISTModel"
  params:
    num_classes: 62
```

## Aggregation Algorithms

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
    total_rounds=20
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
    total_rounds=20
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

# Erdős-Rényi random graph
topology = create_topology("erdos", num_nodes=10, p=0.3)

# k-regular graph
topology = create_topology("k-regular", num_nodes=10, k=4)
```

## Custom Datasets

Integrate your own dataset using the adapter pattern:

```python
from torch.utils.data import Dataset
from murmura.data import DatasetAdapter

# Your PyTorch dataset
my_dataset = MyPyTorchDataset()

# Define client partitions (list of indices for each client)
client_partitions = [
    [0, 1, 2, 3],      # Client 0 gets samples 0-3
    [4, 5, 6, 7],      # Client 1 gets samples 4-7
    # ...
]

# Wrap in adapter
dataset_adapter = DatasetAdapter(
    dataset=my_dataset,
    client_partitions=client_partitions
)
```

## Custom Models

Use any PyTorch model:

```python
import torch.nn as nn

def my_model_factory():
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
```

## LEAF Benchmark Integration

Murmura includes built-in support for LEAF benchmark datasets:

```yaml
data:
  adapter: "leaf.femnist"  # or "leaf.celeba"
  params:
    data_path: "leaf/data/femnist/data"

model:
  factory: "examples.leaf.models.LEAFFEMNISTModel"
  params:
    num_classes: 62
```

## Project Structure

```
murmura/
├── murmura/                  # Main package
│   ├── core/                 # Core components (Node, Network)
│   ├── topology/             # Network topologies
│   ├── aggregation/          # Aggregation algorithms
│   ├── attacks/              # Byzantine attacks
│   ├── data/                 # Data adapters
│   ├── config/               # Configuration system
│   ├── utils/                # Utilities
│   ├── examples/             # Usage examples
│   │   ├── configs/          # Example configs
│   │   └── leaf/             # LEAF benchmark integration
│   └── cli.py                # CLI interface
├── tests/                    # Unit tests
└── pyproject.toml            # Project configuration
```

## Development

### Install development dependencies

```bash
# Using dependency groups (recommended)
uv sync --group dev

# Or using optional dependencies
uv pip install -e ".[dev]"
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

### Using uv run (directly run commands without activating venv)

```bash
uv run pytest tests/
uv run black murmura/
uv run murmura run config.yaml
```

## Citation

If you use Murmura in your research, please cite:

```bibtex
@software{murmura2024,
  title={Murmura: A Modular Framework for Decentralized Federated Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/murmura}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- LEAF benchmark framework: https://leaf.cmu.edu
- BALANCE algorithm: [citation]
- UBAR algorithm: [citation]
- Sketchguard algorithm: [citation]
