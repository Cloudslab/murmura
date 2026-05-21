# Murmura

A modular, config-driven framework for **decentralized federated learning** with Byzantine-resilient aggregation.

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

## Overview

Murmura lets you define a complete decentralized FL experiment — topology, aggregation algorithm, attack scenario, and execution backend — in a single YAML file and run it with one command. Swap any component without touching the training loop.

```bash
murmura run experiments/my_experiment.yaml
```

## Features

- **Multiple aggregation algorithms**: FedAvg, Krum, BALANCE, Sketchguard, UBAR, Evidential Trust
- **Flexible topologies**: Ring, fully-connected, Erdős-Rényi, k-regular
- **Byzantine attack simulation**: Gaussian noise, directed deviation, topology-liar
- **Two execution backends**: Fast in-process simulation or real distributed ZMQ processes
- **Dynamic topology support**: Deterministic random-walk mobility model for time-varying graphs
- **Config-driven**: YAML/JSON experiments with Pydantic validation — fully reproducible
- **Extensible**: Add a new aggregator, topology, or attack by subclassing one base class
- **CLI & Python API**: Run from the command line or integrate programmatically

## Installation

### Using uv (Recommended)

```bash
git clone https://github.com/Cloudslab/murmura.git
cd murmura

uv venv && source .venv/bin/activate
uv pip install -e .

# With development dependencies
uv sync --group dev

# With bundled dataset examples
uv pip install -e ".[examples]"
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### CLI

```bash
# Run an experiment from a config file
murmura run murmura/examples/configs/basic_fedavg.yaml

# List available components
murmura list-components aggregators
murmura list-components topologies
murmura list-components attacks
```

### Example Config

```yaml
experiment:
  name: "my-experiment"
  rounds: 50

topology:
  type: "k-regular"
  num_nodes: 20
  k: 4

aggregation:
  algorithm: "krum"

attack:
  enabled: true
  type: "gaussian"
  percentage: 0.2
  params:
    noise_std: 1.0

training:
  local_epochs: 2
  batch_size: 32
  lr: 0.01

backend: simulation
```

### Python API

```python
from murmura.config import load_config
from murmura.core import Network

config = load_config("my_experiment.yaml")

network = Network.from_config(
    config=config,
    model_factory=your_model_factory,
    dataset_adapter=your_dataset_adapter,
    aggregator_factory=your_aggregator_factory,
)

results = network.train()
```

## Execution Backends

### Simulation (default)

All nodes run in a single Python process with shared memory. Fast for development and hyperparameter search.

```yaml
backend: simulation
```

### Distributed

Each node runs as an independent OS process communicating over ZeroMQ. Round synchronisation uses a shared wall-clock epoch — no central coordinator. A passive monitor collects metrics without influencing training.

```yaml
backend: distributed

distributed:
  transport: ipc       # or tcp for multi-machine
  round_duration_s: 60.0
  startup_grace_s: 5.0
```

Switch from simulation to distributed by changing one line in your config. For multi-machine runs:

```bash
# On each worker node
murmura run-node config.yaml --node-id 1 --coordinator-host 192.168.1.1
```

## Aggregation Algorithms

All aggregators share the same interface and are interchangeable via config:

| Algorithm | Config value | Description |
|---|---|---|
| FedAvg | `fedavg` | Simple averaging — baseline |
| Krum | `krum` | Distance-based Byzantine filtering |
| BALANCE | `balance` | Adaptive distance threshold |
| Sketchguard | `sketchguard` | Count-Sketch compression |
| UBAR | `ubar` | Two-stage: distance filter + loss evaluation |
| Evidential Trust | `evidential_trust` | Uncertainty-aware peer scoring |

### Adding a Custom Aggregator

```python
from murmura.aggregation.base import Aggregator

class MyAggregator(Aggregator):
    def aggregate(self, node_id, own_state, neighbor_states, round_num, **kwargs):
        # your logic here
        return aggregated_state

    def get_statistics(self):
        return {}
```

Then set `algorithm: my_aggregator` in your config and register it in the CLI factory.

## Network Topologies

```python
from murmura.topology import create_topology

topology = create_topology("ring",     num_nodes=10)
topology = create_topology("fully",    num_nodes=10)
topology = create_topology("erdos",    num_nodes=10, p=0.3)
topology = create_topology("k-regular", num_nodes=10, k=4)
```

### Dynamic Topology (Mobility Model)

Add a `mobility` block to use a time-varying graph G^t. Each node independently reconstructs the same graph from a shared seed — no inter-process communication required.

```yaml
mobility:
  area_size: 100.0
  comm_range: 40.0
  max_speed: 8.0
  seed: 42
  ensure_connected: true
```

## Byzantine Attacks

```yaml
# Gaussian noise
attack:
  enabled: true
  type: gaussian
  percentage: 0.3
  params:
    noise_std: 1.0

# Directed deviation
attack:
  enabled: true
  type: directed_deviation
  percentage: 0.3
  params:
    lambda_param: -5.0

# Topology liar (DMTT experiments)
attack:
  enabled: true
  type: topology_liar
  percentage: 0.3
  params:
    model_attack_type: gaussian
    noise_std: 10.0
```

## Project Structure

```
murmura/
├── murmura/
│   ├── core/               # Network orchestrator and Node
│   ├── aggregation/        # FedAvg, Krum, BALANCE, Sketchguard, UBAR, EvidentialTrust
│   ├── topology/           # Topology generators + MobilityModel
│   ├── attacks/            # Gaussian, DirectedDeviation, TopologyLiar
│   ├── distributed/        # ZMQ backend (NodeProcess, Monitor, Runner)
│   ├── dmtt/               # DMTT trust protocol (DMTTNodeState, DMTTNodeProcess)
│   ├── config/             # Pydantic schema + YAML/JSON loader
│   ├── utils/              # Factories, device helpers
│   ├── examples/
│   │   ├── configs/        # Example YAML configs
│   │   └── leaf/           # LEAF benchmark integration
│   └── cli.py
├── experiments/
│   └── paper/
│       └── dmtt/           # Three-condition DMTT experiment configs
└── pyproject.toml
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
pytest tests/

# Format
black murmura/ && isort murmura/

# Type check
mypy murmura/

# Lint
ruff murmura/
```

## Citation

If you use Murmura in your research, please cite:

```bibtex
@INPROCEEDINGS{rangwala2026murmura,
  author={Rangwala, Murtaza and Sinnott, Richard O and Buyya, Rajkumar},
  booktitle={2026 IEEE 26th International Symposium on Cluster, Cloud and Internet Computing (CCGrid)}, 
  title={Evidential Trust-Aware Model Personalization in Decentralized Federated Learning for Wearable IoT},
  year={2026},
  publisher = {IEEE Press},
  address = {New Jersey, USA},
  pages = {510-519},
  doi={10.1109/CCGrid68966.2026.00061}}
```

## License

GPL-3.0 — see [LICENSE](LICENSE) for details.
