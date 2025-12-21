# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Murmura is a modular, config-driven framework for decentralized federated learning with Byzantine-resilient aggregation. It supports multiple aggregation algorithms (FedAvg, Krum, BALANCE, Sketchguard, UBAR), flexible network topologies (ring, fully-connected, Erdős-Rényi, k-regular), and Byzantine attack simulation.

## Common Commands

### Installation & Setup
```bash
# Using uv (recommended - extremely fast)
uv venv
source .venv/bin/activate
uv pip install -e .

# Install development dependencies
uv sync --group dev

# Install with LEAF benchmark examples
uv pip install -e ".[examples]"
```

### Running Experiments
```bash
# Run experiment from config file
murmura run murmura/examples/configs/basic_fedavg.yaml

# Or with uv run (without activating venv)
uv run murmura run murmura/examples/configs/basic_fedavg.yaml

# List available components
murmura list-components topologies
murmura list-components aggregators
murmura list-components attacks

# Run example programmatically
python murmura/examples/simple_programmatic.py
```

### Development & Testing
```bash
# Run tests (when test suite exists)
pytest tests/
uv run pytest tests/

# Format code
black murmura/
isort murmura/

# Type checking
mypy murmura/

# Lint with ruff
ruff murmura/
```

### LEAF Dataset Preparation
```bash
# For FEMNIST benchmark
cd leaf/data/femnist
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample
cd ../../..
```

## Architecture & Design

### Core Components

**Network Orchestrator** (`murmura/core/network.py`)
- `Network` class orchestrates decentralized federated learning across nodes
- Manages topology, nodes, and optional Byzantine attacks
- Three-step training loop per round:
  1. Local training on all honest nodes
  2. Model exchange and decentralized aggregation
  3. Evaluation and metrics collection
- Tracks history: mean/std accuracy, honest vs compromised node performance
- Factory method `Network.from_config()` builds network from YAML/JSON config

**Node** (`murmura/core/node.py`)
- Represents individual participant in decentralized network
- Each node owns: model, train/test dataloaders, aggregator instance
- Key methods:
  - `local_train()`: Local SGD training for specified epochs
  - `evaluate()`: Evaluate on test data
  - `aggregate_with_neighbors()`: Apply aggregation algorithm with neighbor states
  - `get_state()` / `set_state()`: Model state management
- Delegates aggregation logic to pluggable `Aggregator` instance

### Aggregation Algorithms

All aggregators inherit from `Aggregator` base class (`murmura/aggregation/base.py`) and implement:
- `aggregate(node_id, own_state, neighbor_states, round_num, **kwargs)` → aggregated state
- `get_statistics()` → monitoring metrics

**Five Built-in Aggregators:**
1. **FedAvg**: Simple averaging (baseline)
2. **Krum**: Multi-Krum distance-based Byzantine filtering
3. **BALANCE**: Distance-based filtering with adaptive thresholds
4. **Sketchguard**: Count-Sketch compression for lightweight filtering
5. **UBAR**: Two-stage (distance + loss evaluation) Byzantine-resilient

**UBAR Architecture** (`murmura/aggregation/ubar.py`):
- Stage 1: Filter neighbors by L2 distance (keep top ρ fraction)
- Stage 2: Evaluate shortlisted models on training sample, select best performer
- Requires `train_loader` and `model_template` passed via kwargs
- Tracks acceptance rates, distances, losses per neighbor

### Configuration System

**Schema** (`murmura/config/schema.py`):
- Pydantic-based validation with strict typing
- Seven config sections: ExperimentConfig, TopologyConfig, AggregationConfig, AttackConfig, TrainingConfig, DataConfig, ModelConfig
- Main `Config` class validates entire experiment configuration

**Loading** (`murmura/config/loader.py`):
- `load_config(path)` supports YAML and JSON
- Returns validated `Config` instance

### Topology System

**Generators** (`murmura/topology/generators.py`):
- `create_topology(type, num_nodes, **params)` factory function
- Four topologies: `ring`, `fully`, `erdos` (random graph), `k-regular`
- Returns `Topology` object with adjacency list (`neighbors` dict)

### Byzantine Attacks

**Two Attack Types** (`murmura/attacks/`):
1. **GaussianAttack**: Adds Gaussian noise to model parameters
2. **DirectedDeviationAttack**: Scales parameters with directional bias (λ parameter)

Both select compromised nodes randomly based on `attack_percentage`.

### Data Adapters

**Base Pattern** (`murmura/data/base.py`):
- `DatasetAdapter` wraps PyTorch Dataset with federated partitions
- `get_client_data(client_id)` returns Subset for that client

**LEAF Integration** (`murmura/examples/leaf/`):
- Built-in support for LEAF benchmark datasets (FEMNIST, CelebA)
- `load_leaf_adapter()` helper loads partitioned LEAF data
- Example models provided in `murmura/examples/leaf/models.py`

### CLI System

**CLI** (`murmura/cli.py`):
- Typer-based CLI with Rich formatting
- `run` command: loads config, creates network, trains, displays results
- `list-components` command: shows available topologies/aggregators/attacks
- Dynamically imports dataset adapters and model factories from config
- Creates aggregator factory based on algorithm type

## Key Design Patterns

**Factory Pattern**:
- Model creation via `model_factory: Callable[[], nn.Module]`
- Aggregator creation via `aggregator_factory: Callable[[int], Aggregator]`
- Enables per-node customization while maintaining clean interfaces

**Adapter Pattern**:
- `DatasetAdapter` wraps any PyTorch Dataset with federated partitioning
- Decouples data loading from network/node logic

**Strategy Pattern**:
- `Aggregator` base class allows pluggable aggregation algorithms
- `Attack` base class allows pluggable Byzantine behaviors

**Config-Driven Architecture**:
- Entire experiment reproducible from single YAML/JSON file
- Pydantic validation ensures correctness before execution

## Code Style

- **Line length**: 100 characters (configured in pyproject.toml)
- **Formatting**: Black + isort with profile="black"
- **Type hints**: Encouraged but not strictly enforced (`disallow_untyped_defs = false`)
- **Target Python**: 3.8+ compatibility

## Important Constraints

1. **State Management**: Model states are always CPU tensors (`get_model_state()` clones to CPU)
2. **Aggregation Context**: UBAR and potentially other aggregators need `train_loader`, `model_template`, `device` passed via kwargs
3. **Topology Integrity**: Number of nodes must match topology.num_nodes
4. **Attack Timing**: Attacks applied AFTER local training, BEFORE aggregation step
5. **Device Handling**: Models moved to device during Node initialization and training

## Extension Points

**Adding New Aggregator**:
1. Inherit from `murmura.aggregation.base.Aggregator`
2. Implement `aggregate()` method
3. Add to CLI factory in `_create_aggregator_factory()`
4. Update schema `AggregationConfig.algorithm` Literal type

**Adding New Topology**:
1. Add generator function in `murmura/topology/generators.py`
2. Update `create_topology()` dispatch
3. Update schema `TopologyConfig.type` Literal type

**Adding New Attack**:
1. Inherit from `murmura.attacks.base.Attack`
2. Implement `is_compromised()` and `apply_attack()`
3. Add to `Network.from_config()` attack factory
4. Update schema `AttackConfig.type` Literal type

**Adding New Dataset**:
1. Create adapter implementing `get_client_data(client_id)` interface
2. Either use LEAF pattern (`leaf.dataset_name`) or custom import path
3. CLI dynamically imports based on config `data.adapter` field
