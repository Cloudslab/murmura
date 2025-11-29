# Murmura Quick Start

Get up and running with Murmura in under 5 minutes!

## 1. Install uv

```bash
# Install uv (extremely fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Set Up Project

```bash
# Clone (or navigate to) the repository
cd murmura

# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install murmura
uv pip install -e .
```

## 3. Run Your First Experiment

### Option A: Simple Programmatic Example (No LEAF data needed)

```bash
python murmura/examples/simple_programmatic.py
```

This runs a simple decentralized learning experiment with synthetic data.

### Option B: Config-Based Experiment

Create a config file `my_experiment.yaml`:

```yaml
experiment:
  name: "my-first-experiment"
  seed: 42
  rounds: 10
  verbose: true

topology:
  type: "ring"
  num_nodes: 5

aggregation:
  algorithm: "fedavg"

attack:
  enabled: false

training:
  local_epochs: 2
  batch_size: 16
  lr: 0.01

data:
  adapter: "custom"  # You'll need to provide your dataset

model:
  factory: "custom"  # You'll need to provide your model
```

Then run:
```bash
murmura run my_experiment.yaml
```

## 4. Explore Available Components

```bash
# List available network topologies
murmura list-components topologies

# List available aggregation algorithms
murmura list-components aggregators

# List available attack types
murmura list-components attacks
```

## 5. Try Different Configurations

### Ring Topology with FedAvg

```yaml
topology:
  type: "ring"
  num_nodes: 10

aggregation:
  algorithm: "fedavg"
```

### Erdős-Rényi with UBAR

```yaml
topology:
  type: "erdos"
  num_nodes: 20
  p: 0.3

aggregation:
  algorithm: "ubar"
  params:
    rho: 0.6
    alpha: 0.5
```

### Byzantine Attack Simulation

```yaml
attack:
  enabled: true
  type: "gaussian"
  percentage: 0.2
  params:
    noise_std: 10.0
```

## 6. Using with LEAF Datasets (Optional)

If you want to use LEAF benchmark datasets:

```bash
# Install example dependencies
uv pip install -e ".[examples]"

# Prepare LEAF data (e.g., FEMNIST)
cd leaf/data/femnist
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample
cd ../../..

# Run with LEAF config
murmura run murmura/examples/configs/basic_fedavg.yaml

# Or with uv run
uv run murmura run murmura/examples/configs/basic_fedavg.yaml
```

## 7. Python API Usage

```python
from murmura import Network
from murmura.topology import create_topology
from murmura.aggregation import FedAvgAggregator
from murmura.core import Node
from murmura.utils import get_device

# Create simple network programmatically
topology = create_topology("ring", num_nodes=5)
# ... create nodes with your data/models ...
network = Network(nodes=nodes, topology=topology)
results = network.train(rounds=10, local_epochs=2, lr=0.01)
```

## Tips

- **Use `uv run`** to run commands without activating venv:
  ```bash
  uv run murmura run config.yaml
  uv run python murmura/examples/simple_programmatic.py
  ```

- **Install dev dependencies** for development:
  ```bash
  uv sync --group dev
  ```

- **Check logs** in verbose mode to understand what's happening:
  ```yaml
  experiment:
    verbose: true
  ```

- **Start simple** with FedAvg and no attacks, then add complexity

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [examples/](murmura/examples/) for more examples
- See [MIGRATION.md](MIGRATION.md) if migrating from the old script
- Explore different aggregation algorithms and topologies

## Getting Help

- Check example configs in `murmura/examples/configs/`
- Run `murmura list-components <type>` for available options
- Look at inline code documentation
- See `murmura/examples/simple_programmatic.py` for Python API usage
