# Murmura
[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Cloudslab/murmura/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/Cloudslab/murmura/blob/python-coverage-comment-action-data/htmlcov/index.html)
[![DOI](https://zenodo.org/badge/923861956.svg)](https://doi.org/10.5281/zenodo.15622123)

**Murmura** is a comprehensive Ray-based framework for federated and decentralized machine learning. Built for researchers and developers, it provides production-ready tools for distributed machine learning with advanced privacy guarantees and flexible network topologies.

## 🌐 What is Murmura?

Murmura is a sophisticated federated learning framework that supports both centralized and fully decentralized learning environments. Built on Ray for distributed computing, it enables researchers to experiment with various network topologies, aggregation strategies, and privacy-preserving techniques across single-node and multi-node clusters.

## 🧩 Key Features

### Core Framework
- 🏗️ **Ray-Based Distributed Computing**  
  Multi-node cluster support with automatic actor lifecycle management and resource optimization

- 🔄 **Flexible Learning Paradigms**  
  Both centralized federated learning and fully decentralized peer-to-peer learning

- 🌐 **Multiple Network Topologies**  
  Star, ring, complete graph, line, and custom topologies with automatic compatibility validation

- ⚡ **Intelligent Resource Management**  
  Automatic eager/lazy dataset loading, CPU/GPU allocation, and placement strategies

### Privacy & Security
- 🔐 **Comprehensive Differential Privacy**  
  Client-level DP with Opacus integration, RDP privacy accounting, and automatic noise calibration

- 🛡️ **Byzantine-Robust Aggregation**  
  Trimmed mean and secure aggregation strategies for adversarial environments

- 📊 **Privacy Budget Tracking**  
  Real-time privacy budget monitoring across clients and training rounds

### Data & Models
- 📦 **Unified Dataset Interface**  
  Seamless integration with HuggingFace datasets, PyTorch datasets, and custom data

- 🎯 **Flexible Data Partitioning**  
  IID and non-IID data distribution with Dirichlet and quantity-based partitioning

- 🤖 **PyTorch Model Integration**  
  Easy integration with existing PyTorch models and automatic DP adaptation

### Trust Monitoring (EdgeDrift)
- 🛡️ **Adaptive Trust Monitoring**  
  CUSUM-based detection for gradient manipulation and label flipping attacks

- 📊 **Trust-Weighted Aggregation**  
  Dynamic trust scoring with polynomial decay and consensus-based validation

- 🔍 **Multi-Layer Detection**  
  Parameter-space, loss-space, and consensus-based anomaly detection

### Monitoring & Visualization
- 📈 **Real-Time Training Visualization**  
  Network topology visualization, training progress tracking, and metrics export

- 🔍 **Comprehensive Monitoring**  
  Actor health checks, resource usage tracking, and event-driven architecture

## 🚀 Quick Start

### Installation

```bash
# Install with Poetry
poetry install

# Or with pip
pip install murmura
```

### Basic Usage

```python
from murmura.orchestration.learning_process import FederatedLearningProcess
from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.aggregation.aggregation_config import AggregationConfig

# Configure federated learning
config = OrchestrationConfig(
    num_clients=10,
    num_rounds=50,
    topology_type="star",
    aggregation_config=AggregationConfig(strategy="fedavg")
)

# Run federated learning
process = FederatedLearningProcess(config)
results = process.run()
```

### Examples

Explore complete examples in the `murmura/examples/` directory:

- **`dp_mnist_example.py`** - Basic federated learning with MNIST and DP
- **`dp_decentralized_mnist_example.py`** - Decentralized learning with trust monitoring
- **`dp_cifar10_example.py`** - Federated learning with CIFAR-10
- **`dp_ham10000_example.py`** - Medical imaging federated learning

### Trust Monitoring Experiments

Use the enhanced evaluation scripts in the `scripts/` directory:

- **`test_enhanced_trust_evaluation.sh`** - Comprehensive trust monitoring evaluation
- **`analyze_enhanced_results.py`** - Analysis and visualization of results

```bash
# Run trust monitoring experiments
cd scripts/
./test_enhanced_trust_evaluation.sh

# Analyze results
python analyze_enhanced_results.py enhanced_trust_results_TIMESTAMP --create-plots
```

## 🏗️ Architecture

### Core Components

- **Learning Processes** - `FederatedLearningProcess` and `DecentralizedLearningProcess` for different learning paradigms
- **Cluster Manager** - Ray-based distributed computing with multi-node support
- **Aggregation Strategies** - FedAvg, TrimmedMean, GossipAvg with DP variants
- **Network Topologies** - Flexible network structures for decentralized learning
- **Privacy Framework** - Comprehensive differential privacy with Opacus integration

## 📊 Supported Aggregation Strategies

| Strategy       | Type          | Privacy-Enabled | Best For                    |
|----------------|---------------|-----------------|------------------------------|
| **FedAvg**     | Centralized   | ✅              | Standard federated learning  |
| **TrimmedMean**| Centralized   | ✅              | Adversarial environments     |
| **GossipAvg**  | Decentralized | ✅              | Peer-to-peer networks        |

## 🌐 Network Topologies

- **Star** - Central server with spoke clients (federated learning)
- **Ring** - Circular peer-to-peer communication
- **Complete** - Full mesh networking (all-to-all)
- **Line** - Sequential peer communication
- **Custom** - User-defined adjacency matrices

## 🛠️ Development

### Setup

```bash
poetry install
poetry shell
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=murmura tests/

# Run excluding integration tests
pytest -m "not integration"
```

### Code Quality

```bash
ruff check          # Linting
ruff format         # Code formatting
mypy murmura/       # Type checking
```

## 🔮 Future Roadmap

- **Enhanced Privacy Techniques** - Homomorphic encryption and secure multi-party computation
- **Advanced Network Simulation** - Realistic network conditions and fault injection
- **AI Agent Integration** - Autonomous learning agents for dynamic environments
- **Real-world Deployment Tools** - Production deployment and monitoring capabilities

## 🤝 Contributing

We'd love your help building Murmura.  
Start by checking out the [issues](https://github.com/murtazahr/murmura/issues) or submitting a [pull request](https://github.com/murtazahr/murmura/pulls).

## 📄 License

Licensed under the **GNU GPLv3**. See [LICENSE](LICENSE) for more details.

## 📬 Contact

For questions or feedback, [open an issue](https://github.com/murtazahr/murmura/issues) or email [mrangwala@student.unimelb.edu.au](mailto:mrangwala@student.unimelb.edu.au).

## 📰 Stay Updated

Subscribe to our newsletter to receive updates on Murmura's development and be the first to know about new features and releases. Visit our [website](https://murmura-landing-page.vercel.app/) for more information.
