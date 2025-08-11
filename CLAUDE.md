# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Murmura is a Ray-based framework for federated and decentralized machine learning with advanced privacy guarantees and trust monitoring capabilities. This is a PhD research project focused on privacy-preserving machine learning.

## Development Commands

### Setup and Installation
```bash
# Install dependencies using Poetry
poetry install

# Activate virtual environment
poetry shell
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=murmura tests/

# Run excluding integration tests (faster, no external dependencies)
pytest -m "not integration"

# Run a specific test file
pytest tests/test_specific_file.py

# Run a specific test function
pytest tests/test_file.py::test_function_name
```

### Code Quality
```bash
# Linting - check code style issues
ruff check

# Format code automatically
ruff format

# Type checking
mypy murmura/
```

### Running Examples
```bash
# Basic federated learning with MNIST
python murmura/examples/dp_mnist_example.py

# Decentralized learning with trust monitoring
python murmura/examples/dp_decentralized_mnist_example.py

# CIFAR-10 federated learning
python murmura/examples/dp_cifar10_example.py

# Medical imaging (HAM10000)
python murmura/examples/dp_ham10000_example.py
```

### Trust Monitoring Experiments
```bash
# Run comprehensive trust evaluation
cd scripts/
./test_enhanced_trust_evaluation.sh

# Analyze results with visualization
python analyze_enhanced_results.py enhanced_trust_results_TIMESTAMP --create-plots
```

## Architecture Overview

### Core Components

**murmura/orchestration/** - Central coordination layer
- `FederatedLearningProcess` - Manages centralized federated learning with server-client architecture
- `DecentralizedLearningProcess` - Coordinates peer-to-peer decentralized learning
- `ClusterManager` - Ray-based distributed computing with multi-node support

**murmura/aggregation/** - Aggregation strategies
- `FedAvg` - Standard federated averaging
- `TrimmedMean` - Byzantine-robust aggregation for adversarial environments
- `GossipAvg` - Decentralized peer-to-peer averaging
- All strategies support differential privacy variants

**murmura/trust_monitoring/** - EdgeDrift trust monitoring system
- CUSUM-based anomaly detection for gradient and label attacks
- Trust-weighted aggregation with polynomial decay
- Multi-layer detection: parameter-space, loss-space, consensus-based
- Dynamic trust scoring and malicious node isolation

**murmura/privacy/** - Differential privacy implementation
- Opacus integration for client-level DP
- RDP privacy accounting with automatic noise calibration
- Privacy budget tracking across training rounds

**murmura/network_management/** - Network topology management
- Supports star, ring, complete, line, and custom topologies
- Automatic topology validation for decentralized scenarios
- Dynamic neighbor discovery and connection management

**murmura/node/** - Client implementation
- `ClientActor` - Ray actor for distributed training nodes
- Handles local training, gradient computation, and communication
- Supports both honest and malicious behavior simulation

**murmura/attacks/** - Attack simulation
- Gradient manipulation attacks with configurable intensity
- Label flipping attacks for data poisoning
- Progressive attack intensity over rounds

**murmura/data_processing/** - Data handling
- Unified interface for HuggingFace and PyTorch datasets
- IID and non-IID partitioning (Dirichlet, quantity-based)
- Automatic data distribution across clients

## Key Design Patterns

1. **Actor-Based Distribution**: Uses Ray actors for distributed computing, enabling scalable multi-node deployments

2. **Configuration-Driven**: Extensive use of Pydantic models for configuration validation and management

3. **Strategy Pattern**: Aggregation strategies are pluggable and interchangeable

4. **Event-Driven Monitoring**: Trust monitoring uses event-based architecture for real-time detection

5. **Privacy by Design**: Differential privacy is integrated at the core, not bolted on

## Working with Trust Monitoring

The EdgeDrift trust monitoring system is central to the defensive capabilities:

1. Trust scores range from 0.0 (malicious) to 1.0 (trusted)
2. Uses CUSUM for change detection with adaptive thresholds
3. Polynomial decay for trust score updates provides smooth transitions
4. Consensus validation across multiple detection layers

When modifying trust monitoring:
- Check `murmura/trust_monitoring/trust_monitor.py` for core logic
- Test with `scripts/test_enhanced_trust_evaluation.sh`
- Analyze results using `scripts/analyze_enhanced_results.py`

## Common Workflows

### Adding a New Dataset
1. Create dataset class in `murmura/data_processing/`
2. Implement `BaseDataset` interface
3. Add partitioning logic for federated distribution
4. Update examples to demonstrate usage

### Implementing a New Attack
1. Add attack class in `murmura/attacks/`
2. Implement gradient or data manipulation logic
3. Update `ClientActor` to support the attack
4. Test with trust monitoring enabled

### Creating a New Aggregation Strategy
1. Create strategy in `murmura/aggregation/`
2. Inherit from base aggregation class
3. Implement aggregation logic with optional DP support
4. Add configuration in `AggregationConfig`

## Important Notes

- Always run `ruff format` before committing code
- Run `pytest -m "not integration"` for quick local testing
- Integration tests require internet for HuggingFace datasets
- Use `poetry add` for new dependencies to maintain lock file
- Trust monitoring adds ~5-10% overhead to training time
- Ray dashboard available at http://localhost:8265 during training