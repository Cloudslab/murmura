# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies
poetry install
poetry shell

# Or with pip
pip install murmura
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=murmura tests/

# Run excluding integration tests (faster)
pytest -m "not integration"

# Run single test file
pytest tests/path/to/test_file.py

# Run single test
pytest tests/path/to/test_file.py::test_function_name
```

### Code Quality
```bash
# Linting
ruff check

# Code formatting
ruff format  

# Type checking
mypy murmura/
```

## Architecture Overview

Murmura is a Ray-based federated learning framework supporting both centralized and decentralized learning with differential privacy. The architecture follows a modular design:

### Core Components

**Learning Processes** (`murmura/orchestration/learning_process/`)
- `LearningProcess`: Abstract base class defining common interface
- `FederatedLearningProcess`: Centralized federated learning with star topology
- `DecentralizedLearningProcess`: Peer-to-peer learning with various topologies
- Both processes use `OrchestrationConfig` for unified configuration

**Cluster Management** (`murmura/orchestration/`)
- `ClusterManager`: Ray-based distributed computing with multi-node support
- `TopologyCoordinator`: Manages network topologies for decentralized learning
- Supports placement groups, resource allocation, and actor lifecycle management

**Aggregation Strategies** (`murmura/aggregation/`)
- `FedAvg`: Standard federated averaging
- `TrimmedMean`: Byzantine-robust aggregation
- `GossipAvg`: Decentralized peer-to-peer averaging
- All strategies support differential privacy variants

**Network Topologies** (`murmura/network_management/`)
- Star (federated), Ring, Complete graph, Line, Custom adjacency matrices
- `TopologyManager`: Handles neighbor relationships and communication patterns
- Automatic compatibility validation between topology and aggregation strategy

**Privacy Framework** (`murmura/privacy/`)
- Comprehensive differential privacy with Opacus integration
- `DPConfig`: Privacy configuration with automatic noise tuning
- `PrivacyAccountant`: RDP-based privacy budget tracking
- Support for client-level and central DP with subsampling amplification

**Data Processing** (`murmura/data_processing/`)
- `MDataset`: Unified interface for HuggingFace, PyTorch, and custom datasets
- `Partitioner`: IID and non-IID (Dirichlet) data distribution
- Lazy vs eager loading strategies for large datasets

### Configuration System

The framework uses Pydantic-based configuration with hierarchical structure:

- `OrchestrationConfig`: Top-level configuration including training parameters
- `TopologyConfig`: Network topology settings
- `AggregationConfig`: Strategy-specific parameters
- `DPConfig`: Differential privacy settings
- `ResourceConfig`: Ray cluster and resource allocation

**Critical Configuration Requirements:**
- `feature_columns` and `label_column` must be specified for each dataset
- Multi-node clusters require proper `RayClusterConfig` setup
- Privacy-enabled learning requires `DPConfig` with appropriate epsilon/delta values

### Dataset Integration

The framework requires explicit column specification for generic dataset handling:

```python
# For image datasets
feature_columns=["image"]
label_column="label"

# For text datasets  
feature_columns=["text"]
label_column="sentiment"

# For tabular datasets
feature_columns=["feature1", "feature2", "feature3"]
label_column="target"
```

### Privacy Implementation

Differential privacy is implemented at multiple levels:
- **Client-level DP**: Noise added to model updates before aggregation
- **Central DP**: Noise added during aggregation (optional)
- **Subsampling amplification**: Privacy amplification through client/data sampling
- **Automatic noise calibration**: Noise multiplier computed to achieve target epsilon

### Multi-Node Support

Ray-based distribution with:
- Automatic cluster detection and resource management
- Placement groups for actor distribution across nodes
- Lazy vs eager dataset loading based on cluster size
- Health monitoring and failure recovery

## Examples Structure

Located in `murmura/examples/`:
- `mnist_example.py`: Basic federated learning
- `dp_mnist_example.py`: DP-enabled federated learning
- `decentralized_mnist_example.py`: Peer-to-peer learning
- `skin_lesion_example.py`: Medical imaging use case

Each example demonstrates full configuration including data partitioning, model setup, and visualization.

## Testing Strategy

Tests are organized by module with integration tests marked:
- Unit tests for individual components
- Integration tests requiring external resources (HuggingFace datasets)
- Privacy tests validating epsilon/delta bounds
- Multi-node cluster tests for distributed functionality

Use `pytest -m "not integration"` for faster local development.

## Common Patterns

**Configuration Creation:**
Always use the hierarchical config objects rather than plain dictionaries for new code.

**Model Wrapping:**
Use `TorchModelWrapper` for PyTorch models with automatic DP compatibility.

**Dataset Loading:**
Use `MDataset.load_dataset_with_multinode_support()` for consistent multi-node behavior.

**Error Handling:**
The framework includes comprehensive validation at initialization - configuration errors are caught early before training begins.