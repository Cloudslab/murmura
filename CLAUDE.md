# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Testing:**
- `pytest` - Run all tests
- `pytest tests/path/to/specific_test.py` - Run specific test file
- `pytest -m "not integration"` - Run tests excluding integration tests that require external resources

**Code Quality:**
- `ruff check` - Run linting
- `ruff format` - Format code
- `mypy murmura/` - Run type checking
- `pytest --cov=murmura tests/` - Run tests with coverage

**Setup:**
- `poetry install` - Install dependencies
- `poetry shell` - Activate virtual environment

## Architecture Overview

Murmura is a decentralized federated learning framework built on Ray for distributed computing. The core architecture consists of several interconnected layers:

### Core Components

**Learning Processes** (`murmura/orchestration/learning_process/`):
- `LearningProcess` (abstract base) - Defines common interface for federated/decentralized learning
- `FederatedLearningProcess` - Centralized aggregation with star topology
- `DecentralizedLearningProcess` - Peer-to-peer learning with various topologies
- Handles actor creation, dataset distribution, model synchronization, and training coordination

**Cluster Management** (`murmura/orchestration/cluster_manager.py`):
- `ClusterManager` - Orchestrates Ray actors across single/multi-node clusters
- Manages actor lifecycle, dataset distribution strategies (eager/lazy loading)
- Handles topology setup and network coordination
- Provides resource monitoring and health checks

**Network Topologies** (`murmura/network_management/`):
- `TopologyManager` - Creates and manages network graphs (star, ring, complete, line, custom)
- `TopologyCoordinator` - Handles communication patterns between nodes
- `TopologyCompatibilityManager` - Validates topology-strategy compatibility

**Aggregation Strategies** (`murmura/aggregation/`):
- `AggregationStrategy` interface with coordination modes (centralized/decentralized)
- Implementations: FedAvg, TrimmedMean, GossipAvg
- `AggregationStrategyFactory` for strategy instantiation
- Coordinate with topologies for distributed aggregation

**Data Processing** (`murmura/data_processing/`):
- `MDataset` - Unified interface for HuggingFace datasets, PyTorch datasets, and partitioned data
- `Partitioner` - Handles IID/non-IID data distribution (Dirichlet, quantity-based)
- `PartitionerFactory` - Creates appropriate partitioners based on configuration

**Client Actors** (`murmura/node/client_actor.py`):
- `VirtualClientActor` - Ray actor representing individual federated learning clients
- Handles local training, model updates, and peer communication
- Supports lazy/eager dataset loading for memory efficiency

### Key Design Patterns

**Configuration-Driven Architecture:**
All components use Pydantic models for type-safe configuration (e.g., `OrchestrationConfig`, `AggregationConfig`, `TopologyConfig`).

**Ray-Based Distribution:**
Built on Ray for actor-based distributed computing with automatic failover and resource management across single/multi-node clusters.

**Strategy Pattern:**
Aggregation strategies and partitioners use factory pattern for pluggable implementations.

**Observer Pattern:**
Training events are broadcast through `TrainingMonitor` to registered observers for visualization and metrics collection.

## Working with Examples

Examples in `murmura/examples/` demonstrate complete workflows:
- `mnist_example.py` - Basic federated learning with MNIST
- `decentralized_mnist_example.py` - Decentralized learning without central coordinator
- `skin_lesion_example.py` - Medical imaging federated learning

Examples use `OrchestrationConfig` to specify:
- Number of clients, training rounds, topology type
- Aggregation strategy, data partitioning method
- Ray cluster configuration and resource allocation

## Important Implementation Notes

**Dataset Distribution:**
The framework automatically chooses between eager (load all data immediately) and lazy (load on-demand) strategies based on dataset size and cluster resources.

**Actor Validation:**
After initialization, all actors are validated to ensure proper dataset and model distribution before training begins.

**Multi-Node Support:**
The cluster manager detects and adapts to single-node vs multi-node Ray clusters automatically.

**Differential Privacy** (`murmura/privacy/`):
The framework includes comprehensive differential privacy support built on Opacus:
- `DPConfig` - Configuration for privacy parameters with presets for MNIST and skin lesion datasets
- `DPTorchModelWrapper` - DP-aware model wrapper that integrates Opacus for client-level privacy
- `PrivacyAccountant` - Tracks privacy budget across clients and rounds using RDP accounting
- `DPFedAvg`, `DPSecureAggregation`, `DPTrimmedMean` - DP-enabled aggregation strategies
- Examples: `dp_mnist_example.py` and `dp_skin_lesion_example.py` demonstrate usage