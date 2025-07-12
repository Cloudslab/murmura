# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Requirements

- **Python Version**: >= 3.11 (enforced in pyproject.toml and CI/CD)
- **License**: GPL-3.0-only
- **Build System**: Poetry (poetry-core)
- **Git LFS**: Required for large files (run `git lfs install`)

## Essential Commands

### Setup and Development
```bash
# Install dependencies and activate environment
poetry install
poetry shell

# Alternative installation via pip
pip install murmura

# Running examples
python murmura/examples/dp_mnist_example.py
python murmura/examples/dp_cifar10_example.py
python murmura/examples/dp_decentralized_mnist_example.py
python murmura/examples/dp_decentralized_cifar10_example.py
python murmura/examples/dp_ham10000_example.py
python murmura/examples/dp_decentralized_ham10000_example.py
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=murmura tests/

# Run specific test file
pytest tests/test_specific_file.py

# Run single test
pytest tests/test_file.py::test_function_name

# Skip integration tests (faster, no external dependencies)
pytest -m "not integration"

# Run tests with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_pattern"

# Run comprehensive CUSUM trust detection evaluation
./test_cusum_detection.sh
```

### Code Quality
```bash
# Linting
ruff check

# Auto-format code
ruff format

# Type checking
mypy murmura/

# Check specific file
mypy murmura/path/to/file.py

# All quality checks before committing
ruff check && ruff format && mypy murmura/
```

## High-Level Architecture

### Core Design Pattern
Murmura uses Ray's actor model for distributed computing. Each client is a Ray actor that maintains its own state and communicates through Ray's distributed messaging system.

### Key Architectural Components

1. **Learning Process Layer** (`orchestration/`)
   - `FederatedLearningProcess`: Manages centralized federated learning with a server actor
   - `DecentralizedLearningProcess`: Manages peer-to-peer learning without central coordination
   - Both inherit from `BaseLearningProcess` and handle the overall training lifecycle

2. **Actor System** (`node/`, `orchestration/cluster_manager.py`)
   - `ClientActor`: Ray actors representing individual clients/nodes
   - `ServerActor`: Central aggregation server (federated mode only)
   - `ClusterManager`: Manages Ray cluster initialization and actor placement

3. **Network Topology** (`network_management/`)
   - Defines communication patterns between nodes
   - Supports star (federated), ring, complete, line, and custom topologies
   - Automatically validates topology compatibility with learning paradigm

4. **Aggregation Framework** (`aggregation/`)
   - Strategy pattern for different aggregation methods
   - Each strategy has a base version and a DP-enabled variant
   - Aggregation happens either centrally (federated) or locally (decentralized)

5. **Privacy System** (`privacy/`)
   - Deep integration with Opacus for differential privacy
   - Privacy budget tracking across all clients and rounds
   - Automatic model wrapping and noise calibration
   - RDP (Rényi DP) accounting with amplification by subsampling

### Critical Design Decisions

1. **Ray Actor Lifecycle**: Actors are created per-round to avoid state persistence issues. This is intentional for fault tolerance.

2. **Dataset Loading**: Supports both eager (load all data upfront) and lazy (load per-round) strategies based on dataset size and memory constraints.

3. **Model Distribution**: Models are serialized and distributed via Ray's object store, not through file systems.

4. **Decentralized Communication**: Uses Ray's actor references for peer-to-peer communication, abstracting away network details.

5. **Configuration Hierarchy**: Nested configuration objects (OrchestrationConfig contains AggregationConfig, PrivacyConfig, etc.) for modularity. Pydantic models provide validation and type safety.

6. **Multi-Node Support**: Automatic resource allocation and cluster discovery for distributed deployments across multiple machines.

### Common Patterns

1. **Adding New Aggregation Strategy**:
   - Create class in `aggregation/strategies/`
   - Implement `aggregate()` method
   - Add DP variant if applicable
   - Register in `AggregationStrategyFactory`

2. **Adding New Dataset**:
   - Implement `DatasetInterface` in `data_processing/datasets/`
   - Handle partitioning logic
   - Add to dataset registry

3. **Adding New Network Topology**:
   - Extend `NetworkTopology` class
   - Implement adjacency matrix generation
   - Validate compatibility with learning paradigms

### Important Considerations

- Always check if running in distributed mode before making Ray-specific calls
- Privacy-enabled models require compatible aggregation strategies
- Decentralized learning requires compatible network topologies (not star)
- Integration tests require Ray cluster and may download external datasets
- Type hints are enforced - run mypy before committing

### Debugging and Development Tips

1. **Ray Dashboard**: Access at http://localhost:8265 when running locally for cluster monitoring
2. **Logging**: Set environment variable `RAY_DEDUP_LOGS=0` to see all actor logs
3. **Memory Issues**: Use lazy dataset loading for large datasets like HAM10000
4. **Testing New Features**: Always test with both federated and decentralized modes
5. **Privacy Budget**: Monitor epsilon values carefully - they accumulate across rounds

### Project Structure Quick Reference

- `murmura/orchestration/` - Entry point for all learning processes
- `murmura/aggregation/strategies/` - Add new aggregation methods here
- `murmura/data_processing/datasets/` - Add new dataset implementations here
- `murmura/models/` - Model architectures (CNN, ResNet, etc.)
- `murmura/privacy/` - Differential privacy implementation
- `murmura/visualization/` - Training metrics and network topology visualization

### Common Error Patterns

1. **"Actor not found"**: Usually means Ray cluster isn't initialized properly
2. **"Incompatible topology"**: Check that topology matches learning paradigm (e.g., no star topology for decentralized)
3. **"Privacy budget exceeded"**: Reduce rounds or adjust epsilon/delta values
4. **Type errors**: Run `mypy` to catch before runtime
5. **"Failed to load dataset"**: Check dataset path and ensure proper file permissions
6. **Memory errors with large datasets**: Switch to lazy loading strategy in dataset configuration
7. **Ray object store full**: Increase Ray object store memory or use smaller batch sizes
8. **"feature_columns must be specified"**: OrchestrationConfig requires explicit feature_columns and label_column settings for each dataset
9. **"Dataset distribution validation failed"**: Indicates issues with lazy loading or column distribution - check actor dataset state

### Attack and Security Research Features

**IMPORTANT**: This codebase includes attack simulation capabilities for research purposes only. These features are designed for defensive security research and should not be used maliciously.

#### Attack Configuration (`attacks/`)
- **Label Flipping**: Simulates data poisoning attacks where malicious clients flip labels
- **Gradient Manipulation**: Models gradient-based attacks with noise injection and sign flipping
- **Malicious Client Simulation**: Automated creation of malicious client actors with configurable attack parameters
- **Attack Intensity Progression**: Supports linear, exponential, and step-based attack escalation patterns

#### Trust Monitoring (`trust_monitoring/`)
- **Anomaly Detection**: Statistical methods (CUSUM, Z-score, IQR) for detecting malicious behavior
- **Trust Score Management**: Dynamic trust scoring with decay and recovery mechanisms
- **Neighbor Consensus**: Consensus-based detection using neighbor comparisons
- **Real-time Monitoring**: Event-driven trust monitoring with configurable thresholds

#### Configuration Examples
```python
# Enable attack simulation for research
attack_config = AttackConfig(
    malicious_clients_ratio=0.2,  # 20% malicious clients
    attack_type="label_flipping",
    attack_intensity_start=0.1,
    attack_intensity_end=1.0,
    intensity_progression="linear"
)

# Enable trust monitoring
trust_config = TrustMonitorConfig(
    enable_trust_monitoring=True,
    anomaly_detection_method="cusum",
    suspicion_threshold=0.7
)

config = OrchestrationConfig(
    attack_config=attack_config,
    trust_monitoring=trust_config,
    # ... other config
)
```

### CI/CD Pipeline

The project uses GitHub Actions for automated testing and deployment:

1. **Main CI Pipeline** (`ci.yml`):
   - Runs on all PRs and pushes
   - Formats code with `ruff format`
   - Lints with `ruff check`
   - Type checks with `mypy`
   - Runs tests with coverage reporting
   
2. **Coverage Reporting** (`coverage.yml`):
   - Posts coverage reports on pull requests
   - Coverage thresholds:
     - Green: >= 90%
     - Orange: >= 70%
     - Red: < 70%

3. **Website Deployment** (`landing-page-deploy.yml`):
   - Deploys project documentation website

### Additional Development Notes

1. **Dependency Management**:
   - `poetry.lock` is not tracked in git
   - Always run `poetry lock` when updating dependencies
   - Use `poetry install` to ensure consistent environments

2. **MyPy Configuration**:
   - The following modules have import checking disabled:
     - `torch.*`, `ray.*`, `datasets.*`, `matplotlib.*`, `networkx.*`
   - This is configured in `pyproject.toml`

3. **Test Configuration**:
   - Integration tests marked with `integration` marker
   - DeprecationWarning filtered out in pytest
   - Coverage uses relative file paths

4. **Example Scripts**:
   - All examples support command-line arguments
   - Trust monitoring can be enabled via CLI flags
   - Network topology can be specified
   - Support creating animations of training process

5. **Trust Detection Evaluation**:
   - `test_cusum_detection.sh` provides comprehensive evaluation
   - Tests across multiple datasets and topologies
   - Creates detailed reports in `cusum_detection_results/`