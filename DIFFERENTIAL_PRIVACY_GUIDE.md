# Differential Privacy Implementation Guide

This guide explains how to use the comprehensive differential privacy (DP) implementation in the Murmura federated learning framework.

## Overview

The DP implementation provides state-of-the-art privacy protection using the Opacus library, with automatic parameter tuning and comprehensive privacy accounting. It supports both MNIST and skin lesion classification with optimized hyperparameters for good utility.

## Key Features

✅ **Opacus Integration**: Built on Meta's production-ready DP library  
✅ **Automatic Parameter Tuning**: Computes optimal noise multipliers  
✅ **Privacy Accounting**: Tracks (ε, δ) privacy budget with RDP accounting  
✅ **Multiple Presets**: High, medium, low privacy configurations  
✅ **Federated & Decentralized**: Works with all topologies  
✅ **Model Compatibility**: Automatic model fixing for DP training  
✅ **Comprehensive Testing**: Unit and integration tests included  

## Quick Start

### 1. Basic DP Training (MNIST)

```bash
# Run federated learning with differential privacy
python murmura/examples/dp_mnist_example.py \
    --enable_dp \
    --dp_preset medium_privacy \
    --num_actors 5 \
    --rounds 10 \
    --create_summary
```

### 2. Custom Privacy Configuration

```bash
# Run with custom privacy parameters
python murmura/examples/dp_mnist_example.py \
    --enable_dp \
    --dp_preset custom \
    --target_epsilon 5.0 \
    --target_delta 1e-5 \
    --max_grad_norm 1.0 \
    --enable_client_dp \
    --rounds 15
```

### 3. Skin Lesion Classification with DP

```bash
# Medical data with high privacy protection
python murmura/examples/dp_skin_lesion_example.py \
    --enable_dp \
    --dp_preset high_privacy \
    --num_actors 3 \
    --rounds 5 \
    --batch_size 16
```

## Privacy Presets

### High Privacy (ε ≤ 3.0)
- **Use Case**: Sensitive medical data, strict privacy requirements
- **Performance**: Lower accuracy but strong privacy guarantees
- **Configuration**: `--dp_preset high_privacy`

### Medium Privacy (ε = 8-10)
- **Use Case**: General federated learning, good privacy-utility tradeoff
- **Performance**: Good accuracy with reasonable privacy protection
- **Configuration**: `--dp_preset medium_privacy` (default)

### Low Privacy (ε ≥ 16)
- **Use Case**: Less sensitive data, utility-focused
- **Performance**: High accuracy with basic privacy protection
- **Configuration**: `--dp_preset low_privacy`

## Architecture Components

### 1. DPConfig
Central configuration for all DP parameters:

```python
from murmura.privacy.dp_config import DPConfig

# MNIST optimized config
config = DPConfig.create_for_mnist()

# Custom config
config = DPConfig(
    target_epsilon=8.0,
    target_delta=1e-5,
    max_grad_norm=1.0,
    enable_client_dp=True
)
```

### 2. DPTorchModelWrapper
Drop-in replacement for TorchModelWrapper with DP support:

```python
from murmura.privacy.dp_model_wrapper import DPTorchModelWrapper
from murmura.privacy.dp_config import DPConfig

dp_config = DPConfig.create_for_mnist()
dp_model = DPTorchModelWrapper(
    model=your_model,
    dp_config=dp_config,
    optimizer_class=torch.optim.SGD,  # SGD recommended for DP
    optimizer_kwargs={"lr": 0.01, "momentum": 0.9}
)

# Train with automatic DP
results = dp_model.train(data, labels, batch_size=64, epochs=1)
print(f"Privacy spent: ε={results['privacy_spent']['epsilon']:.3f}")
```

### 3. Privacy Accountant
Tracks privacy budget across clients and rounds:

```python
from murmura.privacy.privacy_accountant import PrivacyAccountant

accountant = PrivacyAccountant(dp_config)

# Record training round
accountant.record_training_round(
    client_id="client_1",
    noise_multiplier=1.2,
    sample_rate=0.1,
    steps=100,
    round_number=1
)

# Check privacy status
summary = accountant.get_privacy_summary()
print(f"Global privacy utilization: {summary['global_privacy']['utilization_percentage']:.1f}%")
```

### 4. DP Aggregation Strategies
Privacy-preserving aggregation for central servers:

```python
from murmura.privacy.dp_aggregation import DPFedAvg

dp_aggregator = DPFedAvg(dp_config)
aggregated_params = dp_aggregator.aggregate(
    parameters_list=client_parameters,
    weights=client_weights
)
```

## Privacy vs. Utility Guidance

### For MNIST:
- **ε = 1.0**: ~85% accuracy (high privacy)
- **ε = 8.0**: ~95% accuracy (medium privacy)  
- **ε = 16.0**: ~97% accuracy (low privacy)

### For Skin Lesion:
- **ε = 3.0**: ~65% accuracy (high privacy)
- **ε = 10.0**: ~75% accuracy (medium privacy)
- **ε = 20.0**: ~80% accuracy (low privacy)

## Best Practices

### 1. Hyperparameter Selection
- **Learning Rate**: Use smaller LR with DP (0.001-0.01)
- **Batch Size**: Larger batches work better (64-128)
- **Gradient Clipping**: 1.0-1.2 for most models
- **Optimizer**: SGD with momentum preferred over Adam

### 2. Model Architecture
- **Batch Normalization**: Essential for DP training stability
- **Dropout**: Helps with DP noise tolerance
- **Model Size**: Smaller models often perform better with DP

### 3. Privacy Budget Management
- **Monitor Usage**: Check `privacy_spent` after each round
- **Early Stopping**: Stop training when budget approaches limit
- **Multi-Round Training**: Distribute budget across rounds

### 4. Debugging DP Issues

#### Low Accuracy
1. **Increase ε**: Try higher privacy budget
2. **Tune Learning Rate**: Often needs to be lower
3. **Adjust Batch Size**: Larger batches reduce noise impact
4. **Check Model Compatibility**: Use `ModuleValidator.validate()`

#### Privacy Budget Exhausted
1. **Reduce Training Steps**: Fewer epochs or rounds
2. **Increase Noise**: Higher noise multiplier (auto-tuned by default)
3. **Lower Sample Rate**: Smaller batch size relative to dataset

#### Training Instability
1. **Gradient Clipping**: Try lower `max_grad_norm`
2. **Batch Normalization**: Ensure model uses BatchNorm
3. **Learning Rate**: Reduce by 2-5x compared to non-DP training

## Example Configurations

### Conservative (Medical Data)
```python
config = DPConfig(
    target_epsilon=1.0,    # Very private
    target_delta=1e-6,     # Conservative delta
    max_grad_norm=0.5,     # Aggressive clipping
    secure_mode=True       # Maximum security
)
```

### Balanced (General Use)
```python
config = DPConfig(
    target_epsilon=8.0,    # Reasonable privacy
    target_delta=1e-5,     # Standard delta
    max_grad_norm=1.0,     # Standard clipping
    auto_tune_noise=True   # Optimal noise
)
```

### Utility-Focused (Less Sensitive Data)
```python
config = DPConfig(
    target_epsilon=16.0,   # Higher budget
    target_delta=1e-4,     # Relaxed delta
    max_grad_norm=2.0,     # Minimal clipping
    mechanism="gaussian"   # Standard mechanism
)
```

## Testing

Run the complete DP test suite:

```bash
# All DP tests
poetry run pytest tests/privacy/ -v

# Specific test categories
poetry run pytest tests/privacy/dp_config_test.py -v
poetry run pytest tests/privacy/dp_integration_test.py -v
```

## Troubleshooting

### Common Issues

1. **Opacus ImportError**: Install with `poetry install` or `pip install opacus`
2. **CUDA Memory Issues**: Reduce batch size or use `BatchMemoryManager`
3. **Model Validation Errors**: Use `ModuleValidator.fix()` for automatic fixes
4. **Poor Convergence**: Reduce learning rate and increase batch size

### Getting Help

- Check logs for detailed DP parameter information
- Use `--log_level DEBUG` for verbose privacy accounting
- Monitor privacy spent with `get_privacy_spent()`
- Visualize training with `--create_summary`

## Integration with Existing Code

The DP implementation is designed as a drop-in replacement:

```python
# Replace this:
model = TorchModelWrapper(...)

# With this:
model = DPTorchModelWrapper(..., dp_config=dp_config)

# Everything else stays the same!
results = learning_process.execute()
```

No changes needed to aggregation strategies, topologies, or orchestration - DP is fully integrated into the existing framework architecture.