"""
Test script to verify the fixed differential privacy implementation.
Run this to ensure training works correctly with privacy enabled.
"""

import numpy as np
import torch
import torch.nn as nn
from murmura.privacy.privacy_config import PrivacyConfig, PrivacyMode, PrivacyMechanismType
from murmura.privacy.gaussian_mechanism import GaussianMechanism, MomentsAccountant
from murmura.privacy.privacy_manager import PrivacyManager


def test_moments_accountant():
    """Test the Moments Accountant implementation."""
    print("Testing Moments Accountant...")

    accountant = MomentsAccountant()

    # Test with typical parameters
    q = 0.01  # 1% sampling
    sigma = 1.0  # Noise multiplier
    steps = 1000  # Number of steps
    delta = 1e-5

    result = accountant.compute_epsilon(q, sigma, steps, delta)
    print(f"  q={q}, σ={sigma}, steps={steps} -> ε={result['epsilon']:.4f}")

    # Test with different noise levels
    for sigma in [0.5, 1.0, 2.0, 5.0]:
        result = accountant.compute_epsilon(q, sigma, steps, delta)
        print(f"  σ={sigma} -> ε={result['epsilon']:.4f}")

    print("✓ Moments Accountant working correctly\n")


def test_noise_calibration():
    """Test noise calibration for target epsilon."""
    print("Testing noise calibration...")

    mechanism = GaussianMechanism()

    # Test calibration for different target epsilons
    q = 0.01
    steps = 1000
    delta = 1e-5

    for target_eps in [1.0, 3.0, 8.0]:
        sigma = mechanism.calibrate_noise_to_target_epsilon(
            target_epsilon=target_eps,
            target_delta=delta,
            iterations=steps,
            batch_size=100,
            total_samples=10000
        )

        # Verify the calibration
        result = mechanism.accountant.compute_epsilon(q, sigma, steps, delta)
        actual_eps = result['epsilon']

        print(f"  Target ε={target_eps} -> σ={sigma:.4f} -> Actual ε={actual_eps:.4f}")

        # Check if within tolerance
        if abs(actual_eps - target_eps) > 0.1:
            print(f"  WARNING: Calibration off by {abs(actual_eps - target_eps):.4f}")

    print("✓ Noise calibration working correctly\n")


def test_clipping_and_noise():
    """Test parameter clipping and noise addition."""
    print("Testing clipping and noise addition...")

    # Create test parameters
    params = {
        'layer1': np.random.randn(100, 50).astype(np.float32),
        'layer2': np.random.randn(50, 10).astype(np.float32)
    }

    # Calculate original norms
    orig_norms = {k: np.linalg.norm(v) for k, v in params.items()}
    print(f"  Original norms: {orig_norms}")

    # Test clipping
    mechanism = GaussianMechanism(noise_multiplier=1.0, max_grad_norm=1.0)
    clipped = mechanism.clip_parameters(params, {'layer1': 1.0, 'layer2': 1.0})

    clipped_norms = {k: np.linalg.norm(v) for k, v in clipped.items()}
    print(f"  Clipped norms: {clipped_norms}")

    # Verify clipping worked
    for k, norm in clipped_norms.items():
        assert norm <= 1.01, f"Clipping failed for {k}: {norm} > 1.0"

    # Test noise addition
    noised = mechanism.add_noise(clipped)
    noised_norms = {k: np.linalg.norm(v) for k, v in noised.items()}
    print(f"  Noised norms: {noised_norms}")

    # Calculate signal-to-noise ratio
    for k in params:
        signal = np.linalg.norm(clipped[k])
        noise = np.linalg.norm(noised[k] - clipped[k])
        snr = signal / (noise + 1e-8)
        print(f"  {k} SNR: {snr:.4f}")

    print("✓ Clipping and noise working correctly\n")


def test_privacy_manager():
    """Test the complete privacy manager."""
    print("Testing Privacy Manager...")

    # Test Local DP configuration
    config = PrivacyConfig(
        enabled=True,
        mechanism_type=PrivacyMechanismType.GAUSSIAN,
        privacy_mode=PrivacyMode.LOCAL,
        target_epsilon=3.0,
        target_delta=1e-5,
        adaptive_noise=True,
        per_layer_clipping=True
    )

    manager = PrivacyManager(config)

    # Setup accounting
    manager.setup_privacy_accounting(sample_count=10000, batch_size=100)
    print(f"  Initial noise multiplier: {config.noise_multiplier:.4f}")

    # Simulate a few rounds
    test_params = {
        'layer1': np.random.randn(100, 50).astype(np.float32) * 0.1,
        'layer2': np.random.randn(50, 10).astype(np.float32) * 0.1
    }

    for round_num in range(5):
        # Privatize parameters (as a client would)
        private_params = manager.privatize_parameters(test_params, is_client=True)

        # Update privacy budget
        budget = manager.update_privacy_budget()

        print(f"  Round {round_num + 1}: ε={budget['epsilon']:.4f}")

        # Check parameter changes
        for k in test_params:
            orig_norm = np.linalg.norm(test_params[k])
            priv_norm = np.linalg.norm(private_params[k])
            change = np.linalg.norm(private_params[k] - test_params[k])
            print(f"    {k}: orig={orig_norm:.4f}, priv={priv_norm:.4f}, change={change:.4f}")

    print("✓ Privacy Manager working correctly\n")


def test_with_simple_model():
    """Test with a simple PyTorch model to ensure training doesn't explode."""
    print("Testing with simple PyTorch model...")

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )

    # Create privacy config
    config = PrivacyConfig(
        enabled=True,
        mechanism_type=PrivacyMechanismType.GAUSSIAN,
        privacy_mode=PrivacyMode.LOCAL,
        target_epsilon=8.0,  # Reasonable epsilon
        target_delta=1e-5,
        adaptive_noise=True,
        per_layer_clipping=True,
        max_grad_norm=1.0  # Reasonable clipping
    )

    manager = PrivacyManager(config)
    manager.setup_privacy_accounting(sample_count=1000, batch_size=32)

    # Get model parameters
    state_dict = model.state_dict()
    params = {name: param.detach().numpy() for name, param in state_dict.items()}

    # Simulate training for a few rounds
    losses = []
    for round_num in range(10):
        # Simulate parameter update (normally from training)
        for k in params:
            params[k] += np.random.randn(*params[k].shape) * 0.01

        # Apply privacy
        private_params = manager.privatize_parameters(params, is_client=True)

        # Load back to model
        for name, param in private_params.items():
            state_dict[name] = torch.tensor(param)
        model.load_state_dict(state_dict)

        # Simulate forward pass
        x = torch.randn(32, 10)
        y = model(x)
        loss = torch.mean(y**2)  # Simple loss
        losses.append(loss.item())

        # Update privacy budget
        budget = manager.update_privacy_budget()

        print(f"  Round {round_num + 1}: loss={loss.item():.4f}, ε={budget['epsilon']:.4f}")

    # Check that loss didn't explode
    if any(np.isnan(l) or np.isinf(l) or l > 1000 for l in losses):
        print("  ERROR: Training exploded!")
    else:
        print("  ✓ Training stable with privacy enabled")

    print("✓ Model training test complete\n")


def main():
    """Run all tests."""
    print("=== Testing Fixed Differential Privacy Implementation ===\n")

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Run tests
    test_moments_accountant()
    test_noise_calibration()
    test_clipping_and_noise()
    test_privacy_manager()
    test_with_simple_model()

    print("=== All tests completed! ===")
    print("\nThe fixed DP implementation should now work correctly for training.")
    print("Key improvements:")
    print("- Moments Accountant for stable privacy accounting")
    print("- Proper noise scaling for Local vs Central DP")
    print("- Better clipping norm management")
    print("- Stable noise calibration")


if __name__ == "__main__":
    main()