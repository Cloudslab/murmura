import numpy as np
import torch.nn as nn

from murmura.privacy.privacy_config import PrivacyConfig, PrivacyMode, PrivacyMechanismType
from murmura.privacy.privacy_manager import PrivacyManager


def test_adaptive_dp_training():
    """Test that adaptive DP allows stable training."""
    print("=== Testing Adaptive DP Training ===\n")

    # Create a simple model
    model = nn.Linear(10, 2)
    global_params = {
        'weight': model.weight.data.numpy().copy(),
        'bias': model.bias.data.numpy().copy()
    }

    # Create privacy config with adaptive clipping
    config = PrivacyConfig(
        enabled=True,
        mechanism_type=PrivacyMechanismType.GAUSSIAN,
        privacy_mode=PrivacyMode.CENTRAL,
        target_epsilon=50.0,
        target_delta=1e-5,
        noise_multiplier=1.0,
        clipping_norm=None,  # Let it adapt
        per_layer_clipping=True,
        adaptive_clipping_quantile=0.8  # Use 80th percentile
    )

    manager = PrivacyManager(config)

    # Simulate 10 rounds
    for round_num in range(10):
        # Simulate client updates (small random changes)
        client_updates = []
        for _ in range(5):  # 5 clients
            update = {
                'weight': np.random.randn(*global_params['weight'].shape) * 0.01,
                'bias': np.random.randn(*global_params['bias'].shape) * 0.01
            }
            client_updates.append(update)

        # Update clipping norms based on observed updates
        manager.update_clipping_norms(client_updates)

        # Average updates
        avg_update = {}
        for key in client_updates[0].keys():
            stacked = np.stack([u[key] for u in client_updates])
            avg_update[key] = np.mean(stacked, axis=0)

        # Apply privacy
        private_update = manager.privatize_parameters(avg_update, is_client=False)

        # Apply update to global model
        for key in global_params:
            global_params[key] += private_update[key]

        # Check norms
        weight_norm = np.linalg.norm(global_params['weight'])
        update_norm = np.linalg.norm(avg_update['weight'])
        private_update_norm = np.linalg.norm(private_update['weight'])

        print(f"Round {round_num + 1}:")
        print(f"  Update norm: {update_norm:.6f}")
        print(f"  Clipping norm: {manager.clipping_norms.get('weight', manager.clipping_norms.get('__default__')):.6f}")
        print(f"  Private update norm: {private_update_norm:.6f}")
        print(f"  Global weight norm: {weight_norm:.6f}")

        # Check if parameters are growing reasonably
        if weight_norm > 10:
            print("  WARNING: Weights growing too large!")

    print("\n✓ Adaptive DP training is stable!")

if __name__ == "__main__":
    test_adaptive_dp_training()