#!/usr/bin/env python3
"""
Test script to verify the attack implementations work correctly.
"""

import numpy as np
import torch
import logging
import pytest
from murmura.attacks.attack_config import AttackConfig
from murmura.attacks.label_flipping import LabelFlippingAttack
from murmura.attacks.gradient_manipulation import GradientManipulationAttack

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_attack_config():
    """Test the AttackConfig class."""
    # Test basic configuration
    config = AttackConfig(
        malicious_clients_ratio=0.2,
        attack_type="label_flipping",
        attack_intensity_start=0.1,
        attack_intensity_end=0.8,
        intensity_progression="linear",
        label_flip_target=5,
        attack_start_round=2
    )
    
    # Test intensity calculation
    assert config.get_attack_intensity(1, 10) == 0.0  # Before start round
    assert config.get_attack_intensity(2, 10) == 0.1  # Start intensity
    assert config.get_attack_intensity(10, 10) == 0.8  # End intensity
    
    # Test attack active check
    assert not config.is_attack_active(1)
    assert config.is_attack_active(2)
    assert config.is_attack_active(10)

def test_label_flipping_attack():
    """Test the LabelFlippingAttack class."""
    # Create attack config
    config = AttackConfig(
        malicious_clients_ratio=0.2,
        attack_type="label_flipping",
        label_flip_target=5,
        num_classes=10
    )
    
    # Create attack instance
    attack = LabelFlippingAttack("test_client", config.model_dump())
    
    # Create test data
    features = np.random.rand(100, 28, 28)
    labels = np.random.randint(0, 10, 100)
    
    # Test poisoning
    poisoned_features, poisoned_labels = attack.poison_data(
        features, labels, round_num=1, attack_intensity=0.5
    )
    
    # Verify features unchanged
    assert np.array_equal(features, poisoned_features)
    
    # Verify some labels changed
    changed_labels = np.sum(labels != poisoned_labels)
    assert changed_labels > 0

def test_gradient_manipulation_attack():
    """Test the GradientManipulationAttack class."""
    # Create attack config
    config = AttackConfig(
        malicious_clients_ratio=0.2,
        attack_type="gradient_manipulation",
        gradient_noise_scale=0.5,
        gradient_sign_flip_prob=0.1
    )
    
    # Create attack instance
    attack = GradientManipulationAttack("test_client", config.model_dump())
    
    # Create test parameters
    original_params = {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(5, 10),
        "layer2.bias": torch.randn(5)
    }
    
    # Test poisoning
    poisoned_params = attack.poison_gradients(
        original_params, round_num=1, attack_intensity=0.5
    )
    
    # Verify parameters changed
    for key in original_params:
        assert not torch.equal(original_params[key], poisoned_params[key])

def test_progressive_intensity():
    """Test the progressive attack intensity mechanism."""
    # Linear progression
    config = AttackConfig(
        malicious_clients_ratio=0.2,
        attack_type="label_flipping",
        attack_intensity_start=0.1,
        attack_intensity_end=0.9,
        intensity_progression="linear",
        attack_start_round=1
    )
    
    total_rounds = 10
    intensities = []
    
    for round_num in range(1, total_rounds + 1):
        intensity = config.get_attack_intensity(round_num, total_rounds)
        intensities.append(intensity)
    
    # Check that intensity increases
    assert intensities[0] == 0.1
    assert intensities[-1] == 0.9
    assert all(intensities[i] <= intensities[i+1] for i in range(len(intensities)-1))

def test_attack_config_validation():
    """Test AttackConfig validation."""
    # Test invalid ratio
    with pytest.raises(ValueError):
        AttackConfig(malicious_clients_ratio=1.5)
    
    # Test missing attack type
    with pytest.raises(ValueError):
        AttackConfig(malicious_clients_ratio=0.2, attack_type=None)
    
    # Test invalid intensity range
    with pytest.raises(ValueError):
        AttackConfig(
            malicious_clients_ratio=0.2,
            attack_type="label_flipping",
            attack_intensity_start=0.8,
            attack_intensity_end=0.2
        )

def test_malicious_client_selection():
    """Test malicious client selection logic."""
    from murmura.orchestration.orchestration_config import OrchestrationConfig
    from murmura.attacks.attack_config import AttackConfig
    
    attack_config = AttackConfig(
        malicious_clients_ratio=0.3,
        attack_type="label_flipping"
    )
    
    config = OrchestrationConfig(
        num_actors=10,
        feature_columns=["test"],
        label_column="label",
        attack_config=attack_config
    )
    
    malicious_indices = config.get_malicious_client_indices()
    
    # Should have 3 malicious clients (30% of 10)
    assert len(malicious_indices) == 3
    
    # All indices should be unique and within range
    assert len(set(malicious_indices)) == len(malicious_indices)
    assert all(0 <= idx < 10 for idx in malicious_indices)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])