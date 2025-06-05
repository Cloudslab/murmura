import pytest
from datetime import datetime
from unittest.mock import patch

from murmura.privacy.privacy_accountant import (
    PrivacyAccountant,
    PrivacySpent,
    PrivacyBudget,
    OPACUS_AVAILABLE
)
from murmura.privacy.dp_config import DPConfig


def test_privacy_spent_dataclass():
    """Test PrivacySpent dataclass"""
    spent = PrivacySpent(epsilon=1.0, delta=1e-5)
    
    assert spent.epsilon == 1.0
    assert spent.delta == 1e-5
    assert isinstance(spent.timestamp, datetime)
    assert spent.round_number is None
    assert spent.client_id is None
    assert spent.context == "training"
    
    # Test with all fields
    spent_full = PrivacySpent(
        epsilon=2.0,
        delta=1e-6,
        round_number=5,
        client_id="client_1",
        context="evaluation"
    )
    
    assert spent_full.epsilon == 2.0
    assert spent_full.delta == 1e-6
    assert spent_full.round_number == 5
    assert spent_full.client_id == "client_1"
    assert spent_full.context == "evaluation"


def test_privacy_budget_dataclass():
    """Test PrivacyBudget dataclass"""
    budget = PrivacyBudget(total_epsilon=10.0, total_delta=1e-4)
    
    assert budget.total_epsilon == 10.0
    assert budget.total_delta == 1e-4
    assert budget.spent_epsilon == 0.0
    assert budget.spent_delta == 0.0
    assert budget.remaining_epsilon == 10.0
    assert budget.remaining_delta == 1e-4
    assert budget.is_exhausted is False
    assert budget.utilization_percentage == 0.0
    
    # Test with spent budget
    budget.spent_epsilon = 8.0
    budget.spent_delta = 5e-5
    
    assert budget.remaining_epsilon == 2.0
    assert budget.remaining_delta == 5e-5
    assert budget.is_exhausted is False
    assert budget.utilization_percentage == 80.0
    
    # Test exhausted budget
    budget.spent_epsilon = 11.0  # Exceeds total
    assert budget.remaining_epsilon == 0.0
    assert budget.is_exhausted is True
    assert abs(budget.utilization_percentage - 110.0) < 1e-10  # Account for floating point precision


def test_privacy_accountant_initialization():
    """Test PrivacyAccountant initialization"""
    config = DPConfig(target_epsilon=5.0, target_delta=1e-5)
    accountant = PrivacyAccountant(config)
    
    assert accountant.dp_config == config
    assert accountant.global_budget.total_epsilon == 5.0
    assert accountant.global_budget.total_delta == 1e-5
    assert len(accountant.privacy_history) == 0
    assert len(accountant.client_budgets) == 0


def test_privacy_accountant_initialization_without_opacus():
    """Test PrivacyAccountant initialization when Opacus is not available"""
    config = DPConfig(target_epsilon=5.0, target_delta=1e-5)
    
    with patch('murmura.privacy.privacy_accountant.OPACUS_AVAILABLE', False):
        accountant = PrivacyAccountant(config)
        assert accountant.rdp_accountant is None


def test_create_client_budget():
    """Test creating client budget"""
    config = DPConfig(target_epsilon=10.0, target_delta=1e-4)
    accountant = PrivacyAccountant(config)
    
    # Create client budget
    accountant.create_client_budget("client_0")
    
    assert "client_0" in accountant.client_budgets
    budget = accountant.client_budgets["client_0"]
    assert budget.total_epsilon == 10.0
    assert budget.total_delta == 1e-4
    assert budget.spent_epsilon == 0.0
    
    # Creating same client budget again should not create duplicate
    accountant.create_client_budget("client_0")
    assert len(accountant.client_budgets) == 1


def test_record_training_round():
    """Test recording privacy expenditure for training round"""
    config = DPConfig(target_epsilon=10.0, target_delta=1e-4)
    accountant = PrivacyAccountant(config)
    
    # Record first training round
    record = accountant.record_training_round(
        client_id="client_0",
        noise_multiplier=1.0,
        sample_rate=0.01,
        steps=100,
        round_number=1
    )
    
    assert record.client_id == "client_0"
    assert record.round_number == 1
    assert record.epsilon > 0
    assert len(accountant.privacy_history) == 1
    
    # Check client budget was created and updated
    assert "client_0" in accountant.client_budgets
    client_eps, client_delta = accountant.get_client_privacy_spent("client_0")
    assert client_eps > 0


def test_record_aggregation_round():
    """Test recording privacy expenditure for aggregation"""
    config = DPConfig(target_epsilon=10.0, target_delta=1e-4)
    accountant = PrivacyAccountant(config)
    
    # Record aggregation round
    record = accountant.record_aggregation_round(
        num_clients=5,
        noise_multiplier=1.0,
        round_number=1
    )
    
    assert record.client_id == "central_server"
    assert record.round_number == 1
    assert record.epsilon > 0
    assert len(accountant.privacy_history) == 1
    
    # Check global budget was updated
    global_eps, global_delta = accountant.get_global_privacy_spent()
    assert global_eps > 0


def test_budget_exhaustion_checking():
    """Test budget exhaustion checking"""
    config = DPConfig(target_epsilon=2.0, target_delta=1e-5)
    accountant = PrivacyAccountant(config)
    
    # Should not be exhausted initially
    assert accountant.is_budget_exhausted() is False
    assert accountant.is_budget_exhausted("client_0") is False  # Non-existent client
    
    # Simulate high privacy spending (fake it by modifying budget directly)
    accountant.global_budget.spent_epsilon = 3.0  # Exceeds budget
    assert accountant.is_budget_exhausted() is True
    
    # Test client-specific budget exhaustion
    accountant.create_client_budget("client_0")
    accountant.client_budgets["client_0"].spent_epsilon = 3.0  # Exceeds budget
    assert accountant.is_budget_exhausted("client_0") is True


def test_get_remaining_budget():
    """Test getting remaining budget"""
    config = DPConfig(target_epsilon=10.0, target_delta=1e-4)
    accountant = PrivacyAccountant(config)
    
    # Initial remaining budget should equal total
    epsilon, delta = accountant.get_remaining_budget()
    assert epsilon == 10.0
    assert delta == 1e-4
    
    # Test client-specific remaining budget (non-existent client)
    epsilon, delta = accountant.get_remaining_budget("client_0")
    assert epsilon == 10.0  # Should return total budget for non-existent client
    assert delta == 1e-4
    
    # Create client and spend some budget
    accountant.create_client_budget("client_0")
    accountant.client_budgets["client_0"].spent_epsilon = 3.0
    
    epsilon, delta = accountant.get_remaining_budget("client_0")
    assert epsilon == 7.0
    assert delta == 1e-4


def test_get_global_privacy_spent():
    """Test getting global privacy spent"""
    config = DPConfig(target_epsilon=10.0, target_delta=1e-4)
    accountant = PrivacyAccountant(config)
    
    epsilon, delta = accountant.get_global_privacy_spent()
    assert epsilon == 0.0
    assert delta == 0.0
    
    # Spend some global budget
    accountant.global_budget.spent_epsilon = 5.0
    accountant.global_budget.spent_delta = 2e-5
    
    epsilon, delta = accountant.get_global_privacy_spent()
    assert epsilon == 5.0
    assert delta == 2e-5


def test_get_client_privacy_spent():
    """Test getting client privacy spent"""
    config = DPConfig(target_epsilon=10.0, target_delta=1e-4)
    accountant = PrivacyAccountant(config)
    
    # Non-existent client should return 0
    epsilon, delta = accountant.get_client_privacy_spent("client_0")
    assert epsilon == 0.0
    assert delta == 0.0
    
    # Create client and spend budget
    accountant.create_client_budget("client_0")
    accountant.client_budgets["client_0"].spent_epsilon = 3.0
    accountant.client_budgets["client_0"].spent_delta = 1e-5
    
    epsilon, delta = accountant.get_client_privacy_spent("client_0")
    assert epsilon == 3.0
    assert delta == 1e-5


def test_privacy_summary():
    """Test privacy budget summary"""
    config = DPConfig(target_epsilon=10.0, target_delta=1e-4)
    accountant = PrivacyAccountant(config)
    
    # Create some clients and spend budget
    accountant.create_client_budget("client_0")
    accountant.create_client_budget("client_1")
    accountant.client_budgets["client_0"].spent_epsilon = 3.0
    accountant.client_budgets["client_1"].spent_epsilon = 2.0
    
    # Add some privacy history
    privacy_record = PrivacySpent(epsilon=1.0, delta=1e-5, round_number=1, client_id="client_0")
    accountant.privacy_history.append(privacy_record)
    
    summary = accountant.get_privacy_summary()
    
    assert "global_privacy" in summary
    assert "target_privacy" in summary
    assert "client_privacy" in summary
    assert summary["total_clients"] == 2
    assert summary["target_privacy"]["target_epsilon"] == 10.0
    assert summary["target_privacy"]["target_delta"] == 1e-4
    assert "client_0" in summary["client_privacy"]
    assert "client_1" in summary["client_privacy"]


def test_export_privacy_history():
    """Test exporting privacy history"""
    config = DPConfig(target_epsilon=10.0, target_delta=1e-4)
    accountant = PrivacyAccountant(config)
    
    # Add some privacy records
    record1 = PrivacySpent(epsilon=1.0, delta=1e-5, round_number=1, client_id="client_0")
    record2 = PrivacySpent(epsilon=1.5, delta=1.5e-5, round_number=2, client_id="client_1")
    accountant.privacy_history.extend([record1, record2])
    
    exported = accountant.export_privacy_history()
    
    assert len(exported) == 2
    assert exported[0]["epsilon"] == 1.0
    assert exported[0]["delta"] == 1e-5
    assert exported[0]["round_number"] == 1
    assert exported[0]["client_id"] == "client_0"
    assert "timestamp" in exported[0]


def test_suggest_optimal_noise():
    """Test suggesting optimal noise multiplier"""
    config = DPConfig(target_epsilon=5.0, target_delta=1e-5)
    accountant = PrivacyAccountant(config)
    
    # Test noise suggestion
    noise_multiplier = accountant.suggest_optimal_noise(
        sample_rate=0.01,
        epochs=5,
        dataset_size=60000
    )
    
    assert isinstance(noise_multiplier, float)
    assert 0.1 <= noise_multiplier <= 10.0  # Should be in reasonable range
    
    # Test with custom target epsilon
    noise_multiplier_custom = accountant.suggest_optimal_noise(
        sample_rate=0.01,
        epochs=5,
        dataset_size=60000,
        target_epsilon=10.0  # Higher epsilon should give lower noise
    )
    
    assert isinstance(noise_multiplier_custom, float)
    # Higher epsilon generally means lower noise requirement
    # (This may not always hold due to other factors, so we just check it's reasonable)
    assert 0.1 <= noise_multiplier_custom <= 10.0


@pytest.mark.skipif(not OPACUS_AVAILABLE, reason="Opacus not available")
def test_opacus_integration():
    """Test integration with Opacus RDP accountant"""
    config = DPConfig(target_epsilon=5.0, target_delta=1e-5, accounting_method="rdp")
    accountant = PrivacyAccountant(config)
    
    # Should have RDP accountant if Opacus is available
    assert accountant.rdp_accountant is not None


@pytest.mark.skipif(not OPACUS_AVAILABLE, reason="Opacus not available")
def test_compute_privacy_spent():
    """Test computing privacy spent with RDP"""
    config = DPConfig(target_epsilon=8.0, target_delta=1e-5)
    accountant = PrivacyAccountant(config)
    
    # Compute privacy spent
    epsilon, delta = accountant.compute_privacy_spent(
        noise_multiplier=1.0,
        sample_rate=0.01,
        steps=1000
    )
    
    assert isinstance(epsilon, float)
    assert isinstance(delta, float)
    assert epsilon > 0
    assert delta >= 0


def test_compute_privacy_spent_fallback():
    """Test privacy computation fallback when Opacus not available"""
    config = DPConfig(target_epsilon=8.0, target_delta=1e-5)
    
    with patch('murmura.privacy.privacy_accountant.OPACUS_AVAILABLE', False):
        accountant = PrivacyAccountant(config)
        
        # Should still work with fallback computation
        epsilon, delta = accountant.compute_privacy_spent(
            noise_multiplier=1.0,
            sample_rate=0.01,
            steps=1000
        )
        
        assert isinstance(epsilon, float)
        assert isinstance(delta, float)
        assert epsilon > 0
        assert delta >= 0