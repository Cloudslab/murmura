import pytest
from pydantic import ValidationError

from murmura.privacy.dp_config import DPConfig, DPMechanism, DPAccountingMethod


def test_default_initialization():
    """Test default initialization of DPConfig"""
    config = DPConfig()
    
    # Check default values
    assert config.target_epsilon == 8.0
    assert config.target_delta == 1e-5
    assert config.max_grad_norm == 1.0
    assert config.mechanism == DPMechanism.GAUSSIAN
    assert config.accounting_method == DPAccountingMethod.RDP


def test_custom_initialization():
    """Test initialization with custom values"""
    config = DPConfig(
        target_epsilon=5.0,
        target_delta=1e-6,
        max_grad_norm=2.0,
        mechanism=DPMechanism.LAPLACE,
        accounting_method=DPAccountingMethod.GDP
    )
    
    assert config.target_epsilon == 5.0
    assert config.target_delta == 1e-6
    assert config.max_grad_norm == 2.0
    assert config.mechanism == DPMechanism.LAPLACE
    assert config.accounting_method == DPAccountingMethod.GDP


def test_epsilon_validation():
    """Test validation for target_epsilon"""
    # Valid epsilon values
    config = DPConfig(target_epsilon=1.0)
    assert config.target_epsilon == 1.0
    
    config = DPConfig(target_epsilon=50.0)
    assert config.target_epsilon == 50.0
    
    # Invalid epsilon values
    with pytest.raises(ValidationError):
        DPConfig(target_epsilon=0.05)  # Below minimum
    
    with pytest.raises(ValidationError):
        DPConfig(target_epsilon=100.0)  # Above maximum


def test_delta_validation():
    """Test validation for target_delta"""
    # Valid delta values
    config = DPConfig(target_delta=1e-8)
    assert config.target_delta == 1e-8
    
    config = DPConfig(target_delta=1e-3)
    assert config.target_delta == 1e-3
    
    # Invalid delta values
    with pytest.raises(ValidationError):
        DPConfig(target_delta=1e-9)  # Below minimum
    
    with pytest.raises(ValidationError):
        DPConfig(target_delta=1e-2)  # Above maximum


def test_max_grad_norm_validation():
    """Test validation for max_grad_norm"""
    # Valid max_grad_norm values
    config = DPConfig(max_grad_norm=0.1)
    assert config.max_grad_norm == 0.1
    
    config = DPConfig(max_grad_norm=10.0)
    assert config.max_grad_norm == 10.0
    
    # Invalid max_grad_norm values
    with pytest.raises(ValidationError):
        DPConfig(max_grad_norm=0.05)  # Below minimum
    
    with pytest.raises(ValidationError):
        DPConfig(max_grad_norm=20.0)  # Above maximum


def test_mechanism_enum_values():
    """Test DPMechanism enum values"""
    assert DPMechanism.GAUSSIAN == "gaussian"
    assert DPMechanism.LAPLACE == "laplace"
    assert DPMechanism.UNIFORM == "uniform"
    
    # Test all mechanisms can be used
    for mechanism in DPMechanism:
        config = DPConfig(mechanism=mechanism)
        assert config.mechanism == mechanism


def test_accounting_method_enum_values():
    """Test DPAccountingMethod enum values"""
    assert DPAccountingMethod.RDP == "rdp"
    assert DPAccountingMethod.GDP == "gdp"
    assert DPAccountingMethod.PRIV_ACCOUNTANT == "prv"
    
    # Test all accounting methods can be used
    for method in DPAccountingMethod:
        config = DPConfig(accounting_method=method)
        assert config.accounting_method == method


def test_model_dump():
    """Test the model_dump method for creating config dictionaries"""
    config = DPConfig(
        target_epsilon=5.0,
        mechanism=DPMechanism.LAPLACE
    )
    
    config_dict = config.model_dump()
    
    assert isinstance(config_dict, dict)
    assert config_dict["target_epsilon"] == 5.0
    assert config_dict["mechanism"] == "laplace"


def test_model_validation_integration():
    """Test integration with model validation"""
    # Valid configuration
    config_dict = {
        "target_epsilon": 3.0,
        "target_delta": 1e-6,
        "max_grad_norm": 1.5,
        "mechanism": "gaussian",
        "accounting_method": "rdp"
    }
    
    config = DPConfig.model_validate(config_dict)
    assert config.target_epsilon == 3.0
    assert config.target_delta == 1e-6
    assert config.max_grad_norm == 1.5
    assert config.mechanism == DPMechanism.GAUSSIAN
    assert config.accounting_method == DPAccountingMethod.RDP


def test_config_equality():
    """Test that configs with same values are equal"""
    config1 = DPConfig(target_epsilon=5.0, mechanism=DPMechanism.LAPLACE)
    config2 = DPConfig(target_epsilon=5.0, mechanism=DPMechanism.LAPLACE)
    
    assert config1 == config2


def test_preset_configurations():
    """Test that preset configurations are reasonable for different datasets"""
    # MNIST-style configuration (smaller dataset)
    mnist_config = DPConfig(
        target_epsilon=8.0,
        target_delta=1e-5,
        max_grad_norm=1.0
    )
    
    assert mnist_config.target_epsilon == 8.0
    assert mnist_config.target_delta == 1e-5
    
    # Medical dataset configuration (more privacy required)
    medical_config = DPConfig(
        target_epsilon=3.0,
        target_delta=1e-6,
        max_grad_norm=0.5
    )
    
    assert medical_config.target_epsilon == 3.0
    assert medical_config.target_delta == 1e-6
    assert medical_config.max_grad_norm == 0.5


def test_noise_multiplier_validation():
    """Test validation for noise_multiplier"""
    # Valid noise_multiplier values
    config = DPConfig(noise_multiplier=0.5)
    assert config.noise_multiplier == 0.5
    
    config = DPConfig(noise_multiplier=5.0)
    assert config.noise_multiplier == 5.0
    
    # Invalid noise_multiplier values
    with pytest.raises(ValidationError):
        DPConfig(noise_multiplier=0.05)  # Below minimum
    
    with pytest.raises(ValidationError):
        DPConfig(noise_multiplier=15.0)  # Above maximum


def test_sample_rate_validation():
    """Test validation for sample_rate"""
    # Valid sample_rate values
    config = DPConfig(sample_rate=0.1)
    assert config.sample_rate == 0.1
    
    config = DPConfig(sample_rate=1.0)
    assert config.sample_rate == 1.0
    
    # Invalid sample_rate values
    with pytest.raises(ValidationError):
        DPConfig(sample_rate=0.0005)  # Below minimum
    
    with pytest.raises(ValidationError):
        DPConfig(sample_rate=1.5)  # Above maximum


def test_federated_learning_flags():
    """Test federated learning specific flags"""
    # Test enable_client_dp and enable_central_dp
    config = DPConfig(enable_client_dp=True, enable_central_dp=False)
    assert config.enable_client_dp is True
    assert config.enable_central_dp is False
    
    config = DPConfig(enable_client_dp=False, enable_central_dp=True)
    assert config.enable_client_dp is False
    assert config.enable_central_dp is True


def test_federated_learning_validation():
    """Test that at least one DP mode must be enabled"""
    # This should fail validation
    with pytest.raises(ValidationError):
        DPConfig(enable_client_dp=False, enable_central_dp=False)


def test_model_validator():
    """Test the model validator catches invalid configurations"""
    # Invalid delta >= 1.0
    with pytest.raises(ValidationError):
        DPConfig(target_delta=1.5)
    
    # Invalid negative noise multiplier (handled by pydantic)
    with pytest.raises(ValidationError):
        DPConfig(noise_multiplier=-0.5)


def test_privacy_budget_methods():
    """Test privacy budget checking methods"""
    config = DPConfig(target_epsilon=5.0, target_delta=1e-5)
    
    # Test privacy spent message
    message = config.get_privacy_spent_message(3.0, 1e-6)
    assert "ε=3.000" in message
    assert "δ=1.00e-06" in message
    assert "ε=5.0" in message
    
    # Test privacy exhaustion
    assert config.is_privacy_exhausted(6.0) is True  # Exceeds budget
    assert config.is_privacy_exhausted(4.0) is False  # Within budget
    assert config.is_privacy_exhausted(5.0) is True  # Exactly at budget


def test_class_method_factories():
    """Test class method factory functions"""
    # Test MNIST factory
    mnist_config = DPConfig.create_for_mnist()
    assert mnist_config.target_epsilon == 8.0
    assert mnist_config.target_delta == 1e-5
    assert mnist_config.max_grad_norm == 1.0
    assert mnist_config.enable_client_dp is True
    assert mnist_config.enable_central_dp is False
    
    # Test HAM10000 factory
    ham10000_config = DPConfig.create_for_ham10000()
    assert ham10000_config.target_epsilon == 10.0
    assert ham10000_config.target_delta == 1e-4
    assert ham10000_config.max_grad_norm == 1.2
    
    # Test high privacy factory
    high_privacy_config = DPConfig.create_high_privacy()
    assert high_privacy_config.target_epsilon == 1.0
    assert high_privacy_config.target_delta == 1e-6
    assert high_privacy_config.max_grad_norm == 0.8


def test_subsampling_factory():
    """Test subsampling configuration factory"""
    config = DPConfig.create_with_subsampling(
        target_epsilon=3.0,
        client_sampling_rate=0.2,
        data_sampling_rate=0.5,
        dataset_size=10000
    )
    
    assert config.target_epsilon == 3.0
    assert config.target_delta == 1.0 / 10000  # 1e-4
    assert config.client_sampling_rate == 0.2
    assert config.data_sampling_rate == 0.5
    assert config.use_amplification_by_subsampling is True


def test_amplification_methods():
    """Test privacy amplification calculation methods"""
    # Without amplification
    config = DPConfig(use_amplification_by_subsampling=False, sample_rate=0.5)
    assert config.get_amplified_sample_rate() == 0.5
    assert config.get_amplification_factor() == 1.0
    
    # With amplification
    config = DPConfig(
        use_amplification_by_subsampling=True,
        client_sampling_rate=0.2,
        data_sampling_rate=0.5
    )
    assert config.get_amplified_sample_rate() == 0.1  # 0.2 * 0.5
    assert config.get_amplification_factor() == 0.1


def test_advanced_parameters():
    """Test advanced configuration parameters"""
    config = DPConfig(
        alphas=[1.5, 2.0, 5.0, 10.0],
        auto_tune_noise=False,
        strict_privacy_check=False,
        secure_mode=True
    )
    
    assert config.alphas == [1.5, 2.0, 5.0, 10.0]
    assert config.auto_tune_noise is False
    assert config.strict_privacy_check is False
    assert config.secure_mode is True


def test_subsampling_rate_validation():
    """Test validation for subsampling rates"""
    # Valid rates
    config = DPConfig(client_sampling_rate=0.5, data_sampling_rate=0.3)
    assert config.client_sampling_rate == 0.5
    assert config.data_sampling_rate == 0.3
    
    # Invalid rates
    with pytest.raises(ValidationError):
        DPConfig(client_sampling_rate=0.005)  # Below minimum
    
    with pytest.raises(ValidationError):
        DPConfig(data_sampling_rate=1.5)  # Above maximum