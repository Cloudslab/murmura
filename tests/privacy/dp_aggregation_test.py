import pytest
import numpy as np
from unittest.mock import Mock, patch

from murmura.privacy.dp_aggregation import DPFedAvg, DPSecureAggregation, DPTrimmedMean
from murmura.privacy.dp_config import DPConfig, DPMechanism


class TestDPFedAvg:
    """Test cases for DPFedAvg differential privacy aggregation strategy."""

    def test_init_with_central_dp_enabled(self):
        """Test initialization with central DP enabled."""
        dp_config = DPConfig(
            enable_central_dp=True,
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=1.0
        )
        strategy = DPFedAvg(dp_config)
        
        assert strategy.dp_config == dp_config
        assert strategy.round_count == 0
        assert strategy.logger.name == "murmura.dp_aggregation.dp_fedavg"

    def test_init_with_central_dp_disabled_logs_warning(self, caplog):
        """Test initialization logs warning when central DP is disabled."""
        dp_config = DPConfig(
            enable_central_dp=False,
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0
        )
        strategy = DPFedAvg(dp_config)
        
        assert "Central DP is disabled" in caplog.text
        assert "will perform regular FedAvg" in caplog.text

    def test_aggregate_empty_parameters_raises_error(self):
        """Test that empty parameters list raises ValueError."""
        dp_config = DPConfig(enable_central_dp=True, epsilon=1.0, delta=1e-5)
        strategy = DPFedAvg(dp_config)
        
        with pytest.raises(ValueError, match="Empty parameters list"):
            strategy.aggregate([])

    def test_aggregate_mismatched_weights_raises_error(self):
        """Test that mismatched weights and parameters length raises error."""
        dp_config = DPConfig(enable_central_dp=True, epsilon=1.0, delta=1e-5)
        strategy = DPFedAvg(dp_config)
        
        params = [{"layer1": np.array([1.0, 2.0])}, {"layer1": np.array([3.0, 4.0])}]
        weights = [0.5]  # Wrong length
        
        with pytest.raises(ValueError, match="Weights and parameters list must have same length"):
            strategy.aggregate(params, weights)

    def test_aggregate_without_central_dp(self):
        """Test aggregation without central DP (regular FedAvg)."""
        dp_config = DPConfig(enable_central_dp=False, epsilon=1.0, delta=1e-5)
        strategy = DPFedAvg(dp_config)
        
        params = [
            {"layer1": np.array([1.0, 2.0])},
            {"layer1": np.array([3.0, 4.0])}
        ]
        
        result = strategy.aggregate(params)
        
        expected = np.array([2.0, 3.0])  # Simple average
        assert np.allclose(result["layer1"], expected)
        assert strategy.round_count == 1

    def test_aggregate_with_custom_weights(self):
        """Test aggregation with custom weights."""
        dp_config = DPConfig(enable_central_dp=False, epsilon=1.0, delta=1e-5)
        strategy = DPFedAvg(dp_config)
        
        params = [
            {"layer1": np.array([1.0, 2.0])},
            {"layer1": np.array([3.0, 4.0])}
        ]
        weights = [0.3, 0.7]
        
        result = strategy.aggregate(params, weights)
        
        # Weighted average: 0.3 * [1,2] + 0.7 * [3,4] = [2.4, 3.4]
        expected = np.array([2.4, 3.4])
        assert np.allclose(result["layer1"], expected)

    def test_aggregate_with_round_number(self):
        """Test aggregation with explicit round number."""
        dp_config = DPConfig(enable_central_dp=False, epsilon=1.0, delta=1e-5)
        strategy = DPFedAvg(dp_config)
        
        params = [{"layer1": np.array([1.0, 2.0])}]
        strategy.aggregate(params, round_number=5)
        
        assert strategy.round_count == 5

    @patch('numpy.random.normal')
    def test_aggregate_with_central_dp_gaussian(self, mock_normal):
        """Test aggregation with central DP using Gaussian mechanism."""
        mock_normal.return_value = np.array([0.1, 0.2])
        
        dp_config = DPConfig(
            enable_central_dp=True,
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=1.0,
            mechanism=DPMechanism.GAUSSIAN
        )
        strategy = DPFedAvg(dp_config)
        
        params = [{"layer1": np.array([1.0, 2.0])}]
        result = strategy.aggregate(params)
        
        # Should add noise to the average
        expected = np.array([1.1, 2.2])  # [1.0, 2.0] + [0.1, 0.2]
        assert np.allclose(result["layer1"], expected)
        
        # Verify noise was called with correct parameters
        mock_normal.assert_called_once()
        args = mock_normal.call_args[0]
        assert args[0] == 0  # mean
        assert args[1] == 2.0  # noise_scale = noise_multiplier * sensitivity = 1.0 * 2.0

    @patch('numpy.random.laplace')
    def test_aggregate_with_central_dp_laplace(self, mock_laplace):
        """Test aggregation with central DP using Laplace mechanism."""
        mock_laplace.return_value = np.array([0.1, 0.2])
        
        dp_config = DPConfig(
            enable_central_dp=True,
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=1.0,
            mechanism=DPMechanism.LAPLACE
        )
        strategy = DPFedAvg(dp_config)
        
        params = [{"layer1": np.array([1.0, 2.0])}]
        result = strategy.aggregate(params)
        
        expected = np.array([1.1, 2.2])
        assert np.allclose(result["layer1"], expected)

    def test_add_dp_noise_disabled(self):
        """Test _add_dp_noise when central DP is disabled."""
        dp_config = DPConfig(enable_central_dp=False, epsilon=1.0, delta=1e-5)
        strategy = DPFedAvg(dp_config)
        
        params = np.array([1.0, 2.0])
        result = strategy._add_dp_noise(params, "test_param")
        
        assert np.array_equal(result, params)  # No noise added


class TestDPSecureAggregation:
    """Test cases for DPSecureAggregation for decentralized learning."""

    def test_init(self):
        """Test initialization."""
        dp_config = DPConfig(epsilon=1.0, delta=1e-5)
        strategy = DPSecureAggregation(dp_config)
        
        assert strategy.dp_config == dp_config
        assert strategy.round_count == 0
        assert strategy.logger.name == "murmura.dp_aggregation.dp_secure"

    def test_aggregate_empty_parameters_raises_error(self):
        """Test that empty parameters list raises ValueError."""
        dp_config = DPConfig(epsilon=1.0, delta=1e-5)
        strategy = DPSecureAggregation(dp_config)
        
        with pytest.raises(ValueError, match="Empty parameters list"):
            strategy.aggregate([])

    def test_aggregate_without_dp(self):
        """Test aggregation without DP noise."""
        dp_config = DPConfig(enable_central_dp=False, epsilon=1.0, delta=1e-5)
        strategy = DPSecureAggregation(dp_config)
        
        params = [
            {"layer1": np.array([1.0, 2.0])},
            {"layer1": np.array([3.0, 4.0])}
        ]
        
        result = strategy.aggregate(params)
        
        expected = np.array([2.0, 3.0])  # Simple average
        assert np.allclose(result["layer1"], expected)

    @patch('numpy.random.normal')
    def test_aggregate_with_dp_noise(self, mock_normal):
        """Test aggregation with DP noise for decentralized setting."""
        mock_normal.return_value = np.array([0.1, 0.2])
        
        dp_config = DPConfig(
            enable_central_dp=True,  # Using this flag for aggregation DP
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=1.0
        )
        strategy = DPSecureAggregation(dp_config)
        
        params = [
            {"layer1": np.array([1.0, 2.0])},
            {"layer1": np.array([3.0, 4.0])}
        ]
        
        result = strategy.aggregate(params)
        
        # Should have noise added
        assert not np.array_equal(result["layer1"], np.array([2.0, 3.0]))

    def test_add_decentralized_dp_noise_scaling(self):
        """Test that noise scaling works correctly for different numbers of neighbors."""
        dp_config = DPConfig(
            enable_central_dp=True,
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=1.0
        )
        strategy = DPSecureAggregation(dp_config)
        
        params = np.array([1.0, 2.0])
        
        # Test with different neighbor counts
        result_2_neighbors = strategy._add_decentralized_dp_noise(params, "test", 2)
        result_4_neighbors = strategy._add_decentralized_dp_noise(params, "test", 4)
        
        # Both should be different from original (noise added)
        assert not np.array_equal(result_2_neighbors, params)
        assert not np.array_equal(result_4_neighbors, params)


class TestDPTrimmedMean:
    """Test cases for DPTrimmedMean aggregation strategy."""

    def test_init(self):
        """Test initialization with valid trim ratio."""
        dp_config = DPConfig(epsilon=1.0, delta=1e-5)
        strategy = DPTrimmedMean(dp_config, trim_ratio=0.2)
        
        assert strategy.dp_config == dp_config
        assert strategy.trim_ratio == 0.2
        assert strategy.round_count == 0

    def test_init_clamps_trim_ratio(self):
        """Test that trim ratio is clamped to valid range."""
        dp_config = DPConfig(epsilon=1.0, delta=1e-5)
        
        # Test too high
        strategy_high = DPTrimmedMean(dp_config, trim_ratio=0.8)
        assert strategy_high.trim_ratio == 0.5
        
        # Test negative
        strategy_neg = DPTrimmedMean(dp_config, trim_ratio=-0.1)
        assert strategy_neg.trim_ratio == 0.0

    def test_aggregate_empty_parameters_raises_error(self):
        """Test that empty parameters list raises ValueError."""
        dp_config = DPConfig(epsilon=1.0, delta=1e-5)
        strategy = DPTrimmedMean(dp_config)
        
        with pytest.raises(ValueError, match="Empty parameters list"):
            strategy.aggregate([])

    def test_aggregate_few_clients_fallback(self, caplog):
        """Test fallback to simple average when too few clients."""
        dp_config = DPConfig(enable_central_dp=False, epsilon=1.0, delta=1e-5)
        strategy = DPTrimmedMean(dp_config, trim_ratio=0.1)
        
        params = [
            {"layer1": np.array([1.0, 2.0])},
            {"layer1": np.array([3.0, 4.0])}
        ]
        
        result = strategy.aggregate(params)
        
        assert "Trimmed mean requires at least 3 clients" in caplog.text
        assert "Falling back to simple average" in caplog.text
        assert np.allclose(result["layer1"], np.array([2.0, 3.0]))

    def test_compute_trimmed_mean_no_trimming(self):
        """Test trimmed mean computation when no trimming needed."""
        dp_config = DPConfig(epsilon=1.0, delta=1e-5)
        strategy = DPTrimmedMean(dp_config, trim_ratio=0.0)
        
        param_stack = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = strategy._compute_trimmed_mean(param_stack)
        
        expected = np.array([3.0, 4.0])  # Regular mean
        assert np.allclose(result, expected)

    def test_compute_trimmed_mean_with_trimming(self):
        """Test trimmed mean computation with actual trimming."""
        dp_config = DPConfig(epsilon=1.0, delta=1e-5)
        strategy = DPTrimmedMean(dp_config, trim_ratio=0.2)
        
        # Create params where trimming makes a difference
        param_stack = np.array([
            [1.0, 1.0],   # Will be trimmed (lowest)
            [2.0, 2.0], 
            [3.0, 3.0],
            [4.0, 4.0],
            [10.0, 10.0]  # Will be trimmed (highest)
        ])
        
        result = strategy._compute_trimmed_mean(param_stack)
        
        # Should average middle 3 values: [2,2], [3,3], [4,4] -> [3,3]
        expected = np.array([3.0, 3.0])
        assert np.allclose(result, expected)

    def test_aggregate_multiple_clients_no_dp(self):
        """Test aggregation with multiple clients without DP."""
        dp_config = DPConfig(enable_central_dp=False, epsilon=1.0, delta=1e-5)
        strategy = DPTrimmedMean(dp_config, trim_ratio=0.1)
        
        params = [
            {"layer1": np.array([1.0, 2.0])},
            {"layer1": np.array([2.0, 3.0])},
            {"layer1": np.array([3.0, 4.0])},
            {"layer1": np.array([4.0, 5.0])},
            {"layer1": np.array([5.0, 6.0])}
        ]
        
        result = strategy.aggregate(params)
        
        # With 5 clients and trim_ratio=0.1, should trim 0 (int(5*0.1)=0)
        # So result should be regular mean: [3.0, 4.0]
        expected = np.array([3.0, 4.0])
        assert np.allclose(result["layer1"], expected)

    @patch('numpy.random.normal')
    def test_aggregate_with_dp_noise(self, mock_normal):
        """Test aggregation with DP noise."""
        mock_normal.return_value = np.array([0.1, 0.2])
        
        dp_config = DPConfig(
            enable_central_dp=True,
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=1.0
        )
        strategy = DPTrimmedMean(dp_config, trim_ratio=0.1)
        
        params = [
            {"layer1": np.array([1.0, 2.0])},
            {"layer1": np.array([2.0, 3.0])},
            {"layer1": np.array([3.0, 4.0])}
        ]
        
        result = strategy.aggregate(params)
        
        # Should have noise added to the trimmed mean
        assert not np.array_equal(result["layer1"], np.array([2.0, 3.0]))

    def test_simple_average_with_dp_fallback(self):
        """Test the simple average fallback method."""
        dp_config = DPConfig(enable_central_dp=False, epsilon=1.0, delta=1e-5)
        strategy = DPTrimmedMean(dp_config)
        
        params = [
            {"layer1": np.array([1.0, 2.0])},
            {"layer1": np.array([3.0, 4.0])}
        ]
        
        result = strategy._simple_average_with_dp(params)
        
        expected = np.array([2.0, 3.0])
        assert np.allclose(result["layer1"], expected)

    def test_add_dp_noise_sensitivity_scaling(self):
        """Test that DP noise sensitivity is scaled for trimmed mean."""
        dp_config = DPConfig(
            enable_central_dp=True,
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=1.0
        )
        strategy = DPTrimmedMean(dp_config, trim_ratio=0.2)
        
        params = np.array([1.0, 2.0])
        result = strategy._add_dp_noise(params, "test_param")
        
        # Should add noise (result different from input)
        assert not np.array_equal(result, params)