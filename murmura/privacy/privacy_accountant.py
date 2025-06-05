import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from murmura.privacy.dp_config import DPConfig

try:
    from opacus.accountants import RDPAccountant  # type: ignore[import-untyped]
    from opacus.accountants.utils import get_noise_multiplier  # type: ignore[import-untyped]

    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    RDPAccountant = None


@dataclass
class PrivacySpent:
    """Record of privacy expenditure"""

    epsilon: float
    delta: float
    timestamp: datetime = field(default_factory=datetime.now)
    round_number: Optional[int] = None
    client_id: Optional[str] = None
    context: str = "training"


@dataclass
class PrivacyBudget:
    """Privacy budget tracking"""

    total_epsilon: float
    total_delta: float
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.total_epsilon - self.spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        return max(0.0, self.total_delta - self.spent_delta)

    @property
    def is_exhausted(self) -> bool:
        # Only check epsilon exhaustion for standard DP
        # Delta is typically a fixed parameter, not "spent"
        return self.spent_epsilon >= self.total_epsilon

    @property
    def utilization_percentage(self) -> float:
        return (self.spent_epsilon / self.total_epsilon) * 100.0


class PrivacyAccountant:
    """
    Comprehensive privacy accountant for federated learning with differential privacy.

    Tracks privacy expenditure across multiple clients and rounds, supporting
    both local and central differential privacy.
    """

    def __init__(self, dp_config: DPConfig):
        """
        Initialize privacy accountant.

        Args:
            dp_config: Differential privacy configuration
        """
        self.dp_config = dp_config
        self.logger = logging.getLogger("murmura.privacy_accountant")

        # Privacy budget tracking
        self.global_budget = PrivacyBudget(
            total_epsilon=dp_config.target_epsilon,
            total_delta=dp_config.target_delta or 1e-5,
        )

        # Per-client privacy tracking
        self.client_budgets: Dict[str, PrivacyBudget] = {}
        self.privacy_history: List[PrivacySpent] = []

        # Opacus integration
        self.rdp_accountant: Optional[RDPAccountant] = None
        if OPACUS_AVAILABLE and dp_config.accounting_method.value == "rdp":
            self._initialize_rdp_accountant()

        self.logger.info(
            f"Initialized privacy accountant with budget "
            f"(ε={self.global_budget.total_epsilon}, δ={self.global_budget.total_delta})"
        )

    def _initialize_rdp_accountant(self) -> None:
        """Initialize RDP accountant from Opacus"""
        try:
            if self.dp_config.alphas:
                self.rdp_accountant = RDPAccountant()
                # Set custom alphas if provided
                # Note: This might need adjustment based on Opacus version
            else:
                self.rdp_accountant = RDPAccountant()

            self.logger.info("Initialized RDP accountant")
        except Exception as e:
            self.logger.warning(f"Could not initialize RDP accountant: {e}")
            self.rdp_accountant = None

    def create_client_budget(self, client_id: str) -> None:
        """
        Create privacy budget for a specific client.

        Args:
            client_id: Unique client identifier
        """
        if client_id not in self.client_budgets:
            self.client_budgets[client_id] = PrivacyBudget(
                total_epsilon=self.dp_config.target_epsilon,
                total_delta=self.dp_config.target_delta or 1e-5,
            )
            self.logger.debug(f"Created privacy budget for client {client_id}")

    def compute_privacy_spent(
        self,
        noise_multiplier: float,
        sample_rate: float,
        steps: int,
        delta: Optional[float] = None,
        apply_amplification: bool = False,
        client_sampling_rate: Optional[float] = None,
        data_sampling_rate: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute privacy spent for given parameters with optional subsampling amplification.

        Args:
            noise_multiplier: Noise multiplier used in training
            sample_rate: Base subsampling rate
            steps: Number of training steps
            delta: Target delta (uses config default if None)
            apply_amplification: Whether to apply subsampling amplification
            client_sampling_rate: Client sampling rate (for amplification)
            data_sampling_rate: Data sampling rate (for amplification)

        Returns:
            Tuple of (epsilon, delta) spent
        """
        if delta is None:
            delta = self.dp_config.target_delta or 1e-5

        # Apply subsampling amplification if requested
        effective_sample_rate = sample_rate
        if apply_amplification and client_sampling_rate is not None and data_sampling_rate is not None:
            # Combine client and data sampling rates for amplification
            amplification_factor = client_sampling_rate * data_sampling_rate
            effective_sample_rate = sample_rate * amplification_factor
            
            self.logger.debug(
                f"Applying subsampling amplification: "
                f"client_rate={client_sampling_rate:.3f}, data_rate={data_sampling_rate:.3f}, "
                f"amplification_factor={amplification_factor:.3f}, "
                f"effective_sample_rate={effective_sample_rate:.6f} (vs original {sample_rate:.6f})"
            )

        if self.rdp_accountant and OPACUS_AVAILABLE:
            try:
                # Use Opacus RDP accountant for precise computation
                epsilon = self.rdp_accountant.get_epsilon(
                    delta=delta,
                    sample_rate=effective_sample_rate,
                    noise_multiplier=noise_multiplier,
                    steps=steps,
                )
                return epsilon, delta
            except Exception as e:
                self.logger.warning(f"RDP computation failed: {e}, using approximation")

        # Fallback to approximate computation
        # This is a simplified approximation - use Opacus for production
        sigma = noise_multiplier
        q = effective_sample_rate
        T = steps

        # Basic composition bound (not tight)
        epsilon_per_step = (q * q) / (2 * sigma * sigma)
        epsilon = epsilon_per_step * T

        self.logger.debug(
            f"Approximate privacy: ε={epsilon:.4f} for {steps} steps (σ={sigma}, q={q})"
        )

        return epsilon, delta

    def record_training_round(
        self,
        client_id: str,
        noise_multiplier: float,
        sample_rate: float,
        steps: int,
        round_number: int,
    ) -> PrivacySpent:
        """
        Record privacy expenditure for a training round.

        Args:
            client_id: Client identifier
            noise_multiplier: Noise multiplier used
            sample_rate: Subsampling rate
            steps: Number of steps in this round
            round_number: Round number

        Returns:
            PrivacySpent record
        """
        # Ensure client budget exists
        self.create_client_budget(client_id)

        # Compute privacy spent this round
        epsilon_spent, delta_spent = self.compute_privacy_spent(
            noise_multiplier=noise_multiplier, sample_rate=sample_rate, steps=steps
        )

        # Update client budget
        client_budget = self.client_budgets[client_id]
        client_budget.spent_epsilon += epsilon_spent
        # Delta is not accumulated in standard DP - it's a fixed parameter
        # We track it for reference but don't accumulate it
        client_budget.spent_delta = delta_spent

        # Update global budget (maximum across clients for worst-case)
        max_client_epsilon = max(
            budget.spent_epsilon for budget in self.client_budgets.values()
        )

        self.global_budget.spent_epsilon = max_client_epsilon
        # Global delta remains the target delta (not accumulated)

        # Create privacy record
        privacy_record = PrivacySpent(
            epsilon=epsilon_spent,
            delta=delta_spent,
            round_number=round_number,
            client_id=client_id,
            context=f"training_round_{round_number}",
        )

        self.privacy_history.append(privacy_record)

        self.logger.debug(
            f"Client {client_id} round {round_number}: "
            f"spent (ε={epsilon_spent:.4f}, δ={delta_spent:.2e}), "
            f"total (ε={client_budget.spent_epsilon:.4f}, δ={client_budget.spent_delta:.2e})"
        )

        return privacy_record

    def record_aggregation_round(
        self, num_clients: int, noise_multiplier: float, round_number: int
    ) -> PrivacySpent:
        """
        Record privacy expenditure for central aggregation.

        Args:
            num_clients: Number of participating clients
            noise_multiplier: Noise multiplier for aggregation
            round_number: Round number

        Returns:
            PrivacySpent record
        """
        # Central DP typically has different privacy analysis
        # This is a simplified version
        epsilon_spent = 1.0 / (noise_multiplier * noise_multiplier)
        delta_spent = 0.0  # Pure epsilon DP for simplicity

        # Update global budget
        self.global_budget.spent_epsilon += epsilon_spent

        privacy_record = PrivacySpent(
            epsilon=epsilon_spent,
            delta=delta_spent,
            round_number=round_number,
            client_id="central_server",
            context=f"aggregation_round_{round_number}",
        )

        self.privacy_history.append(privacy_record)

        self.logger.debug(
            f"Central aggregation round {round_number}: "
            f"spent (ε={epsilon_spent:.4f}, δ={delta_spent:.2e})"
        )

        return privacy_record

    def get_global_privacy_spent(self) -> Tuple[float, float]:
        """Get total global privacy expenditure"""
        return self.global_budget.spent_epsilon, self.global_budget.spent_delta

    def get_client_privacy_spent(self, client_id: str) -> Tuple[float, float]:
        """Get privacy expenditure for specific client"""
        if client_id not in self.client_budgets:
            return 0.0, 0.0

        budget = self.client_budgets[client_id]
        return budget.spent_epsilon, budget.spent_delta

    def is_budget_exhausted(self, client_id: Optional[str] = None) -> bool:
        """
        Check if privacy budget is exhausted.

        Args:
            client_id: Check specific client budget, or global if None

        Returns:
            True if budget is exhausted
        """
        if client_id is None:
            return self.global_budget.is_exhausted

        if client_id not in self.client_budgets:
            return False

        return self.client_budgets[client_id].is_exhausted

    def get_remaining_budget(
        self, client_id: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Get remaining privacy budget.

        Args:
            client_id: Get specific client budget, or global if None

        Returns:
            Tuple of (remaining_epsilon, remaining_delta)
        """
        if client_id is None:
            budget = self.global_budget
        else:
            if client_id not in self.client_budgets:
                return (
                    self.dp_config.target_epsilon,
                    self.dp_config.target_delta or 1e-5,
                )
            budget = self.client_budgets[client_id]

        return budget.remaining_epsilon, budget.remaining_delta

    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get comprehensive privacy summary"""
        global_eps, global_delta = self.get_global_privacy_spent()

        client_summaries = {}
        for client_id, budget in self.client_budgets.items():
            client_summaries[client_id] = {
                "spent_epsilon": budget.spent_epsilon,
                "spent_delta": budget.spent_delta,
                "remaining_epsilon": budget.remaining_epsilon,
                "remaining_delta": budget.remaining_delta,
                "utilization_percentage": budget.utilization_percentage,
                "is_exhausted": budget.is_exhausted,
            }

        return {
            "global_privacy": {
                "spent_epsilon": global_eps,
                "spent_delta": global_delta,
                "remaining_epsilon": self.global_budget.remaining_epsilon,
                "remaining_delta": self.global_budget.remaining_delta,
                "utilization_percentage": self.global_budget.utilization_percentage,
                "is_exhausted": self.global_budget.is_exhausted,
            },
            "target_privacy": {
                "target_epsilon": self.dp_config.target_epsilon,
                "target_delta": self.dp_config.target_delta,
            },
            "client_privacy": client_summaries,
            "total_rounds": len(
                set(r.round_number for r in self.privacy_history if r.round_number)
            ),
            "total_clients": len(self.client_budgets),
            "privacy_mechanism": self.dp_config.mechanism.value,
            "accounting_method": self.dp_config.accounting_method.value,
        }

    def export_privacy_history(self) -> List[Dict[str, Any]]:
        """Export privacy history for analysis"""
        return [
            {
                "epsilon": record.epsilon,
                "delta": record.delta,
                "timestamp": record.timestamp.isoformat(),
                "round_number": record.round_number,
                "client_id": record.client_id,
                "context": record.context,
            }
            for record in self.privacy_history
        ]

    def suggest_optimal_noise(
        self,
        sample_rate: float,
        epochs: int,
        dataset_size: int,
        target_epsilon: Optional[float] = None,
    ) -> float:
        """
        Suggest optimal noise multiplier for given training parameters.

        Args:
            sample_rate: Subsampling rate
            epochs: Number of epochs
            dataset_size: Size of training dataset
            target_epsilon: Target epsilon (uses config default if None)

        Returns:
            Suggested noise multiplier
        """
        if target_epsilon is None:
            target_epsilon = self.dp_config.target_epsilon

        target_delta = self.dp_config.target_delta or (1.0 / dataset_size)

        # Compute number of steps
        steps = epochs * (dataset_size // int(dataset_size * sample_rate))

        if OPACUS_AVAILABLE:
            try:
                noise_multiplier = get_noise_multiplier(
                    target_epsilon=target_epsilon,
                    target_delta=target_delta,
                    sample_rate=sample_rate,
                    steps=steps,
                )

                self.logger.info(
                    f"Suggested noise multiplier: {noise_multiplier:.3f} "
                    f"for ε={target_epsilon}, δ={target_delta:.2e}, "
                    f"{steps} steps"
                )

                return noise_multiplier
            except Exception as e:
                self.logger.warning(f"Could not compute optimal noise: {e}")

        # Fallback approximation
        # This is very rough - use Opacus in production
        noise_multiplier = np.sqrt(2 * np.log(1.25 / target_delta)) / target_epsilon
        noise_multiplier = max(
            0.1, min(10.0, noise_multiplier)
        )  # Clamp to reasonable range

        self.logger.warning(
            f"Using approximate noise multiplier: {noise_multiplier:.3f}. "
            "Install Opacus for precise computation."
        )

        return noise_multiplier
