import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class PrivacySpent:
    """Record of privacy budget consumption."""

    epsilon: float
    delta: float
    mechanism: str
    round_number: int
    timestamp: float = field(default_factory=lambda: __import__("time").time())
    additional_info: Dict[str, Any] = field(default_factory=dict)


class PrivacyBudgetExceeded(Exception):
    """Raised when privacy budget is exceeded."""

    pass


class PrivacyAccountant(ABC):
    """
    Abstract base class for privacy accounting.

    Privacy accountants track cumulative privacy loss across multiple
    mechanism invocations in federated learning.
    """

    def __init__(self, total_epsilon: float, total_delta: float = 0.0):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.privacy_history: List[PrivacySpent] = []
        self.logger = logging.getLogger(f"murmura.privacy.{self.__class__.__name__}")

    @abstractmethod
    def add_mechanism(
        self,
        mechanism_epsilon: float,
        mechanism_delta: float = 0.0,
        mechanism_name: str = "unknown",
        round_number: int = 0,
        **kwargs,
    ) -> None:
        """Add a privacy mechanism to the accountant."""
        pass

    @abstractmethod
    def get_current_privacy_spent(self) -> Tuple[float, float]:
        """Get current privacy budget consumption as (epsilon, delta)."""
        pass

    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget as (epsilon, delta)."""
        current_eps, current_delta = self.get_current_privacy_spent()
        return (
            max(0.0, self.total_epsilon - current_eps),
            max(0.0, self.total_delta - current_delta),
        )

    def is_budget_exceeded(self) -> bool:
        """Check if privacy budget has been exceeded."""
        current_eps, current_delta = self.get_current_privacy_spent()
        return current_eps > self.total_epsilon or current_delta > self.total_delta

    def get_budget_utilization(self) -> Dict[str, float]:
        """Get budget utilization as percentages."""
        current_eps, current_delta = self.get_current_privacy_spent()
        return {
            "epsilon_used_pct": min(100.0, (current_eps / self.total_epsilon) * 100),
            "delta_used_pct": min(100.0, (current_delta / self.total_delta) * 100)
            if self.total_delta > 0
            else 0.0,
        }

    def check_and_add_mechanism(
        self,
        mechanism_epsilon: float,
        mechanism_delta: float = 0.0,
        mechanism_name: str = "unknown",
        round_number: int = 0,
        **kwargs,
    ) -> None:
        """Add mechanism after checking budget constraints."""
        # Simulate adding the mechanism to check if it would exceed budget
        temp_accountant = self._create_copy()
        temp_accountant.add_mechanism(
            mechanism_epsilon, mechanism_delta, mechanism_name, round_number, **kwargs
        )

        if temp_accountant.is_budget_exceeded():
            current_eps, current_delta = self.get_current_privacy_spent()
            raise PrivacyBudgetExceeded(
                f"Adding mechanism '{mechanism_name}' would exceed privacy budget. "
                f"Current: (ε={current_eps:.4f}, δ={current_delta:.2e}), "
                f"Total: (ε={self.total_epsilon:.4f}, δ={self.total_delta:.2e}), "
                f"Mechanism: (ε={mechanism_epsilon:.4f}, δ={mechanism_delta:.2e})"
            )

        # If check passes, add the mechanism for real
        self.add_mechanism(
            mechanism_epsilon, mechanism_delta, mechanism_name, round_number, **kwargs
        )

    @abstractmethod
    def _create_copy(self) -> "PrivacyAccountant":
        """Create a copy of this accountant for budget checking."""
        pass

    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get comprehensive privacy summary."""
        current_eps, current_delta = self.get_current_privacy_spent()
        remaining_eps, remaining_delta = self.get_remaining_budget()
        utilization = self.get_budget_utilization()

        return {
            "total_budget": {"epsilon": self.total_epsilon, "delta": self.total_delta},
            "spent": {"epsilon": current_eps, "delta": current_delta},
            "remaining": {"epsilon": remaining_eps, "delta": remaining_delta},
            "utilization": utilization,
            "mechanisms_count": len(self.privacy_history),
            "budget_exceeded": self.is_budget_exceeded(),
            "history": [
                {
                    "round": entry.round_number,
                    "mechanism": entry.mechanism,
                    "epsilon": entry.epsilon,
                    "delta": entry.delta,
                    "timestamp": entry.timestamp,
                }
                for entry in self.privacy_history
            ],
        }


class BasicAccountant(PrivacyAccountant):
    """
    Basic privacy accountant using simple composition.

    Uses the basic composition theorem: (ε₁, δ₁) + (ε₂, δ₂) = (ε₁ + ε₂, δ₁ + δ₂)
    This is conservative but simple and always valid.
    """

    def __init__(self, total_epsilon: float, total_delta: float = 0.0):
        super().__init__(total_epsilon, total_delta)
        self.cumulative_epsilon = 0.0
        self.cumulative_delta = 0.0

    def add_mechanism(
        self,
        mechanism_epsilon: float,
        mechanism_delta: float = 0.0,
        mechanism_name: str = "unknown",
        round_number: int = 0,
        **kwargs,
    ) -> None:
        """Add mechanism using basic composition."""
        self.cumulative_epsilon += mechanism_epsilon
        self.cumulative_delta += mechanism_delta

        self.privacy_history.append(
            PrivacySpent(
                epsilon=mechanism_epsilon,
                delta=mechanism_delta,
                mechanism=mechanism_name,
                round_number=round_number,
                additional_info=kwargs,
            )
        )

        self.logger.debug(
            f"Added mechanism '{mechanism_name}' (ε={mechanism_epsilon:.4f}, δ={mechanism_delta:.2e}). "
            f"Total: (ε={self.cumulative_epsilon:.4f}, δ={self.cumulative_delta:.2e})"
        )

    def get_current_privacy_spent(self) -> Tuple[float, float]:
        """Get current privacy spent using basic composition."""
        return (self.cumulative_epsilon, self.cumulative_delta)

    def _create_copy(self) -> "BasicAccountant":
        """Create a copy for budget checking."""
        copy = BasicAccountant(self.total_epsilon, self.total_delta)
        copy.cumulative_epsilon = self.cumulative_epsilon
        copy.cumulative_delta = self.cumulative_delta
        copy.privacy_history = self.privacy_history.copy()
        return copy


class RDPAccountant(PrivacyAccountant):
    """
    Rényi Differential Privacy (RDP) accountant.

    Provides tighter composition bounds for iterative algorithms like SGD.
    Tracks privacy loss at multiple orders α and converts to (ε, δ)-DP when needed.
    """

    def __init__(
        self,
        total_epsilon: float,
        total_delta: float = 1e-5,
        orders: Optional[List[float]] = None,
    ):
        super().__init__(total_epsilon, total_delta)

        # Default RDP orders following TensorFlow Privacy
        if orders is None:
            orders = [1.1] + list(range(2, 65))
        self.orders = orders

        # Track RDP at each order
        self.rdp_eps = {order: 0.0 for order in orders}

    def add_mechanism(
        self,
        mechanism_epsilon: float,
        mechanism_delta: float = 0.0,
        mechanism_name: str = "gaussian_mechanism",
        round_number: int = 0,
        noise_multiplier: Optional[float] = None,
        sampling_rate: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Add mechanism to RDP accountant.

        For Gaussian mechanism, provide noise_multiplier and optionally sampling_rate
        for privacy amplification.
        """

        if mechanism_name == "gaussian_mechanism" and noise_multiplier is not None:
            # Compute RDP for Gaussian mechanism
            rdp_eps_per_order = self._compute_gaussian_rdp(
                noise_multiplier, sampling_rate
            )

            # Add to cumulative RDP
            for order, rdp_eps in rdp_eps_per_order.items():
                if order in self.rdp_eps:
                    self.rdp_eps[order] += rdp_eps

        else:
            # Fallback: convert (ε, δ)-DP to RDP (conservative)
            for order in self.orders:
                if order > 1:
                    # Conservative conversion: RDP(α, ε + δ)
                    self.rdp_eps[order] += mechanism_epsilon + mechanism_delta

        self.privacy_history.append(
            PrivacySpent(
                epsilon=mechanism_epsilon,
                delta=mechanism_delta,
                mechanism=mechanism_name,
                round_number=round_number,
                additional_info={
                    "noise_multiplier": noise_multiplier,
                    "sampling_rate": sampling_rate,
                    **kwargs,
                },
            )
        )

        # Log current privacy spent
        current_eps, current_delta = self.get_current_privacy_spent()
        self.logger.debug(
            f"Added RDP mechanism '{mechanism_name}'. "
            f"Current (ε, δ): ({current_eps:.4f}, {current_delta:.2e})"
        )

    def _compute_gaussian_rdp(
        self, noise_multiplier: float, sampling_rate: Optional[float] = None
    ) -> Dict[float, float]:
        """Compute RDP for Gaussian mechanism, optionally with subsampling."""
        rdp_eps = {}

        for order in self.orders:
            if order == 1.0:
                # RDP at order 1 is undefined, skip
                continue

            # Basic Gaussian RDP: ε(α) = α / (2 * σ²)
            base_rdp = order / (2 * noise_multiplier**2)

            # Apply privacy amplification if subsampling
            if sampling_rate is not None and sampling_rate < 1.0:
                # Simplified amplification (exact formula is more complex)
                amplified_rdp = base_rdp * sampling_rate * sampling_rate
                rdp_eps[order] = amplified_rdp
            else:
                rdp_eps[order] = base_rdp

        return rdp_eps

    def get_current_privacy_spent(self) -> Tuple[float, float]:
        """Convert current RDP to (ε, δ)-DP."""
        if not self.rdp_eps:
            return (0.0, 0.0)

        # Find optimal order for conversion
        best_epsilon = float("inf")

        for order, rdp_eps in self.rdp_eps.items():
            if order <= 1 or rdp_eps <= 0:
                continue

            # RDP to (ε, δ)-DP conversion: ε = ρ + log(1/δ)/(α-1)
            if self.total_delta > 0:
                epsilon = rdp_eps + math.log(1 / self.total_delta) / (order - 1)
                best_epsilon = min(best_epsilon, epsilon)

        return (best_epsilon if best_epsilon != float("inf") else 0.0, self.total_delta)

    def _create_copy(self) -> "RDPAccountant":
        """Create a copy for budget checking."""
        copy = RDPAccountant(self.total_epsilon, self.total_delta, self.orders)
        copy.rdp_eps = self.rdp_eps.copy()
        copy.privacy_history = self.privacy_history.copy()
        return copy


class ZCDPAccountant(PrivacyAccountant):
    """
    Zero-Concentrated Differential Privacy (zCDP) accountant.

    Simpler than RDP with elegant composition: ρ₁ + ρ₂ = ρ₁ + ρ₂
    Conversion to (ε, δ)-DP: ε = ρ + 2√(ρ ln(1/δ))
    """

    def __init__(self, total_epsilon: float, total_delta: float = 1e-5):
        super().__init__(total_epsilon, total_delta)
        self.cumulative_rho = 0.0

    def add_mechanism(
        self,
        mechanism_epsilon: float,
        mechanism_delta: float = 0.0,
        mechanism_name: str = "gaussian_mechanism",
        round_number: int = 0,
        noise_multiplier: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Add mechanism to zCDP accountant."""

        if mechanism_name == "gaussian_mechanism" and noise_multiplier is not None:
            # For Gaussian mechanism: ρ = 1/(2σ²)
            rho = 1.0 / (2 * noise_multiplier**2)
        else:
            # Conservative conversion from (ε, δ)-DP to zCDP
            # This is an approximation - exact conversion is complex
            rho = mechanism_epsilon**2 / 4

        self.cumulative_rho += rho

        self.privacy_history.append(
            PrivacySpent(
                epsilon=mechanism_epsilon,
                delta=mechanism_delta,
                mechanism=mechanism_name,
                round_number=round_number,
                additional_info={
                    "rho": rho,
                    "noise_multiplier": noise_multiplier,
                    **kwargs,
                },
            )
        )

        current_eps, current_delta = self.get_current_privacy_spent()
        self.logger.debug(
            f"Added zCDP mechanism '{mechanism_name}' (ρ={rho:.4f}). "
            f"Current (ε, δ): ({current_eps:.4f}, {current_delta:.2e})"
        )

    def get_current_privacy_spent(self) -> Tuple[float, float]:
        """Convert zCDP to (ε, δ)-DP."""
        if self.cumulative_rho <= 0 or self.total_delta <= 0:
            return (0.0, 0.0)

        # zCDP to (ε, δ)-DP: ε = ρ + 2√(ρ ln(1/δ))
        epsilon = self.cumulative_rho + 2 * math.sqrt(
            self.cumulative_rho * math.log(1 / self.total_delta)
        )
        return (epsilon, self.total_delta)

    def get_current_rho(self) -> float:
        """Get current zCDP parameter ρ."""
        return self.cumulative_rho

    def _create_copy(self) -> "ZCDPAccountant":
        """Create a copy for budget checking."""
        copy = ZCDPAccountant(self.total_epsilon, self.total_delta)
        copy.cumulative_rho = self.cumulative_rho
        copy.privacy_history = self.privacy_history.copy()
        return copy


def create_privacy_accountant(
    accountant_type: str, total_epsilon: float, total_delta: float = 1e-5, **kwargs
) -> PrivacyAccountant:
    """
    Factory function to create privacy accountants.

    Args:
        accountant_type: Type of accountant ('basic', 'rdp', 'zcdp')
        total_epsilon: Total privacy budget (ε)
        total_delta: Total failure probability (δ)
        **kwargs: Additional arguments for specific accountants

    Returns:
        Configured privacy accountant
    """
    accountant_type = accountant_type.lower()

    if accountant_type == "basic":
        return BasicAccountant(total_epsilon, total_delta)
    elif accountant_type == "rdp":
        orders = kwargs.get("orders", None)
        return RDPAccountant(total_epsilon, total_delta, orders)
    elif accountant_type == "zcdp":
        return ZCDPAccountant(total_epsilon, total_delta)
    else:
        raise ValueError(
            f"Unknown accountant type: {accountant_type}. "
            f"Supported types: 'basic', 'rdp', 'zcdp'"
        )


class PrivacyMonitor:
    """
    Real-time privacy budget monitoring for production systems.

    Provides alerts, logging, and telemetry for privacy budget consumption.
    """

    def __init__(self, accountant: PrivacyAccountant, warning_threshold: float = 0.8):
        self.accountant = accountant
        self.warning_threshold = warning_threshold
        self.logger = logging.getLogger("murmura.privacy.monitor")
        self.alerts_sent = set()  # Prevent duplicate alerts

    def check_budget_status(self) -> Dict[str, Any]:
        """Check current budget status and trigger alerts if needed."""
        utilization = self.accountant.get_budget_utilization()
        status = {"utilization": utilization, "alerts": []}

        epsilon_pct = utilization["epsilon_used_pct"]
        delta_pct = utilization["delta_used_pct"]

        # Check for warnings
        if epsilon_pct >= self.warning_threshold * 100:
            alert = (
                f"Privacy budget warning: {epsilon_pct:.1f}% of epsilon budget consumed"
            )
            if "epsilon_warning" not in self.alerts_sent:
                self.logger.warning(alert)
                self.alerts_sent.add("epsilon_warning")
            status["alerts"].append(alert)

        if delta_pct >= self.warning_threshold * 100:
            alert = f"Privacy budget warning: {delta_pct:.1f}% of delta budget consumed"
            if "delta_warning" not in self.alerts_sent:
                self.logger.warning(alert)
                self.alerts_sent.add("delta_warning")
            status["alerts"].append(alert)

        # Check for budget exceeded
        if self.accountant.is_budget_exceeded():
            alert = "CRITICAL: Privacy budget exceeded!"
            if "budget_exceeded" not in self.alerts_sent:
                self.logger.error(alert)
                self.alerts_sent.add("budget_exceeded")
            status["alerts"].append(alert)

        return status

    def log_mechanism_added(self, mechanism_name: str, round_number: int) -> None:
        """Log when a new mechanism is added."""
        utilization = self.accountant.get_budget_utilization()
        self.logger.info(
            f"Privacy mechanism '{mechanism_name}' added for round {round_number}. "
            f"Budget utilization: ε={utilization['epsilon_used_pct']:.1f}%, "
            f"δ={utilization['delta_used_pct']:.1f}%"
        )

    def get_telemetry(self) -> Dict[str, Any]:
        """Get telemetry data for monitoring dashboards."""
        summary = self.accountant.get_privacy_summary()
        return {
            "privacy_budget": summary,
            "recent_mechanisms": summary["history"][-10:],  # Last 10 mechanisms
            "alerts_count": len(self.alerts_sent),
            "monitoring_status": "healthy"
            if not self.accountant.is_budget_exceeded()
            else "critical",
        }
