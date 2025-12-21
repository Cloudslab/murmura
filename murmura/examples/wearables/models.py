"""Evidential deep learning models for wearable sensor datasets.

These models implement Dirichlet-based evidential classification, outputting
concentration parameters that enable epistemic-aleatoric uncertainty decomposition
for trust-aware aggregation in decentralized federated learning.

Reference:
    Sensoy et al. "Evidential Deep Learning to Quantify Classification Uncertainty."
    NeurIPS 2018.
"""

from typing import Callable, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """Evidential output head that produces Dirichlet concentration parameters.

    Instead of softmax probabilities, outputs non-negative evidence values that
    parameterize a Dirichlet distribution over class probabilities.

    Evidence e_k = exp(z_k) ensures non-negativity.
    Dirichlet parameters α_k = e_k + 1 ensure validity.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing Dirichlet concentration parameters.

        Args:
            x: Input features of shape (batch_size, in_features)

        Returns:
            Dirichlet alpha parameters of shape (batch_size, num_classes)
        """
        logits = self.fc(x)
        # Softplus for numerical stability (alternative to exp)
        evidence = F.softplus(logits)
        alpha = evidence + 1  # Dirichlet parameters must be > 0
        return alpha


def compute_uncertainty(alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute uncertainty metrics from Dirichlet parameters.

    Args:
        alpha: Dirichlet concentration parameters (batch_size, num_classes)

    Returns:
        Dictionary containing:
            - 'probs': Expected class probabilities
            - 'vacuity': Epistemic uncertainty (lack of evidence)
            - 'dissonance': Aleatoric uncertainty (conflicting evidence)
            - 'entropy': Predictive entropy
            - 'strength': Dirichlet strength (total evidence)
    """
    S = alpha.sum(dim=-1, keepdim=True)  # Dirichlet strength
    K = alpha.shape[-1]

    # Expected probabilities
    probs = alpha / S

    # Epistemic uncertainty: vacuity (proportion of missing evidence)
    vacuity = K / S.squeeze(-1)

    # Aleatoric uncertainty: entropy of expected probabilities
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

    # Dissonance: measures conflicting evidence
    # Higher when evidence is spread across multiple classes
    belief = (alpha - 1) / S
    dissonance = entropy  # Simplified; can use more complex measures

    return {
        'probs': probs,
        'vacuity': vacuity,
        'dissonance': dissonance,
        'entropy': entropy,
        'strength': S.squeeze(-1),
    }


class EvidentialLoss(nn.Module):
    """Evidential loss function for training Dirichlet-based classifiers.

    Combines MSE loss for accurate predictions with KL divergence regularization
    to prevent spurious evidence inflation on incorrect classes.

    L = L_MSE + λ * L_KL

    where λ is annealed from 0 to 1 during training.
    """

    def __init__(
        self,
        num_classes: int,
        annealing_epochs: int = 10,
        lambda_weight: float = 1.0,
    ):
        """Initialize evidential loss.

        Args:
            num_classes: Number of output classes
            annealing_epochs: Epochs to anneal lambda from 0 to lambda_weight
            lambda_weight: Maximum weight for KL regularization
        """
        super().__init__()
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs
        self.lambda_weight = lambda_weight

    def forward(
        self,
        alpha: torch.Tensor,
        targets: torch.Tensor,
        epoch: int = 0,
    ) -> torch.Tensor:
        """Compute evidential loss.

        Args:
            alpha: Dirichlet parameters (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            epoch: Current training epoch for lambda annealing

        Returns:
            Scalar loss value
        """
        # Convert targets to one-hot
        y = F.one_hot(targets, self.num_classes).float()

        # Dirichlet strength
        S = alpha.sum(dim=-1, keepdim=True)

        # Expected probabilities
        p = alpha / S

        # MSE loss
        mse_loss = ((y - p) ** 2).sum(dim=-1).mean()

        # KL divergence regularization
        # Remove evidence for correct class before computing KL
        alpha_tilde = y + (1 - y) * alpha

        # KL(Dir(alpha_tilde) || Dir(1))
        kl_loss = self._kl_divergence(alpha_tilde)

        # Annealing coefficient
        lambda_t = min(1.0, epoch / max(1, self.annealing_epochs)) * self.lambda_weight

        return mse_loss + lambda_t * kl_loss

    def _kl_divergence(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence from Dirichlet(alpha) to Dirichlet(1).

        Args:
            alpha: Dirichlet parameters

        Returns:
            Mean KL divergence across batch
        """
        K = alpha.shape[-1]
        ones = torch.ones_like(alpha)

        sum_alpha = alpha.sum(dim=-1)
        sum_ones = ones.sum(dim=-1)

        kl = (
            torch.lgamma(sum_alpha) - torch.lgamma(sum_ones)
            - (torch.lgamma(alpha) - torch.lgamma(ones)).sum(dim=-1)
            + ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha.unsqueeze(-1)))).sum(dim=-1)
        )

        return kl.mean()


# ============================================================================
# Evidential Model Architectures for Wearable Datasets
# ============================================================================


class EvidentialHARClassifier(nn.Module):
    """Evidential MLP classifier for UCI HAR dataset.

    Architecture: Input(561) -> FC(256) -> FC(128) -> Evidential(6)
    """

    def __init__(
        self,
        input_dim: int = 561,
        hidden_dims: Tuple[int, ...] = (256, 128),
        num_classes: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.evidential_head = EvidentialHead(prev_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Dirichlet alpha parameters.

        Args:
            x: Input features (batch_size, 561)

        Returns:
            Dirichlet alpha parameters (batch_size, num_classes)
        """
        features = self.feature_extractor(x)
        alpha = self.evidential_head(features)
        return alpha

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Predict with uncertainty quantification.

        Args:
            x: Input features

        Returns:
            Tuple of (predicted_classes, uncertainty_dict)
        """
        alpha = self.forward(x)
        uncertainty = compute_uncertainty(alpha)
        predictions = alpha.argmax(dim=-1)
        return predictions, uncertainty


class EvidentialPAMAP2Classifier(nn.Module):
    """Evidential MLP classifier for PAMAP2 dataset with sliding windows.

    Architecture: Input(window*features) -> FC(512) -> FC(256) -> FC(128) -> Evidential(12)
    """

    def __init__(
        self,
        input_dim: int = 4000,  # 100 window * 40 features
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        num_classes: int = 12,
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.evidential_head = EvidentialHead(prev_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        alpha = self.evidential_head(features)
        return alpha

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        alpha = self.forward(x)
        uncertainty = compute_uncertainty(alpha)
        predictions = alpha.argmax(dim=-1)
        return predictions, uncertainty


class EvidentialPPGDaLiAClassifier(nn.Module):
    """Evidential MLP classifier for PPG-DaLiA dataset.

    Architecture: Input(window*6) -> FC(256) -> FC(128) -> FC(64) -> Evidential(7)

    Designed for activity recognition from wrist-worn PPG/accelerometer sensors.
    """

    def __init__(
        self,
        input_dim: int = 192,  # 32 window * 6 features (default)
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        num_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.evidential_head = EvidentialHead(prev_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Dirichlet alpha parameters.

        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            Dirichlet alpha parameters (batch_size, num_classes)
        """
        features = self.feature_extractor(x)
        alpha = self.evidential_head(features)
        return alpha

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Predict with uncertainty quantification.

        Args:
            x: Input features

        Returns:
            Tuple of (predicted_classes, uncertainty_dict)
        """
        alpha = self.forward(x)
        uncertainty = compute_uncertainty(alpha)
        predictions = alpha.argmax(dim=-1)
        return predictions, uncertainty


# ============================================================================
# Model Factory Functions
# ============================================================================


def create_har_model(
    input_dim: int = 561,
    hidden_dims: Tuple[int, ...] = (256, 128),
    num_classes: int = 6,
    dropout: float = 0.3,
) -> nn.Module:
    """Create an evidential HAR classifier model.

    Args:
        input_dim: Number of input features (default: 561 for UCI HAR)
        hidden_dims: Tuple of hidden layer sizes
        num_classes: Number of activity classes
        dropout: Dropout probability

    Returns:
        EvidentialHARClassifier model instance
    """
    return EvidentialHARClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout,
    )


def create_pamap2_model(
    input_dim: int = 4000,  # 100 window * 40 features
    hidden_dims: Tuple[int, ...] = (512, 256, 128),
    num_classes: int = 12,
    dropout: float = 0.3,
) -> nn.Module:
    """Create an evidential PAMAP2 classifier model.

    Args:
        input_dim: Number of input features (window_size * num_sensor_features)
        hidden_dims: Tuple of hidden layer sizes
        num_classes: Number of activity classes
        dropout: Dropout probability

    Returns:
        EvidentialPAMAP2Classifier model instance
    """
    return EvidentialPAMAP2Classifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )


def create_ppg_dalia_model(
    input_dim: int = 192,  # 32 window * 6 features (default)
    hidden_dims: Tuple[int, ...] = (256, 128, 64),
    num_classes: int = 7,
    dropout: float = 0.3,
) -> nn.Module:
    """Create an evidential PPG-DaLiA classifier model.

    Args:
        input_dim: Number of input features (window_size * num_sensor_features)
        hidden_dims: Tuple of hidden layer sizes
        num_classes: Number of activity classes
        dropout: Dropout probability

    Returns:
        EvidentialPPGDaLiAClassifier model instance
    """
    return EvidentialPPGDaLiAClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )


def get_wearable_model_factory(
    dataset_type: str,
    **kwargs,
) -> Callable[[], nn.Module]:
    """Get a model factory function for a wearable dataset.

    Args:
        dataset_type: Type of dataset ('uci_har', 'pamap2', 'ppg_dalia')
        **kwargs: Model-specific parameters

    Returns:
        Callable that creates model instances

    Example:
        factory = get_wearable_model_factory('uci_har', dropout=0.5)
        model = factory()  # Creates new evidential model instance
    """
    dataset_type = dataset_type.lower().replace("-", "_")

    if dataset_type == "uci_har":
        return lambda: create_har_model(**kwargs)
    elif dataset_type == "pamap2":
        return lambda: create_pamap2_model(**kwargs)
    elif dataset_type == "ppg_dalia":
        return lambda: create_ppg_dalia_model(**kwargs)
    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Available: 'uci_har', 'pamap2', 'ppg_dalia'"
        )


def get_evidential_loss(
    num_classes: int,
    annealing_epochs: int = 10,
    lambda_weight: float = 1.0,
) -> EvidentialLoss:
    """Get the evidential loss function for training.

    Args:
        num_classes: Number of output classes
        annealing_epochs: Epochs to anneal KL regularization
        lambda_weight: Maximum KL regularization weight

    Returns:
        EvidentialLoss instance
    """
    return EvidentialLoss(
        num_classes=num_classes,
        annealing_epochs=annealing_epochs,
        lambda_weight=lambda_weight,
    )
