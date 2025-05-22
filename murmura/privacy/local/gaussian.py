import numpy as np
from ..mechanism import LocalPrivacyMechanism

class GaussianMechanismLDP(LocalPrivacyMechanism):
    def add_noise(self, data: np.ndarray, epsilon: float, delta: float = None) -> np.ndarray:
        if delta is None:
            raise ValueError("Delta must be specified for Gaussian mechanism.")
        sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = np.random.normal(0, sigma, size=data.shape)
        return data + noise
