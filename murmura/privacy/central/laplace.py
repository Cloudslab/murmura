import numpy as np
from ..mechanism import CentralPrivacyMechanism

class LaplaceMechanismCDP(CentralPrivacyMechanism):
    def add_noise(self, data: np.ndarray, epsilon: float, delta: float = None) -> np.ndarray:
        scale = 1.0 / epsilon
        noise = np.random.laplace(0, scale, size=data.shape)
        return data + noise
