from abc import ABC, abstractmethod
import numpy as np

class PrivacyMechanism(ABC):
    @abstractmethod
    def add_noise(self, data: np.ndarray, epsilon: float, delta: float = None) -> np.ndarray:
        pass

class LocalPrivacyMechanism(PrivacyMechanism):
    pass

class CentralPrivacyMechanism(PrivacyMechanism):
    pass
