"""Device management utilities."""

import torch


def get_device() -> torch.device:
    """Get the best available device for computation.

    Returns:
        torch.device: CUDA if available, MPS (Apple Silicon) if available, otherwise CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
