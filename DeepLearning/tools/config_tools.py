import random
import numpy as np
import torch


def seed_randomness(SEED):
    """Seed randomness across sources

    Args:
        SEED (int): Seed to use

    Returns:
        np.random.Generator: Random number generator
    """
    random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(seed=SEED)
    return rng


def get_device():
    """Get torch device for model training

    Returns:
        str: torch device descriptor
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'
