import torch
import numpy as np
from enum import Enum

"""
From all actions we use only 3 in Pong.

All actions defined here:
https://github.com/openai/gym/blob/ee5ee3a4a5b9d09219ff4c932a45c4a661778cd7/gym/envs/atari/atari_env.py#L219
"""

def action_from_trinary_to_env(action) -> int:
    """
    Maps trinary model output to int action understandable by env
    """
    assert action in (0, 1, 2), f'Wrong action: {action}'
    return {
        0: 0,
        1: 2,
        2: 5
    }[action]

def select_random_action() -> int:
    return np.random.randint(0, 3)

def action_from_model_prediction(x: torch.tensor) -> int:
    return np.argmax(
        x.detach().cpu().numpy()
    )

def epsilon_schedule(eps: float, n_frames: int) -> float:
    """
    Epsilon is annealed linearly from 1 to 0.1 over the n_frames
    """
    change = (1 - 0.05) / n_frames
    eps -= change
    eps = max(eps, 0.05)
    return eps