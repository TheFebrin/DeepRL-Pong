import random
import numpy as np
from typing import Tuple, List


class ReplayMemory:
    """
    Replay memory of some fixed capacity
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []

    def store(
        self,
        prev_phi_value: np.ndarray,
        phi_value: np.ndarray,
        action: int,
        reward: float,
    ):
        self.memory.append((prev_phi_value, phi_value, action, reward))
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample_random_minibatch(
        self, k: int
    ) -> List[Tuple[np.ndarray, np.ndarray, int, float]]:
        """
        Sample a random minibatch of size min(k, size of memory) from the memory.
        """
        k = min(k, len(self.memory))
        return random.sample(self.memory, k)