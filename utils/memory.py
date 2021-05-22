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
        prev_preprocessed_sequence: np.ndarray,
        preprocessed_sequence: np.ndarray,
        action: int,
        reward: int,
    ):
        self.memory.append((prev_preprocessed_sequence, preprocessed_sequence, action, reward))
        if len(self.memory) > self.capacity:
            self.memory = self.memory[1:]

    def sample_random_minibatch(
        self, k: int
    ) -> List[Tuple[np.ndarray, np.ndarray, int, int]]:
        """
        Sample a random minibatch of size min(k, size of memory) from the memory.
        """
        k = min(k, len(self.memory))
        return random.sample(self.memory, k)