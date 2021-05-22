from typing import Tuple

import numpy as np

from utils.action import Action

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
        action: Action,
        reward: int,
    ):
        self.memory.append((prev_preprocessed_sequence, preprocessed_sequence, action, reward))
        if len(self.memory) > self.capacity:
            self.memory = self.memory[1:]

    def sample_random_minibatch(
        self, k: int
    ) -> Tuple[np.ndarray, np.ndarray, Action, int]:
        """
        Sample a random minibatch of size k from the memory
        """
        pass