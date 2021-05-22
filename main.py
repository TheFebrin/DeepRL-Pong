import sys
import gym
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import numba
# @numba.jit(nopython=True)

from enum import Enum
from tqdm import tqdm
from typing import *

from utils.phi import phi
from utils.action import Action
from utils.memory import ReplayMemory
from utils.utils import epsilon_schedule
from models.model import Model
from models.dqn_model import DQN


# TODO: (Dawid) Take those params from argparse or json/yaml config.
N: int     = 10       # capacity of memory D
M: int     = 5        # number of episodes in the loop
T: int     = 5        # TODO



def train(
    model: Model,
    minibatch_size: int = 32,
    eps: float = 1.0     # probability to select a random action
):
    memory = ReplayMemory(capacity=N)
    env: gym.wrappers.time_limit.TimeLimit = gym.make("Pong-v0")
    observation: np.ndarray = env.reset()  # reset environment back to its first state
    done: bool = False
    score: int = 0
    total_steps: int = 0

    for episode in range(M):
        sequence: List[np.ndarray] = [observation]
        preprocessed_sequence: np.ndarray = phi(sequence)  # 84 x 84 x 4

        for t in range(T):
            eps = epsilon_schedule(eps, n_frames=1000000)
            if np.random.rand() < eps:  # with probability eps select a random action
                action: Action = Action.select_random()
            else:
                preprocessed_sequence_tensor = torchvision.transforms.ToTensor()(
                    preprocessed_sequence)  # Note: It will normalize the pixels [0, 1] and do .permute(2, 0, 1)

                logits: torch.tensor = model(
                    x=preprocessed_sequence_tensor.view(1, *preprocessed_sequence_tensor.shape)
                )
                action = Action.from_model_prediction(x=logits)

            # Execute action in emulator and observe reward and next frame
            observation, reward, done, info = env.step(action.value)

            sequence.append(observation)
            sequence = sequence[-4:]  # we need only the last 4 observations
            new_preprocessed_sequence = phi(sequence)

            memory.store(
                prev_preprocessed_sequence=preprocessed_sequence,
                preprocessed_sequence=new_preprocessed_sequence,
                action=action,
                reward=reward
            )
            preprocessed_sequence = new_preprocessed_sequence

            minibatch: List[Tuple[np.ndarray, np.ndarray, Action, int]] = memory.sample_random_minibatch(k=minibatch_size)
            print(len(memory.memory))

            total_steps += 1
            # End of the for loop


def main() -> int:
    train(
        model=DQN()
    )


if __name__ == '__main__':
    sys.exit(main())