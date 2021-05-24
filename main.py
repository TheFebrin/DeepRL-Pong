import sys
import gym
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import numba
# @numba.jit(nopython=True)

from tqdm import tqdm
from typing import *

from utils.phi import (
    phi,
    preprocess,
)
from utils.memory import ReplayMemory
from utils.utils import (
    epsilon_schedule,
    select_random_action,
    action_from_model_prediction,
    action_from_trinary_to_env,
)
from models.model import Model
from models.dqn_model import DQN


def train(
        n_games,  # type: int
        optimizer,  # type: torch.optim
        memory,  # type: ReplayMemory
        model,  # type: Model
        minibatch_size=32,  # type: int
        eps=1.0,  # type: float
        gamma=0.90,  # type: float
):
    """
    :param eps: probability to select a random action
    """
    env: gym.wrappers.time_limit.TimeLimit = gym.make("Pong-v0")
    done: bool = False
    score: int = 0
    total_steps: int = 0
    loss_history: List[float] = []

    for episode in tqdm(range(n_games)):
        observation: np.ndarray = env.reset(
        )  # reset environment back to its first state
        preprocessed_sequence: List[np.ndarray] = [preprocess(observation)]
        phi_value: np.ndarray = phi(sequence)  # 84 x 84 x 4
        done: bool = False

        # start one game
        while not done:
            eps = epsilon_schedule(eps, n_frames=100000)
            if np.random.rand(
            ) < eps:  # with probability eps select a random action
                action: int = select_random_action()
            else:
                logits: torch.tensor = model.forward_np_array(x=phi_value)
                action: int = action_from_model_prediction(x=logits)

            # Execute action in emulator and observe reward and next frame
            observation, reward, done, info = env.step(
                action_from_trinary_to_env(action))

            if not done:
                preprocessed_sequence.append(preprocess(observation))
                preprocessed_sequence = preprocessed_sequence[
                    -4:]  # we need only the last 4 observations
                new_phi_value = phi(preprocessed_sequence)
            else:
                new_phi_value = None

            memory.store(prev_phi_value=phi_value,
                         phi_value=new_phi_value,
                         action=action,
                         reward=reward)
            phi_value = new_phi_value

            loss = model.gradient_update(
                optimizer=optimizer,
                gamma=gamma,
                batch=memory.sample_random_minibatch(k=minibatch_size))
            loss_history.append(loss)
            total_steps += 1
            # End of the for loop

    print('Training finished.')
    print(f'Total steps: {total_steps}')
    plt.figure(figsize=(15, 5))
    plt.plot(loss_history)
    plt.show()


def main() -> int:
    # TODO: (Dawid) Take those params from argparse or json/yaml config.
    N: int = 10  # capacity of memory D
    M: int = 5  # number of episodes in the loop
    T: int = 5
    learning_rate: float = 0.0001

    memory = ReplayMemory(capacity=N)

    model = DQN(
        in_channels=4,
        out_dim=3,
    )

    optimizer = torch.optim.Adam(lr=learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 amsgrad=False,
                                 params=model.parameters())

    train(
        n_games=M,
        memory=memory,
        optimizer=optimizer,
        model=model,
    )


if __name__ == '__main__':
    sys.exit(main())
