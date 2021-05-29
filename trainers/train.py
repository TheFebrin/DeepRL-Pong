from comet_ml import Experiment
import os
import sys
import gym
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import yaml
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


def train(
        n_games,                 # type: int
        optimizer,               # type: torch.optim
        memory,                  # type: ReplayMemory
        model,                   # type: Model
        comet_ml_tag,            # type: str
        minibatch_size = 32,     # type: int
        eps            = 1.0,    # type: float
        eps_n_frames   = 10000,  # type: int
        gamma          = 0.90,   # type: float
        frame_skipping = 4,      # type: int
):
    """
    :param eps: probability to select a random action
    """
    experiment = Experiment(
        api_key=os.environ['COMET_ML_API_KEY'],
        project_name="deep-rl-pong",
        workspace="thefebrin",
    )
    experiment.add_tag(comet_ml_tag)
    experiment.log_parameters({
        n_games: n_games,
        minibatch_size: minibatch_size,
        eps: eps,
        eps_n_frames: eps_n_frames,
        gamma: gamma,
        frame_skipping: frame_skipping,
    })
    experiment.set_model_graph(str(model))

    env: gym.wrappers.time_limit.TimeLimit = gym.make("Pong-v0")
    total_steps: int = 0

    for episode in tqdm(range(n_games)):
        experiment.log_current_epoch(episode)
        observation: np.ndarray = env.reset()  # reset environment back to its first state
        preprocessed_sequence: List[np.ndarray] = [preprocess(observation)]
        phi_value: np.ndarray = phi(preprocessed_sequence)  # 84 x 84 x 4
        done: bool = False
        episode_steps: int = 0
        episode_rewad: int = 0
        episode_action_values: np.ndarray = np.zeros((1, 3))
        maximum_actions_values_sum: float = 0

        # start one game
        while not done:
            eps = epsilon_schedule(eps, n_frames=eps_n_frames)
            if np.random.rand() < eps:  # with probability eps select a random action
                action: int = select_random_action()
            else:
                logits: torch.tensor = model.forward_np_array(x=phi_value)
                action: int = action_from_model_prediction(x=logits)
                maximum_actions_values_sum += logits.detach().numpy().max()
                episode_action_values += logits.detach().numpy()

            # Execute action in emulator and observe reward and next frame
            reward: float = 0.0
            for _ in range(frame_skipping):
                observation, partial_reward, done, info = env.step(action_from_trinary_to_env(action))
                reward += partial_reward
                if done:
                    break

            if not done:
                preprocessed_sequence.append(preprocess(observation))
                preprocessed_sequence = preprocessed_sequence[-4:]  # we need only the last 4 observations
                new_phi_value = phi(preprocessed_sequence)
            else:
                new_phi_value = None

            memory.store(
                prev_phi_value=phi_value,
                phi_value=new_phi_value,
                action=action,
                reward=reward,
            )
            phi_value = new_phi_value

            loss = model.gradient_update(
                optimizer=optimizer,
                gamma=gamma,
                batch=memory.sample_random_minibatch(
                    k=minibatch_size
                )
            )
            experiment.log_metric("every_step_loss", loss, step=total_steps)
            episode_rewad += reward
            episode_steps += 1
            total_steps += 1
        # End of the game
        experiment.log_metric(
            "steps_per_episode", episode_steps, step=episode
        )
        experiment.log_metric(
            "average_episode_reward", episode_rewad / episode_steps, step=episode
        )
        experiment.log_metric(
            "average_episode_maximum_actions_values", maximum_actions_values_sum / episode_steps, step=episode
        )
        plt.bar(['NOOP', 'UP', 'DOWN'], (episode_action_values / episode_steps).ravel())
        experiment.log_figure(
           figure_name="average_episode_action_values", figure=plt, step=episode
        )

    print('Training finished.')
    print(f'Total steps: {total_steps}')

