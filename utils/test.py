import torch
import gym
import numpy as np

from tqdm import tqdm
from typing import *

from utils.phi import (
    phi,
    preprocess,
)
from utils.utils import (
    action_from_model_prediction,
    action_from_trinary_to_env,
)
from models.dqn_model import DQN

def test(
        device,                          # type: str
        n_games,                         # type: int
        model,                           # type: DQN
        frame_skipping=4,                # type: int
):

    if not torch.cuda.is_available():
        print('Cuda not available. Switching device to cpu.')
        device = 'cpu'

    env: gym.wrappers.time_limit.TimeLimit = gym.make("Pong-v0")

    total_steps: int = 0
    total_rewad: int = 0
    maximum_actions_values_sum: float = 0
    best_reward: int = -22
    worst_reward: int = 22

    model.to(device)
    for episode in tqdm(range(n_games)):
        observation: np.ndarray = env.reset()  # reset environment back to its first state
        preprocessed_sequence: List[np.ndarray] = [preprocess(observation)]
        phi_value: np.ndarray = phi(preprocessed_sequence)  # 84 x 84 x 4
        done: bool = False
        episode_action_values: np.ndarray = np.zeros((1, 3))

        # start one game
        while not done:
            logits: torch.tensor = model.predict(x=phi_value, device=device)
            action: int = action_from_model_prediction(x=logits)
            maximum_actions_values_sum += logits.detach().cpu().numpy().max()
            episode_action_values += logits.detach().cpu().numpy()

            # Execute action in emulator and observe reward and next frame
            reward: float = 0.0
            for _ in range(frame_skipping):
                observation, partial_reward, done, _ = env.step(action_from_trinary_to_env(action))
                reward += partial_reward
                if done:
                    break

            if not done:
                preprocessed_sequence.append(preprocess(observation))
                preprocessed_sequence = preprocessed_sequence[-4:]  # we need only the last 4 observations
                new_phi_value = phi(preprocessed_sequence)
            else:
                new_phi_value = None

            phi_value = new_phi_value

            if reward > best_reward:
                best_reward = reward
            if reward < worst_reward:
                worst_reward = reward
            total_rewad += reward
            total_steps += 1
        
        
    print('Testing finished.')
    print(f'Total steps: {total_steps}')
    print(f'Average number of steps per episode: {total_steps/n_games}')
    print(f'Average maximum actions values sum: {maximum_actions_values_sum/total_steps}')
    print(f'Average reward per episode: {total_rewad/n_games}')
    print(f'Highest reward: {best_reward}')
    print(f'Lowest reward {worst_reward}')