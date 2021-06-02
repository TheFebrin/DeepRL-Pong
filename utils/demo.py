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

def demo(
        device,                          # type: str
        model,                           # type: DQN
        frame_skipping=4,                # type: int
):

    if not torch.cuda.is_available():
        print('Cuda not available. Switching device to cpu.')
        device = 'cpu'

    env: gym.wrappers.time_limit.TimeLimit = gym.make("Pong-v0")

    model.to(device)

    observation: np.ndarray = env.reset()  # reset environment back to its first state
    preprocessed_sequence: List[np.ndarray] = [preprocess(observation)]
    phi_value: np.ndarray = phi(preprocessed_sequence)  # 84 x 84 x 4
    done: bool = False

    # start one game
    while not done:
        logits: torch.tensor = model.predict(x=phi_value, device=device)
        action: int = action_from_model_prediction(x=logits)

        # Execute action in emulator and observe reward and next frame
        reward: float = 0.0
        for _ in range(frame_skipping):
            env.render()
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

    env.close()