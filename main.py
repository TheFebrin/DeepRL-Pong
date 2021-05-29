from comet_ml import Experiment
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
from models.dqn_model import DQN
from trainers.train import train


def main() -> int:
    config = yaml.safe_load(open("config.yml"))

    memory = ReplayMemory(capacity=config['N'])

    model = DQN(
        in_channels=config['IN_CHANNELS'],
        out_dim=config['OUT_DIM'],
    )

    optimizer = torch.optim.Adam(
        lr=config['LEARNING_RATE'],
        betas=(0.9, 0.999), eps=1e-8, amsgrad=False,
        params=model.parameters()
    )

    train(
        n_games=config['M'],
        memory=memory,
        optimizer=optimizer,
        model=model,
        minibatch_size=config['MINIBATCH_SIZE'],
        eps=config['EPS'],
        eps_n_frames=config['EPS_N_FRAMES'],
        gamma=config['GAMMA'],
        frame_skipping=config['FRAME_SKIPPING'],
        comet_ml_tag=config['COMET_ML_TAG'],
    )


if __name__ == '__main__':
    sys.exit(main())
