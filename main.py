from comet_ml import Experiment
import sys
import os
import gym
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import numba
# @numba.jit(nopython=True)

from tqdm import tqdm
from typing import *

from utils.memory import ReplayMemory
from utils.test import test
from utils.demo import demo
from models.dqn_model import DQN
from trainers.train import train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', 
        help='Select mode', 
        choices=['train', 'test', 'demo'], 
        default='train',
        )
    args = parser.parse_args()

    config = yaml.safe_load(open("config.yml"))

    if config['LOAD_MODEL']:
        model = DQN(
            in_channels=config['IN_CHANNELS'],
            out_dim=config['OUT_DIM'],
        )
        model_name = config['LOAD_MODEL']
        model.load_model(model_name)
    else:
        model = DQN(
            in_channels=config['IN_CHANNELS'],
            out_dim=config['OUT_DIM'],
        )

    if args.mode=='test':
        test(
            device=config['DEVICE'],
            n_games=config['TEST_GAMES'],
            model=model,
            frame_skipping=config['FRAME_SKIPPING'],
        )
    elif args.mode=='demo':
        demo(
            device=config['DEVICE'],
            model=model,
            frame_skipping=config['FRAME_SKIPPING'],
        )
    else:
        memory = ReplayMemory(capacity=config['N'])

        optimizer_name = config['OPTIMIZER']
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                lr=config['LEARNING_RATE'],
                betas=(0.9, 0.999), eps=1e-8, amsgrad=False,
                params=model.model.parameters()
            )
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                lr=config['LEARNING_RATE'],
                momentum=0.9,
                params=model.model.parameters()
            )
        else:
            raise ValueError(f'Unknown optimizer name: {optimizer_name}')

        experiment = Experiment(
            api_key=os.environ['COMET_ML_API_KEY'],
            project_name=config['COMET_ML_PROJECT_NAME'],
            workspace=config['COMET_ML_WORKSPACE'],
        )

        experiment.set_name(config['COMET_ML_NAME'])
        experiment.add_tag(config['COMET_ML_TAG'])
        experiment.log_parameters({
            'n_games':          config['M'],
            'minibatch_size':   config['MINIBATCH_SIZE'],
            'eps':              config['EPS'],
            'eps_n_frames':     config['EPS_N_FRAMES'],
            'gamma':            config['GAMMA'],
            'frame_skipping':   config['FRAME_SKIPPING'],
            'save_model_every': config['SAVE_MODEL_EVERY']
        })
        experiment.set_model_graph(str(model.model))

        train(
            device=config['DEVICE'],
            n_games=config['M'],
            memory=memory,
            optimizer=optimizer,
            model=model,
            experiment=experiment,
            minibatch_size=config['MINIBATCH_SIZE'],
            eps=config['EPS'],
            eps_n_frames=config['EPS_N_FRAMES'],
            gamma=config['GAMMA'],
            frame_skipping=config['FRAME_SKIPPING'],
            update_model_target_every=config['UPDATE_MODEL_TARGET_EVERY'],
            save_model_every=config['SAVE_MODEL_EVERY'],
            save_average_metrics_every=config['SAVE_AVERAGE_METRICS_EVERY'],
        )


if __name__ == '__main__':
    sys.exit(main())
