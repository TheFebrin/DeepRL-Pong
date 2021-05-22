import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import Model

class DQN(Model):

    def __init__(self, in_channels=4, out_dim=3):
        super(DQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(8, 8), stride=4),
            # nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2)),  # TODO: Experiment with MaxPool
        )

        self.dense_layers = nn.Sequential(
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.dense_layers(x)
        """
        We output the values for each action
        """
        return x
