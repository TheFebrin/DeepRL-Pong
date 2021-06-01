import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union

from models.model_interface import IModel

class Model(IModel):

    def __init__(
        self, in_channels=4, out_dim=3,
    ):
        super(Model, self).__init__()
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

    def forward(self, x: torch.tensor):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.dense_layers(x)
        """
        We output the values for each action
        """
        return x

    def _preprocess_array(
        self,
        device: str,
        x: Union[np.ndarray, List[np.ndarray]]
    ) -> torch.tensor:
        x = torch.tensor(x, dtype=torch.float32) / 255.0
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1)
            x = x.view(1, *x.shape)
        elif len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f'Wrong number of shapes: {x.shape}')
        return x.to(device)

    def forward_np_array(self, x: np.ndarray, device: str) -> torch.tensor:
        return self.forward(
            x=self._preprocess_array(device=device, x=x),
        )