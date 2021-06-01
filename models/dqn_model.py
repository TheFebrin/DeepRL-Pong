import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union

from models.model import Model


class DQN():

    def __init__(
        self, in_channels=4, out_dim=3,
        criterion=None, model_target=None,
    ):
        self.model = Model(in_channels=in_channels, out_dim=out_dim)
        self.mode_target = Model(in_channels=in_channels, out_dim=out_dim)
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion

    def gradient_update(
        self,
        device: str,
        optimizer: torch.optim,
        gamma: float,
        batch: List[Tuple[np.ndarray, np.ndarray, int, float]]
    ) -> float:
        self.model.train()
        preds = self.model.forward_np_array(
            device=device,
            x=np.array([x[0] for x in batch])
        )

        labels = preds.clone().detach()
        labels = labels.to(device)

        next_frames_preds = self.model_target.forward_np_array(
            device=device,
            x=np.array([x[1] if x[1] is not None else x[0] for x in batch])
        ).detach()

        for i, b in enumerate(batch):
            _, next_frame, action, reward = b
            if next_frame is None:  # is it terminal state
                labels[i][action] = reward
            else:
                labels[i][action] = reward + gamma * max(next_frames_preds[i])

        loss = self.criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return float(loss)

    def predict(self, x: np.ndarray, device: str) -> torch.tensor:
        return self.model.forward_np_array(x=x, device=device)

    def update_model_target(self) -> None:
        self.mode_target.load_state_dict(self.model.state_dict())

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, model_name: str) -> None:
        self.model.load_state_dict(torch.load(f'models/saved_models/{model_name}'))
        self.target_model.load_state_dict(torch.load(f'models/saved_models/{model_name}'))

    def to(self, device) -> None:
        self.model.to(device)
        self.target_model.to(device)
