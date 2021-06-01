import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod

class IModel(nn.Module):

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()