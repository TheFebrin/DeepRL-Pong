import torch
import numpy as np
from enum import Enum

class Action(Enum):
    """
    From all actions we use only 3 in Pong
    All actions:
        0: NOOP
        1: FIRE
        2: RIGHT
        3: LEFT
        4: RIGHTFIRE
        5: LEFTFIRE
    """
    NOOP = 0
    RIGHT = 2
    LEFT = 3

    @staticmethod
    def from_id(id_: int):
        if id_ == 0:
            return Action.NOOP
        elif id_ == 1:
            return Action.RIGHT
        elif id_ == 2:
            return Action.LEFT
        else:
            raise ValueError(f'Wrong r: {id_}')

    @staticmethod
    def select_random():
        r = np.random.randint(0, 3)
        return Action.from_id(id_=r)

    @staticmethod
    def from_model_prediction(x: torch.tensor) -> np.ndarray:
        best_action = np.argmax(
            x.detach().cpu().numpy()
        )
        return Action.from_id(id_=best_action)