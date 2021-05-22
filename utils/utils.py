

def epsilon_schedule(eps: float, n_frames: int) -> float:
    """
    Epsilon is annealed linearly from 1 to 0.1 over the n_frames
    """
    change = (1 - 0.1) / n_frames
    eps -= change
    eps = max(eps, 0.1)
    return eps