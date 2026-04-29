"""
=============================================================================
  utils/helpers.py  –  Shared utility functions
=============================================================================
"""

from __future__ import annotations

import random
import time
import warnings
from typing import Callable, Sequence

import numpy as np

warnings.filterwarnings("ignore")

def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)

def moving_avg(data: Sequence[float], window: int = 20) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")

def exponential_smooth(data: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out

def convergence_episode(
    rewards: Sequence[float],
    ratio: float = 0.90,
    window: int = 50,
) -> int:
    s = moving_avg(rewards, window)
    if len(s) == 0:
        return len(rewards)
    peak   = float(np.max(s))
    target = ratio * peak
    hits   = np.where(s >= target)[0]
    return int(hits[0]) + window if len(hits) else len(rewards)

# Blackjack mapping
def blackjack_state(obs: tuple) -> int:
    player, dealer, usable_ace = obs
    p = min(max(int(player), 1), 32)
    d = min(max(int(dealer), 1), 10)
    u = int(usable_ace)
    return (p - 1) * 20 + (d - 1) * 2 + u

def make_env(env_name: str):
    import gymnasium as gym
    return gym.make(env_name)

def get_state_fn(env_name: str) -> Callable[[np.ndarray | tuple | int], int]:
    if env_name == "Blackjack-v1":
        return blackjack_state
    return lambda obs: int(obs)

def epsilon_greedy(q_row: np.ndarray, eps: float, n_actions: int) -> int:
    if np.random.rand() < eps:
        return int(np.random.randint(n_actions))
    return int(np.argmax(q_row))

class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
