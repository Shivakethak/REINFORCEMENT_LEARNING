"""
=============================================================================
  metrics/evaluator.py  –  RL Agent Evaluation Metrics
=============================================================================
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
from utils.helpers import convergence_episode, moving_avg

def rolling_stats(
    rewards: Sequence[float],
    window: int = 50,
) -> dict[str, np.ndarray]:
    arr = np.asarray(rewards, dtype=float)
    n   = len(arr)
    out = {"mean": [], "std": [], "lo": [], "hi": []}
    for i in range(window - 1, n):
        window_data = arr[i - window + 1 : i + 1]
        out["mean"].append(float(np.mean(window_data)))
        out["std"].append(float(np.std(window_data)))
        out["lo"].append(float(np.min(window_data)))
        out["hi"].append(float(np.max(window_data)))
    return {k: np.array(v) for k, v in out.items()}

def qtable_stats(Q: np.ndarray) -> dict:
    greedy = np.argmax(Q, axis=1)
    counts = np.bincount(greedy, minlength=Q.shape[1])
    probs  = counts / counts.sum()
    policy_entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))

    return {
        "max_q":          float(Q.max()),
        "min_q":          float(Q.min()),
        "mean_q":         float(Q.mean()),
        "std_q":          float(Q.std()),
        "value_range":    float(Q.max() - Q.min()),
        "greedy_actions": greedy,
        "action_counts":  counts,
        "policy_entropy": policy_entropy,
    }

def compute_metrics(
    rewards:      Sequence[float],
    ep_lengths:   Sequence[float],
    train_time:   float,
    success_thr:  float,
    smooth_window: int = 30,
) -> dict:
    r = np.asarray(rewards,    dtype=float)
    l = np.asarray(ep_lengths, dtype=float)
    n = len(r)

    q25, q75 = float(np.percentile(r, 25)), float(np.percentile(r, 75))

    seg     = min(100, n)
    first100 = float(np.mean(r[:seg]))
    last100  = float(np.mean(r[-seg:]))

    if abs(first100) > 1e-9:
        improvement = 100.0 * (last100 - first100) / abs(first100)
    else:
        improvement = 0.0

    metrics = {
        "avg_reward":       float(np.mean(r)),
        "cumulative":       float(np.sum(r)),
        "success_rate":     float(100.0 * np.sum(r >= success_thr) / max(n, 1)),
        "conv_speed":       convergence_episode(r, ratio=0.90, window=smooth_window),
        "avg_ep_len":       float(np.mean(l)),
        "variance":         float(np.var(r)),
        "std_dev":          float(np.std(r)),
        "training_time":    float(train_time),
        "max_reward":       float(np.max(r)),
        "min_reward":       float(np.min(r)),
        "median_reward":    float(np.median(r)),
        "reward_iqr":       float(q75 - q25),
        "first_avg_reward": first100,
        "final_avg_reward": last100,
        "improvement_rate": improvement,
        "rewards":          r.tolist(),
        "ep_lengths":       l.tolist(),
        "n_episodes":       n,
        "q25":              q25,
        "q75":              q75,
        "first100_avg":     first100,
        "last100_avg":      last100,
        "rolling":          rolling_stats(r, window=min(50, n // 4 or 1)),
    }
    return metrics

def td_error_stats(td_errors: Sequence[float]) -> dict:
    arr = np.asarray(td_errors, dtype=float)
    return {
        "td_mean": float(np.mean(arr)),
        "td_std":  float(np.std(arr)),
        "td_data": arr.tolist(),
    }
