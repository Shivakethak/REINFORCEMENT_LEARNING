"""
=============================================================================
  agents/dqn.py  –  Deep Q-Network and Double DQN
=============================================================================
"""

from __future__ import annotations
import random
from collections import deque
from typing import Callable, Optional
import numpy as np
from utils.helpers import Timer, set_seeds

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf: deque = deque(maxlen=int(capacity))

    def push(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, mask: float) -> None:
        self.buf.append((
            np.asarray(s,  dtype=np.float64),
            int(a),
            float(r),
            np.asarray(ns, dtype=np.float64),
            float(mask),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, m = zip(*batch)
        return (
            np.array(s,  dtype=np.float64),
            np.array(a,  dtype=np.int64),
            np.array(r,  dtype=np.float64),
            np.array(ns, dtype=np.float64),
            np.array(m,  dtype=np.float64),
        )

    def __len__(self) -> int:
        return len(self.buf)

class MLP:
    def __init__(self, in_dim: int, out_dim: int, lr: float = 5e-4) -> None:
        self.W1 = np.random.randn(in_dim, 128) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(128, dtype=np.float64)
        self.W2 = np.random.randn(128, 64)     * np.sqrt(2.0 / 128)
        self.b2 = np.zeros(64,  dtype=np.float64)
        self.W3 = np.random.randn(64, out_dim)  * np.sqrt(2.0 / 64)
        self.b3 = np.zeros(out_dim, dtype=np.float64)

        self.lr = float(lr)
        self._t  = 0
        self._ms = [np.zeros_like(p) for p in self._params()]
        self._vs = [np.zeros_like(p) for p in self._params()]
        self._x0 = self._z1 = self._a1 = self._z2 = self._a2 = None

    def _params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[np.newaxis]
        self._x0 = x
        self._z1 = x        @ self.W1 + self.b1
        self._a1 = np.maximum(0.0, self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = np.maximum(0.0, self._z2)
        return self._a2 @ self.W3 + self.b3

    def backward(self, target: np.ndarray) -> float:
        out   = self._a2 @ self.W3 + self.b3
        batch = float(self._x0.shape[0])
        mse  = float(np.mean((out - target) ** 2))

        d3  = (out - target) * (2.0 / batch)
        gW3 = self._a2.T @ d3
        gb3 = d3.sum(0)

        d2  = (d3 @ self.W3.T) * (self._z2 > 0)
        gW2 = self._a1.T @ d2
        gb2 = d2.sum(0)

        d1  = (d2 @ self.W2.T) * (self._z1 > 0)
        gW1 = self._x0.T @ d1
        gb1 = d1.sum(0)

        self._adam([gW1, gb1, gW2, gb2, gW3, gb3])
        return mse

    def _adam(self, grads: list[np.ndarray], b1: float = 0.90, b2: float = 0.999, eps: float = 1e-8) -> None:
        self._t += 1
        for i, (p, g) in enumerate(zip(self._params(), grads)):
            self._ms[i] = b1 * self._ms[i] + (1.0 - b1) * g
            self._vs[i] = b2 * self._vs[i] + (1.0 - b2) * g * g
            mh = self._ms[i] / (1.0 - b1 ** self._t)
            vh = self._vs[i] / (1.0 - b2 ** self._t)
            p -= self.lr * mh / (np.sqrt(vh) + eps)

    def copy_from(self, other: "MLP") -> None:
        for dst, src in zip(self._params(), other._params()):
            dst[:] = src

class DQNAgent:
    def __init__(
        self,
        env_name:   str,
        algo:       str   = "DQN",
        n_episodes: int   = 800,
        lr:         float = 5e-4,
        gamma:      float = 0.99,
        epsilon:    float = 1.00,
        eps_decay:  float = 0.995,
        eps_min:    float = 0.01,
        batch_size: int   = 64,
        buf_size:   int   = 10_000,
        tgt_freq:   int   = 100,
        seed:       int   = 42,
    ) -> None:
        self.env_name   = env_name
        self.algo       = algo
        self.n_episodes = n_episodes
        self.lr         = lr
        self.gamma      = gamma
        self.eps_init   = epsilon
        self.eps_decay  = eps_decay
        self.eps_min    = eps_min
        self.batch_size = batch_size
        self.buf_size   = buf_size
        self.tgt_freq   = tgt_freq
        self.seed       = seed

        import gymnasium as gym
        env          = gym.make(env_name)
        in_dim       = int(env.observation_space.shape[0])
        self.n_actions = int(env.action_space.n)
        env.close()

        set_seeds(seed)
        self.online = MLP(in_dim, self.n_actions, lr=lr)
        self.target = MLP(in_dim, self.n_actions, lr=lr)
        self.target.copy_from(self.online)

        self.rewards:         list[float] = []
        self.ep_lengths:      list[int]   = []
        self.losses:          list[float] = []
        self.td_errors:       list[float] = []
        self.epsilon_history: list[float] = []
        self.trained    = False
        self.train_time = 0.0

    def train(self, progress_callback: Optional[Callable[[int, int, float, float], None]] = None) -> None:
        import gymnasium as gym
        set_seeds(self.seed)
        env        = gym.make(self.env_name)
        buf        = ReplayBuffer(self.buf_size)
        eps        = float(self.eps_init)
        step_total = 0

        self.rewards, self.ep_lengths, self.losses, self.td_errors = [], [], [], []
        self.epsilon_history = []

        with Timer() as t:
            for ep in range(self.n_episodes):
                obs, _  = env.reset(seed=ep)
                s       = np.asarray(obs, dtype=np.float64)
                total_r = 0.0
                steps   = 0
                ep_losses: list[float] = []
                done    = False

                while not done:
                    if np.random.rand() < eps:
                        action = env.action_space.sample()
                    else:
                        action = int(np.argmax(self.online.forward(s)[0]))

                    obs_next, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    ns   = np.asarray(obs_next, dtype=np.float64)

                    buf.push(s, action, reward, ns, 0.0 if done else 1.0)
                    s        = ns
                    total_r += float(reward)
                    steps   += 1
                    step_total += 1

                    if len(buf) >= self.batch_size:
                        sb, ab, rb, nsb, mb = buf.sample(self.batch_size)

                        if self.algo == "Double DQN":
                            q_online_next = self.online.forward(nsb)
                            best_actions = np.argmax(q_online_next, axis=1)
                            q_target_next = self.target.forward(nsb)
                            td_target = rb + self.gamma * q_target_next[np.arange(self.batch_size), best_actions] * mb
                        else: # DQN
                            q_next    = self.target.forward(nsb)
                            td_target = rb + self.gamma * np.max(q_next, axis=1) * mb

                        q_pred    = self.online.forward(sb)
                        q_tgt     = q_pred.copy()
                        q_tgt[np.arange(self.batch_size), ab] = td_target

                        loss = self.online.backward(q_tgt)
                        ep_losses.append(loss)

                        if step_total % self.tgt_freq == 0:
                            self.target.copy_from(self.online)

                eps = max(self.eps_min, eps * self.eps_decay)
                self.rewards.append(total_r)
                self.ep_lengths.append(steps)
                self.losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)
                self.td_errors.append(0.0)
                self.epsilon_history.append(eps)

                if progress_callback and ((ep + 1) % 50 == 0 or ep == self.n_episodes - 1):
                    avg50 = float(np.mean(self.rewards[-50:]))
                    progress_callback(ep + 1, self.n_episodes, avg50, eps)

        env.close()
        self.trained    = True
        self.train_time = t.elapsed

    def evaluate(self, n_episodes: int = 100) -> dict[str, float]:
        if not self.trained:
            raise RuntimeError("Agent must be trained before evaluation.")

        import gymnasium as gym
        from config import ENV_INFO
        env = gym.make(self.env_name)
        thr = ENV_INFO[self.env_name]["success_threshold"]
        rews, lens = [], []

        for ep in range(n_episodes):
            obs, _  = env.reset(seed=2000 + ep)
            s       = np.asarray(obs, dtype=np.float64)
            total_r, steps, done = 0.0, 0, False
            while not done:
                a = int(np.argmax(self.online.forward(s)[0]))
                obs_next, r, terminated, truncated, _ = env.step(a)
                done     = terminated or truncated
                s        = np.asarray(obs_next, dtype=np.float64)
                total_r += float(r)
                steps   += 1
            rews.append(total_r)
            lens.append(steps)

        env.close()
        rews_arr = np.array(rews)
        return {
            "eval_avg_reward":   float(np.mean(rews_arr)),
            "eval_success_rate": float(100.0 * np.mean(rews_arr >= thr)),
            "eval_avg_ep_len":   float(np.mean(lens)),
        }
