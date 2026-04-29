"""
=============================================================================
  agents/tabular.py  –  Tabular RL Agents
=============================================================================
"""

from __future__ import annotations
from typing import Callable, Optional
import numpy as np

from config import ENV_INFO
from utils.helpers import Timer, epsilon_greedy, get_state_fn, make_env, set_seeds

class TabularAgent:
    def __init__(
        self,
        env_name:   str,
        algo:       str,
        n_episodes: int   = 8000,
        alpha:      float = 0.10,
        gamma:      float = 0.99,
        epsilon:    float = 1.00,
        eps_decay:  float = 0.999,
        eps_min:    float = 0.05,
        seed:       int   = 42,
    ) -> None:
        if algo not in ("Q-Learning", "SARSA", "Expected SARSA"):
            raise ValueError(f"Unknown algo '{algo}'")

        self.env_name   = env_name
        self.algo       = algo
        self.n_episodes = n_episodes
        self.alpha      = alpha
        self.gamma      = gamma
        self.eps_init   = epsilon
        self.eps_decay  = eps_decay
        self.eps_min    = eps_min
        self.seed       = seed

        self.n_states = ENV_INFO[env_name]["n_states"]

        env_tmp         = make_env(env_name)
        self.n_actions  = int(env_tmp.action_space.n)
        env_tmp.close()

        self.get_state: Callable = get_state_fn(env_name)

        set_seeds(seed)
        self.Q = np.random.uniform(-0.01, 0.01, size=(self.n_states, self.n_actions))

        self.rewards:    list[float] = []
        self.ep_lengths: list[int]   = []
        self.td_errors:  list[float] = []
        self.epsilon_history: list[float] = []
        self.trained    = False
        self.train_time: float = 0.0

    def train(
        self,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> None:
        set_seeds(self.seed)
        env = make_env(self.env_name)
        eps = float(self.eps_init)

        self.rewards, self.ep_lengths, self.td_errors, self.epsilon_history = [], [], [], []

        with Timer() as t:
            for ep in range(self.n_episodes):
                raw, _  = env.reset(seed=ep)
                s       = self.get_state(raw)
                total_r = 0.0
                steps   = 0
                ep_td   = 0.0
                done    = False

                if self.algo == "SARSA":
                    a = epsilon_greedy(self.Q[s], eps, self.n_actions)

                while not done:
                    if self.algo in ("Q-Learning", "Expected SARSA"):
                        a = epsilon_greedy(self.Q[s], eps, self.n_actions)

                    raw_next, reward, terminated, truncated, _ = env.step(a)
                    done = terminated or truncated
                    ns   = self.get_state(raw_next)
                    rf   = float(reward)
                    cont = 0.0 if done else 1.0

                    if self.algo == "Q-Learning":
                        td_target  = rf + self.gamma * float(np.max(self.Q[ns])) * cont
                        td_error   = td_target - self.Q[s, a]
                        self.Q[s, a] += self.alpha * td_error

                    elif self.algo == "SARSA":
                        a_next    = epsilon_greedy(self.Q[ns], eps, self.n_actions)
                        td_target = rf + self.gamma * float(self.Q[ns, a_next]) * cont
                        td_error  = td_target - self.Q[s, a]
                        self.Q[s, a] += self.alpha * td_error
                        a = a_next

                    elif self.algo == "Expected SARSA":
                        q_ns = self.Q[ns]
                        best_a = np.argmax(q_ns)
                        expected_q = 0.0
                        for act in range(self.n_actions):
                            prob = eps / self.n_actions
                            if act == best_a:
                                prob += (1.0 - eps)
                            expected_q += prob * q_ns[act]
                            
                        td_target = rf + self.gamma * expected_q * cont
                        td_error = td_target - self.Q[s, a]
                        self.Q[s, a] += self.alpha * td_error

                    s        = ns
                    total_r += rf
                    ep_td   += abs(td_error)
                    steps   += 1

                eps = max(self.eps_min, eps * self.eps_decay)
                self.rewards.append(total_r)
                self.ep_lengths.append(steps)
                self.td_errors.append(ep_td / max(steps, 1))
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

        env  = make_env(self.env_name)
        thr  = ENV_INFO[self.env_name]["success_threshold"]
        rews, lens = [], []

        for ep in range(n_episodes):
            raw, _ = env.reset(seed=1000 + ep)
            s      = self.get_state(raw)
            total_r, steps, done = 0.0, 0, False
            while not done:
                a = int(np.argmax(self.Q[s]))
                raw_next, r, terminated, truncated, _ = env.step(a)
                done     = terminated or truncated
                s        = self.get_state(raw_next)
                total_r += float(r)
                steps   += 1
            rews.append(total_r)
            lens.append(steps)

        env.close()
        rews_arr = np.array(rews)
        return {
            "eval_avg_reward":  float(np.mean(rews_arr)),
            "eval_success_rate": float(100.0 * np.mean(rews_arr >= thr)),
            "eval_avg_ep_len":  float(np.mean(lens)),
        }

    def get_value_function(self) -> np.ndarray:
        return np.max(self.Q, axis=1)

    def get_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)
