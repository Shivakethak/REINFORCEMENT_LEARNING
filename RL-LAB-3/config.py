"""
=============================================================================
  config.py  –  Central configuration for the RL Evaluation System
=============================================================================
  All environment metadata, algorithm hyperparameter defaults, plotting
  palette, and evaluation thresholds live here so every other module
  imports from one place.
=============================================================================
"""

# ---------------------------------------------------------------------------
#  ENVIRONMENTS
# ---------------------------------------------------------------------------

ENV_INFO: dict = {
    "CliffWalking-v1": {
        "desc": "Cross a grid from start to goal avoiding the cliff.",
        "algorithms": ["Q-Learning", "SARSA", "Expected SARSA"],
        "success_threshold": -20,   # Optimal is -13
        "color": "#b5179e",
        "pill": "pill-magenta",
        "state_type": "discrete",
        "n_states": 48,
        "n_actions": 4,
        "good_avg_reward": -20,
        "expected_convergence": "200–400 episodes",
        "reward_range": (-1000, 0),
    },
    "Blackjack-v1": {
        "desc": "Beat the dealer without exceeding 21.",
        "algorithms": ["Q-Learning", "SARSA", "Expected SARSA"],
        "success_threshold": 0.0,
        "color": "#00f0ff",
        "pill": "pill-cyan",
        "state_type": "continuous-discretised",
        "n_states": 640, # 32 * 10 * 2
        "n_actions": 2,
        "good_avg_reward": -0.05,
        "expected_convergence": "3000–6000 episodes",
        "reward_range": (-1, 1),
    },
    "Acrobot-v1": {
        "desc": "Swing a 2-link pendulum up to a given height.",
        "algorithms": ["DQN", "Double DQN"],
        "success_threshold": -100,
        "color": "#ffcc00",
        "pill": "pill-yellow",
        "state_type": "continuous",
        "n_states": None, 
        "n_actions": 3,
        "good_avg_reward": -80,
        "expected_convergence": "200–500 episodes",
        "reward_range": (-500, -50),
    },
}

# ---------------------------------------------------------------------------
#  ALGORITHM DEFAULTS
# ---------------------------------------------------------------------------

ALGO_DEFAULTS: dict = {
    "Q-Learning": {
        "alpha":     0.10,
        "gamma":     0.99,
        "epsilon":   1.00,
        "eps_decay": 0.995,
        "eps_min":   0.01,
        "n_episodes": 1000,
    },
    "SARSA": {
        "alpha":     0.10,
        "gamma":     0.99,
        "epsilon":   1.00,
        "eps_decay": 0.995,
        "eps_min":   0.01,
        "n_episodes": 1000,
    },
    "Expected SARSA": {
        "alpha":     0.10,
        "gamma":     0.99,
        "epsilon":   1.00,
        "eps_decay": 0.995,
        "eps_min":   0.01,
        "n_episodes": 1000,
    },
    "DQN": {
        "lr":          5e-4,
        "gamma":       0.99,
        "epsilon":     1.00,
        "eps_decay":   0.995,
        "eps_min":     0.01,
        "n_episodes":  800,
        "batch_size":  64,
        "buf_size":    10_000,
        "tgt_freq":    100,
    },
    "Double DQN": {
        "lr":          5e-4,
        "gamma":       0.99,
        "epsilon":     1.00,
        "eps_decay":   0.995,
        "eps_min":     0.01,
        "n_episodes":  800,
        "batch_size":  64,
        "buf_size":    10_000,
        "tgt_freq":    100,
    },
}

ALGO_INFO: dict = {
    "Q-Learning": {
        "type":  "Off-policy TD",
        "desc":  "Learns the optimal Q-function by bootstrapping with the greedy max future reward.",
        "update": "Q(s,a) ← Q(s,a) + α [ r + γ max Q(s',a') − Q(s,a) ]",
    },
    "SARSA": {
        "type":  "On-policy TD",
        "desc":  "Updates using the action actually chosen by the current policy.",
        "update": "Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') − Q(s,a) ]",
    },
    "Expected SARSA": {
        "type":  "On-policy TD (Expected)",
        "desc":  "Uses the expected value of the next state based on the current policy, reducing variance.",
        "update": "Q(s,a) ← Q(s,a) + α [ r + γ Σ π(a|s')Q(s',a) − Q(s,a) ]",
    },
    "DQN": {
        "type":  "Deep Off-policy",
        "desc":  "Neural-network approximation of Q(s,a) with replay buffer and target network.",
        "update": "θ ← θ − α ∇ [ r + γ max Q(s',a'; θ⁻) − Q(s,a; θ) ]²",
    },
    "Double DQN": {
        "type":  "Deep Off-policy (Decoupled)",
        "desc":  "Selects the best action using the online network but evaluates it using the target network, reducing overestimation bias.",
        "update": "θ ← θ − α ∇ [ r + γ Q(s', argmax Q(s',a; θ); θ⁻) − Q(s,a; θ) ]²",
    },
}

# ---------------------------------------------------------------------------
#  METRICS CATALOGUE
# ---------------------------------------------------------------------------

METRICS_CATALOGUE: dict = {
    "avg_reward": {
        "label":       "Average Reward",
        "formula":     "mean(Rₜ)",
        "unit":        "",
        "description": "The mean return per episode across all training episodes.",
        "higher_is_better": True,
    },
    "cumulative": {
        "label":       "Cumulative Reward",
        "formula":     "Σ Rₜ",
        "unit":        "",
        "description": "The total reward accumulated across ALL training episodes.",
        "higher_is_better": True,
    },
    "success_rate": {
        "label":       "Success Rate",
        "formula":     "#{Rₜ ≥ threshold} / N × 100",
        "unit":        "%",
        "description": "Percentage of episodes achieving success.",
        "higher_is_better": True,
    },
    "conv_speed": {
        "label":       "Convergence Speed",
        "formula":     "first ep where smooth(R) ≥ 0.9 × peak",
        "unit":        "ep",
        "description": "The episode index where reward reaches 90% peak.",
        "higher_is_better": False,
    },
    "avg_ep_len": {
        "label":       "Avg Episode Length",
        "formula":     "mean(steps per episode)",
        "unit":        "steps",
        "description": "The mean number of timesteps per episode.",
        "higher_is_better": None,
    },
    "variance": {
        "label":       "Reward Variance",
        "formula":     "Var(Rₜ)",
        "unit":        "",
        "description": "Statistical variance of per-episode rewards.",
        "higher_is_better": False,
    },
    "std_dev": {
        "label":       "Reward Std Dev",
        "formula":     "Std(Rₜ)",
        "unit":        "",
        "description": "Standard deviation of rewards. More interpretable than variance.",
        "higher_is_better": False,
    },
    "training_time": {
        "label":       "Training Time",
        "formula":     "wall-clock seconds",
        "unit":        "s",
        "description": "Wall-clock seconds required to complete training.",
        "higher_is_better": False,
    },
    "max_reward": {
        "label":       "Max Episode Reward",
        "formula":     "max(Rₜ)",
        "unit":        "",
        "description": "The single best episode reward achieved.",
        "higher_is_better": True,
    },
    "min_reward": {
        "label":       "Min Episode Reward",
        "formula":     "min(Rₜ)",
        "unit":        "",
        "description": "The worst episode reward during training.",
        "higher_is_better": True,
    },
    "median_reward": {
        "label":       "Median Reward",
        "formula":     "median(Rₜ)",
        "unit":        "",
        "description": "The median episode reward.",
        "higher_is_better": True,
    },
    "reward_iqr": {
        "label":       "Reward IQR",
        "formula":     "Q75(Rₜ) − Q25(Rₜ)",
        "unit":        "",
        "description": "Interquartile range of rewards (spread measure).",
        "higher_is_better": False,
    },
    "first_avg_reward": {
        "label":       "First 100-ep Avg",
        "formula":     "mean(first 100 Rₜ)",
        "unit":        "",
        "description": "Mean reward over the first 100 episodes.",
        "higher_is_better": True,
    },
    "final_avg_reward": {
        "label":       "Final 100-ep Avg",
        "formula":     "mean(last 100 Rₜ)",
        "unit":        "",
        "description": "Mean reward over the final 100 episodes.",
        "higher_is_better": True,
    },
    "improvement_rate": {
        "label":       "Improvement Rate",
        "formula":     "(R_last100 − R_first100) / |R_first100|",
        "unit":        "%",
        "description": "Percentage improvement over training.",
        "higher_is_better": True,
    },
}

# ---------------------------------------------------------------------------
#  PLOTTING THEME
# ---------------------------------------------------------------------------

BG    = "#0d1117"
PANEL = "#161b22"
GRID  = "#21262d"
TC    = "#8b949e"
PALETTE = [
    "#ff0055", "#00f0ff", "#ffcc00", "#b5179e",
    "#4cc9f0", "#f72585", "#7209b7", "#3a0ca3",
]

FONT_MONO = "monospace"
