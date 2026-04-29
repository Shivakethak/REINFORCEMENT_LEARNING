# RL Agent Evaluation Dashboard - LAB 3

A structured, production-quality Streamlit application for training and evaluating
Reinforcement Learning agents with 13 evaluation metrics, Q-table heatmaps,
policy visualisation, contour plots, and multi-run radar charts.

This project features:
- **Environments**: CliffWalking-v1, Blackjack-v1, Acrobot-v1, LunarLander-v3.
- **Algorithms**: Q-Learning, SARSA, Expected SARSA, DQN, Double DQN.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Project Structure

```
rl_lab_3/
├── app.py                   ← Streamlit entry point
├── config.py                ← All constants (envs, algos, metrics, palette)
├── requirements.txt
│
├── agents/
│   ├── tabular.py           ← Q-Learning, SARSA, Expected SARSA (TabularAgent class)
│   └── dqn.py               ← DQN, Double DQN with NumPy MLP (DQNAgent class)
│
├── metrics/
│   └── evaluator.py         ← 13-metric computation + Q-table stats
│
├── plots/
│   └── visualiser.py        ← All matplotlib figures
│
└── utils/
    └── helpers.py           ← Seeds, smoothing, discretisation, timers
```
