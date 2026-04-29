"""
=============================================================================
  plots/visualiser.py  –  All matplotlib figures for the RL dashboard
=============================================================================
"""

from __future__ import annotations
from typing import Optional, Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from config import BG, FONT_MONO, GRID, PALETTE, PANEL, TC
from utils.helpers import moving_avg

def _ax_style(ax, title: str = "", xlabel: str = "", ylabel: str = "", fontsize: int = 8) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TC, labelsize=fontsize - 1)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)
    if title:
        ax.set_title(title, color="#c9d1d9", fontsize=fontsize, fontweight="bold", pad=5, fontfamily=FONT_MONO)
    if xlabel:
        ax.set_xlabel(xlabel, color=TC, fontsize=fontsize - 1)
    if ylabel:
        ax.set_ylabel(ylabel, color=TC, fontsize=fontsize - 1)

def _fig(w: float, h: float) -> plt.Figure:
    return plt.figure(figsize=(w, h), facecolor=BG)

def make_training_overview(metrics: dict, env_name: str, algo: str, color: str, td_errors: Optional[list] = None) -> plt.Figure:
    rewards  = np.asarray(metrics["rewards"],    dtype=float)
    ep_lens  = np.asarray(metrics["ep_lengths"], dtype=float)
    n        = len(rewards)

    fig = _fig(15, 8)
    gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.48)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rewards, color=color, alpha=0.18, linewidth=0.6, zorder=1)
    if n > 30:
        sm = moving_avg(rewards, 30)
        ax1.plot(range(29, n), sm, color=color, linewidth=2.0, label="30-ep avg", zorder=2)
    conv = metrics.get("conv_speed", n)
    if conv < n:
        ax1.axvline(conv, color="#ffa657", linewidth=1.2, linestyle="--", label=f"conv ≈ ep{conv}", zorder=3)
    thr_line = metrics.get("q75")
    if thr_line:
        ax1.axhline(thr_line, color="#8b949e", linewidth=0.8, linestyle=":", label=f"Q75={thr_line:.1f}", zorder=2)
    _ax_style(ax1, "Episode Reward", "Episode", "Reward")
    ax1.legend(fontsize=6, facecolor=PANEL, edgecolor=GRID, labelcolor=TC)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ep_lens, color="#ffa657", alpha=0.22, linewidth=0.6)
    if n > 30:
        ax2.plot(range(29, n), moving_avg(ep_lens, 30), color="#ffa657", linewidth=2.0)
    ax2.axhline(metrics["avg_ep_len"], color="#8b949e", linewidth=0.9, linestyle="--", label=f"mean={metrics['avg_ep_len']:.1f}")
    _ax_style(ax2, "Episode Length", "Episode", "Steps")
    ax2.legend(fontsize=6, facecolor=PANEL, edgecolor=GRID, labelcolor=TC)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(rewards, bins=30, color=color, alpha=0.7, edgecolor=BG, linewidth=0.4)
    ax3.axvline(metrics["avg_reward"],    color="#ffa657", linewidth=1.4, linestyle="--", label=f"mean={metrics['avg_reward']:.1f}")
    ax3.axvline(metrics["median_reward"], color="#79c0ff", linewidth=1.2, linestyle="-.", label=f"median={metrics['median_reward']:.1f}")
    _ax_style(ax3, "Reward Distribution", "Reward", "Count")
    ax3.legend(fontsize=6, facecolor=PANEL, edgecolor=GRID, labelcolor=TC)

    ax4 = fig.add_subplot(gs[1, 0])
    cum = np.cumsum(rewards)
    ax4.fill_between(range(n), cum, color=color, alpha=0.15)
    ax4.plot(cum, color=color, linewidth=1.8)
    _ax_style(ax4, "Cumulative Reward", "Episode", "Total Reward")

    ax5   = fig.add_subplot(gs[1, 1])
    roll  = metrics.get("rolling", {})
    w_off = 49
    if roll and len(roll.get("mean", [])) > 0:
        xs    = np.arange(w_off, w_off + len(roll["mean"]))
        rmean = np.array(roll["mean"])
        rstd  = np.array(roll["std"])
        ax5.fill_between(xs, rmean - rstd, rmean + rstd, color=color, alpha=0.20, label="± 1 SD")
        ax5.plot(xs, rmean, color=color, linewidth=1.8, label="50-ep mean")
        ax5.plot(xs, np.array(roll["lo"]), color=GRID, linewidth=0.8, linestyle="--", label="min/max")
        ax5.plot(xs, np.array(roll["hi"]), color=GRID, linewidth=0.8, linestyle="--")
    _ax_style(ax5, "Rolling Reward (50-ep window)", "Episode", "Reward")
    ax5.legend(fontsize=6, facecolor=PANEL, edgecolor=GRID, labelcolor=TC)

    ax6 = fig.add_subplot(gs[1, 2])
    if td_errors and len(td_errors) > 0:
        td_arr = np.asarray(td_errors, dtype=float)
        ax6.plot(td_arr, color="#d2a8ff", alpha=0.3, linewidth=0.6)
        if len(td_arr) > 20:
            ax6.plot(range(19, len(td_arr)), moving_avg(td_arr, 20), color="#d2a8ff", linewidth=1.8)
        _ax_style(ax6, "TD Error / Loss Decay", "Episode", "|TD error|")
    else:
        ax6.text(0.5, 0.5, "TD error\nnot recorded\nfor this run", ha="center", va="center", color=TC, fontsize=8, transform=ax6.transAxes)
        _ax_style(ax6, "TD Error / Loss Decay")

    fig.suptitle(f"{env_name}  ·  {algo}  ·  Training Overview", color="#e6edf3", fontsize=11, fontweight="bold", y=1.01, fontfamily=FONT_MONO)
    fig.tight_layout()
    return fig

def _cliff_walking_qtable(Q: np.ndarray, algo: str) -> plt.Figure:
    V = np.max(Q, axis=1).reshape(4, 12)
    policy = np.argmax(Q, axis=1).reshape(4, 12)
    ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    
    fig, ax = plt.subplots(figsize=(14, 4), facecolor=BG)
    im = ax.imshow(V, cmap="Blues", aspect="auto")
    for r in range(4):
        for c in range(12):
            if r == 3 and 0 < c < 11:
                bg = "#3a1a1a"
                cell = "C"
            elif r == 3 and c == 0:
                bg = "#0d2a4a"
                cell = "S"
            elif r == 3 and c == 11:
                bg = "#1a3a2a"
                cell = "G"
            else:
                bg = PANEL
                cell = ""
            rect = plt.Rectangle([c - 0.5, r - 0.5], 1, 1, color=bg, zorder=0)
            ax.add_patch(rect)
            arrow = ARROWS[int(policy[r, c])]
            v_val = float(V[r, c])
            ax.text(c, r, f"{cell}\n{arrow}\n{v_val:.1f}", ha="center", va="center", color="#e6edf3", fontsize=7, fontfamily=FONT_MONO)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.set_title(f"CliffWalking – Value & Policy ({algo})", color="#c9d1d9", fontsize=9, fontweight="bold", fontfamily=FONT_MONO, pad=8)
    fig.tight_layout()
    return fig

def make_qtable_heatmap(Q: np.ndarray, env_name: str, algo: str) -> Optional[plt.Figure]:
    if env_name == "CliffWalking-v1":
        return _cliff_walking_qtable(Q, algo)
    return None

def make_epsilon_decay(epsilon_hist: list[float], env_name: str, algo: str) -> plt.Figure:
    fig, ax = _fig(8, 3), None
    ax = fig.add_subplot(111)
    eps = np.array(epsilon_hist, dtype=float)
    ax.fill_between(range(len(eps)), eps, color="#ffa657", alpha=0.15)
    ax.plot(eps, color="#ffa657", linewidth=2.0, label="ε (exploration)")
    ax.set_ylim(-0.02, 1.05)
    _ax_style(ax, f"Epsilon Decay Schedule  ·  {env_name}  ·  {algo}", "Episode", "ε")
    ax.axhline(eps[-1], color="#8b949e", linewidth=0.8, linestyle="--", label=f"min ε = {eps[-1]:.4f}")
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TC)
    fig.tight_layout()
    return fig

def make_overlay_curves(all_results: list[dict]) -> plt.Figure:
    fig, ax = _fig(14, 4), None
    ax = fig.add_subplot(111)
    for i, r in enumerate(all_results):
        rw  = np.asarray(r["metrics"]["rewards"], dtype=float)
        n   = len(rw)
        col = PALETTE[i % len(PALETTE)]
        lbl = f"{r['env']} / {r['algo']}"
        ax.plot(rw, color=col, alpha=0.10, linewidth=0.7)
        if n > 30:
            sm = moving_avg(rw, 30)
            ax.plot(range(29, n), sm, color=col, linewidth=2.0, label=lbl)
        else:
            ax.plot(rw, color=col, linewidth=2.0, label=lbl)
    _ax_style(ax, "All Runs – Smoothed Reward Curves (30-ep avg)", "Episode", "Reward")
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TC, loc="lower right")
    fig.tight_layout()
    return fig

def make_comparison_bars(all_results: list[dict]) -> Optional[plt.Figure]:
    if not all_results: return None
    labels  = [f"{r['env'][:8]}\n{r['algo']}" for r in all_results]
    colours = [PALETTE[i % len(PALETTE)] for i in range(len(all_results))]
    keys   = ["avg_reward", "success_rate", "conv_speed", "avg_ep_len", "variance", "training_time"]
    titles = ["Avg Reward", "Success Rate (%)", "Convergence (ep)", "Avg Ep Len", "Reward Variance", "Train Time (s)"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), facecolor=BG)
    for ax, key, title in zip(axes.flatten(), keys, titles):
        vals = [r["metrics"][key] for r in all_results]
        bars = ax.bar(range(len(vals)), vals, color=colours, width=0.55, edgecolor=BG, linewidth=0.8)
        for bar, v in zip(bars, vals):
            ypos = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, ypos + 0.01 * abs(ypos) + 0.5, f"{v:.1f}", ha="center", va="bottom", color="#e6edf3", fontsize=6, fontfamily=FONT_MONO)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=6)
        _ax_style(ax, title, "", title)
    fig.suptitle("Multi-run Metric Comparison", color="#e6edf3", fontsize=11, fontweight="bold", y=1.01, fontfamily=FONT_MONO)
    fig.tight_layout()
    return fig

def make_metric_radar(all_results: list[dict]) -> Optional[plt.Figure]:
    if len(all_results) < 1: return None
    metrics_to_show = ["avg_reward", "success_rate", "final_avg_reward", "improvement_rate", "reward_iqr"]
    labels = ["Avg\nReward", "Success\nRate", "Final 100\nAvg", "Improvement\n%", "IQR\n(inverted)"]
    raw = {k: [r["metrics"].get(k, 0) for r in all_results] for k in metrics_to_show}
    def _norm(vals, invert=False):
        arr   = np.array(vals, dtype=float)
        lo, hi = arr.min(), arr.max()
        if hi == lo: return np.ones_like(arr) * 0.5
        n = (arr - lo) / (hi - lo)
        return 1.0 - n if invert else n
    normed = {
        "avg_reward":       _norm(raw["avg_reward"]),
        "success_rate":     _norm(raw["success_rate"]),
        "final_avg_reward": _norm(raw["final_avg_reward"]),
        "improvement_rate": _norm(raw["improvement_rate"]),
        "reward_iqr":       _norm(raw["reward_iqr"], invert=True),
    }
    n_axes = len(metrics_to_show)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 5.5), subplot_kw={"polar": True}, facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.spines["polar"].set_edgecolor(GRID)
    ax.tick_params(colors=TC)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=TC, fontsize=7)
    ax.yaxis.set_tick_params(labelcolor=TC, labelsize=6)
    ax.set_ylim(0, 1)
    for i, r in enumerate(all_results):
        col  = PALETTE[i % len(PALETTE)]
        lbl  = f"{r['env'][:8]}/{r['algo']}"
        vals = [float(normed[k][i]) for k in metrics_to_show]
        vals += vals[:1]
        ax.fill(angles, vals, color=col, alpha=0.12)
        ax.plot(angles, vals, color=col, linewidth=1.8, label=lbl)
    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.4, 1.15), facecolor=PANEL, edgecolor=GRID, labelcolor=TC)
    ax.set_title("Normalised Metric Radar", color="#c9d1d9", fontsize=9, fontweight="bold", pad=15, fontfamily=FONT_MONO)
    fig.tight_layout()
    return fig

def make_episode_length_dist(all_results: list[dict]) -> Optional[plt.Figure]:
    if not all_results: return None
    data   = [r["metrics"]["ep_lengths"]  for r in all_results]
    labels = [f"{r['env'][:8]}\n{r['algo']}" for r in all_results]
    colours = [PALETTE[i % len(PALETTE)] for i in range(len(all_results))]
    fig, ax = _fig(max(8, 3 * len(all_results)), 4.5), None
    ax = fig.add_subplot(111)
    bp = ax.boxplot(
        data, patch_artist=True, notch=False,
        medianprops={"color": "#e6edf3", "linewidth": 1.5},
        whiskerprops={"color": TC, "linewidth": 1.0},
        capprops={"color": TC},
        flierprops={"marker": "o", "markersize": 2, "markerfacecolor": TC, "alpha": 0.5},
    )
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.45)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=7)
    _ax_style(ax, "Episode Length Distribution (Box Plot)", "Run", "Steps / Episode")
    fig.tight_layout()
    return fig

def make_td_error_curve(td_errors: list[float], env_name: str, algo: str) -> Optional[plt.Figure]:
    if not td_errors or all(e == 0 for e in td_errors): return None
    arr = np.asarray(td_errors, dtype=float)
    n   = len(arr)
    fig, ax = _fig(10, 3.5), None
    ax = fig.add_subplot(111)
    ax.plot(arr, color="#d2a8ff", alpha=0.25, linewidth=0.7)
    if n > 20:
        sm = moving_avg(arr, 20)
        ax.plot(range(19, n), sm, color="#d2a8ff", linewidth=2.0, label="20-ep avg")
    _ax_style(ax, f"TD Error Decay  ·  {env_name}  ·  {algo}", "Episode", "Mean |TD error|")
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TC)
    fig.tight_layout()
    return fig

def make_qtable_evolution(snapshots: list[tuple[int, np.ndarray]], env_name: str, algo: str) -> Optional[plt.Figure]:
    if not snapshots: return None
    eps_list  = [ep for ep, _ in snapshots]
    mean_vals = [float(np.mean(np.max(Q, axis=1))) for _, Q in snapshots]
    max_vals  = [float(np.max(Q))  for _, Q in snapshots]
    min_vals  = [float(np.min(Q))  for _, Q in snapshots]
    fig, ax = _fig(10, 3.5), None
    ax = fig.add_subplot(111)
    ax.fill_between(eps_list, min_vals, max_vals, color=PALETTE[0], alpha=0.12, label="Q-value range")
    ax.plot(eps_list, mean_vals, color=PALETTE[0], linewidth=2.0, marker="o", markersize=5, label="Mean V(s)")
    ax.plot(eps_list, max_vals, color=PALETTE[1], linewidth=1.2, linestyle="--", label="max Q")
    ax.plot(eps_list, min_vals, color=PALETTE[3], linewidth=1.2, linestyle="--", label="min Q")
    _ax_style(ax, f"Q-table Value Evolution  ·  {env_name}  ·  {algo}", "Episode checkpoint", "Q-value")
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TC)
    fig.tight_layout()
    return fig
