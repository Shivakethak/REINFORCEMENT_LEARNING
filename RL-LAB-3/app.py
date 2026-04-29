"""
=============================================================================
  app.py  –  RL Agent Evaluation Dashboard  (Main Streamlit Entry Point)
=============================================================================
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Train and Evaluate your RL agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from agents.dqn     import DQNAgent
from agents.tabular import TabularAgent
from config         import ALGO_DEFAULTS, ALGO_INFO, ENV_INFO, METRICS_CATALOGUE, PALETTE
from metrics.evaluator import compute_metrics, qtable_stats
from plots.visualiser  import (
    make_comparison_bars, make_episode_length_dist, make_epsilon_decay,
    make_metric_radar, make_overlay_curves, make_qtable_evolution,
    make_qtable_heatmap, make_td_error_curve, make_training_overview,
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }

section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #e6edf3 !important; }

.dash-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ff0055 0%, #b5179e 55%, #00f0ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.1rem;
}
.dash-sub {
    color: #8b949e;
    font-size: 0.82rem;
    margin-bottom: 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
}
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 7px;
}
.metric-label {
    font-size: 0.62rem;
    color: #000000;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 5px;
}
.metric-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #00f0ff;
    font-family: 'JetBrains Mono', monospace;
}
.metric-unit { font-size: 0.66rem; color: #8b949e; margin-left: 2px; }
.metric-desc {
    font-size: 0.60rem;
    color: #484f58;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.35;
}
.pill {
    display: inline-block; padding: 2px 10px;
    border-radius: 20px; font-size: 0.63rem;
    font-family: 'JetBrains Mono', monospace; font-weight: 700;
}
.pill-magenta { background:#2a1a4a; color:#b5179e; border:1px solid #b5179e; }
.pill-cyan   { background:#0d2a4a; color:#00f0ff; border:1px solid #00f0ff; }
.pill-yellow { background:#3a3a1a; color:#ffcc00; border:1px solid #ffcc00; }
.pill-pink    { background:#3a1a2a; color:#ff0055; border:1px solid #ff0055; }
.section-title {
    font-family: 'JetBrains Mono', monospace; font-weight: bold;
    font-size: 0.68rem; color: #000000; text-transform: uppercase;
    letter-spacing: 2.5px; border-bottom: 1px solid #30363d;
    padding-bottom: 5px; margin-bottom: 14px; margin-top: 24px;
}
.algo-box {
    background: #161b22; border-left: 3px solid #00f0ff;
    border-radius: 0 8px 8px 0; padding: 9px 13px;
    font-size: 0.78rem; color: #000000; margin-bottom: 10px;
}
.update-rule {
    background: #0d2040; border: 1px solid #1f3a60;
    border-radius: 6px; padding: 7px 11px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem; color: #79c0ff; margin-top: 5px;
}
.stButton > button {
    background: linear-gradient(135deg, #ff0055, #b5179e) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important; width: 100%;
}
.footer {
    text-align: center; color: #484f58; font-size: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 18px 0 6px; border-top: 1px solid #21262d; margin-top: 36px;
}
</style>
""", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state["results"] = []

with st.sidebar:
    st.markdown("##  Configuration")
    st.markdown("---")

    chosen_env  = st.selectbox("Environment", list(ENV_INFO.keys()), index=0)
    env_meta    = ENV_INFO[chosen_env]
    chosen_algo = st.selectbox("Algorithm",   env_meta["algorithms"], index=0)

    ainfo = ALGO_INFO[chosen_algo]
    st.markdown(
        f"<div class='algo-box'><b>{chosen_algo}</b> "
        f"<span style='color:#8b949e;font-size:0.70rem;'>({ainfo['type']})</span><br>"
        f"<span style='font-size:0.73rem;color:#8b949e;'>{ainfo['desc']}</span>"
        f"<div class='update-rule'>{ainfo['update']}</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("###  Hyperparameters")
    dflt = ALGO_DEFAULTS[chosen_algo]
    n_eps     = st.slider("Episodes",        200,  8000, dflt["n_episodes"], step=100)
    gamma     = st.slider("Discount  γ",     0.80, 0.999, dflt["gamma"],    step=0.001, format="%.3f")
    epsilon   = st.slider("Initial  ε",      0.50, 1.00,  dflt["epsilon"],  step=0.05)
    eps_decay = st.slider("ε Decay",         0.990, 0.9999, dflt["eps_decay"], step=0.001, format="%.4f")
    eps_min   = st.slider("ε Min",           0.001, 0.10,  dflt["eps_min"],  step=0.001, format="%.3f")

    alpha  = 0.10
    if chosen_algo not in ("DQN", "Double DQN"):
        alpha = st.slider("Learning Rate  α", 0.01, 0.90, dflt["alpha"], step=0.01)

    dqn_lr = 5e-4; batch_size = 64; buf_size = 10_000; tgt_freq = 100
    if chosen_algo in ("DQN", "Double DQN"):
        st.markdown("###  Deep RL Settings")
        dqn_lr     = st.select_slider("DQN LR",
                         options=[1e-4, 3e-4, 5e-4, 1e-3, 3e-3],
                         value=dflt["lr"], format_func=lambda x: f"{x:.0e}")
        batch_size = st.selectbox("Batch Size",  [32, 64, 128], index=1)
        buf_size   = st.selectbox("Buffer Size", [5000, 10_000, 20_000], index=1)
        tgt_freq   = st.slider("Target Sync Freq", 50, 500, dflt["tgt_freq"], step=50)

    st.markdown("---")
    train_clicked = st.button("  Train Agent",       use_container_width=True)
    clear_clicked = st.button(" Clear All Results", use_container_width=True)

if clear_clicked:
    st.session_state["results"] = []
    st.rerun()

st.markdown("<div class='dash-header'>Train and Evaluate your RL agent</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='dash-sub'>Q-Learning · SARSA · Expected SARSA · DQN · Double DQN</div>",
    unsafe_allow_html=True,
)

env_cols = st.columns(4)
for col, (env_name, meta) in zip(env_cols, ENV_INFO.items()):
    border = "border:1px solid #00f0ff;" if env_name == chosen_env else ""
    with col:
        st.markdown(
            f"<div class='metric-card' style='{border}'>"
            f"<div class='metric-label'>{env_name}</div>"
            f"<span class='pill {meta['pill']}'>{meta['algorithms'][0]}</span>"
            f"<div class='metric-desc' style='margin-top:6px;'>{meta['desc']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

if train_clicked:
    key = (chosen_env, chosen_algo)
    st.session_state["results"] = [
        r for r in st.session_state["results"]
        if (r["env"], r["algo"]) != key
    ]

    st.markdown(
        f"<div class='section-title'>Training &nbsp;·&nbsp; {chosen_env} &nbsp;·&nbsp; {chosen_algo}</div>",
        unsafe_allow_html=True,
    )
    prog_bar    = st.progress(0)
    status_text = st.empty()

    def _cb(ep, n, avg50, eps):
        prog_bar.progress(ep / n)
        status_text.markdown(
            f"<small style='color:#8b949e;font-family:monospace;'>"
            f"Episode <b>{ep}</b>/{n} &nbsp;·&nbsp; Avg(last 50): "
            f"<b style='color:#00f0ff;'>{avg50:.2f}</b>"
            f" &nbsp;·&nbsp; ε={eps:.4f}</small>",
            unsafe_allow_html=True,
        )

    try:
        agent = None
        snapshots: list = []

        if chosen_algo in ("DQN", "Double DQN"):
            agent = DQNAgent(
                env_name=chosen_env, algo=chosen_algo, n_episodes=n_eps,
                lr=dqn_lr, gamma=gamma, epsilon=epsilon,
                eps_decay=eps_decay, eps_min=eps_min,
                batch_size=batch_size, buf_size=buf_size, tgt_freq=tgt_freq,
            )
        else:
            agent = TabularAgent(
                env_name=chosen_env, algo=chosen_algo, n_episodes=n_eps,
                alpha=alpha, gamma=gamma, epsilon=epsilon,
                eps_decay=eps_decay, eps_min=eps_min,
            )

        agent.train(progress_callback=_cb)
        prog_bar.progress(1.0)

        if chosen_algo not in ("DQN", "Double DQN") and hasattr(agent, "Q"):
            snapshots = [(n_eps, agent.Q.copy())]

        eval_results = agent.evaluate(n_episodes=100)

        metrics = compute_metrics(
            rewards=agent.rewards,
            ep_lengths=agent.ep_lengths,
            train_time=agent.train_time,
            success_thr=env_meta["success_threshold"],
        )
        metrics.update(eval_results)

        result_record = {
            "env":       chosen_env,
            "algo":      chosen_algo,
            "metrics":   metrics,
            "color":     env_meta["color"],
            "Q":              getattr(agent, "Q",              None),
            "td_errors":      getattr(agent, "td_errors",      []),
            "losses":         getattr(agent, "losses",         []),
            "epsilon_hist":   getattr(agent, "epsilon_history", []),
            "snapshots":      snapshots,
        }
        st.session_state["results"].append(result_record)
        st.success(
            f"  Training complete  ·  {chosen_env} / {chosen_algo}  ·  "
            f"{agent.train_time:.1f}s  ·  Avg Reward: {metrics['avg_reward']:.2f}  ·  "
            f"Greedy Eval: {metrics['eval_avg_reward']:.2f}"
        )

    except Exception as exc:
        st.error(f"Training failed: {exc}")
        st.exception(exc)

results = st.session_state["results"]

if results:
    latest = results[-1]
    m      = latest["metrics"]
    color  = latest["color"]

    st.markdown(
        f"<div class='section-title'> Metrics  ·  {latest['env']}  ·  {latest['algo']}</div>",
        unsafe_allow_html=True,
    )

    CARD_DEFS = [
        ("avg_reward",       f"{m['avg_reward']:.2f}",        "",      "Mean reward/episode"),
        ("cumulative",       f"{m['cumulative']:.0f}",         "",      "Total reward"),
        ("success_rate",     f"{m['success_rate']:.1f}",       "%",     "Episodes ≥ threshold"),
        ("conv_speed",       f"ep {m['conv_speed']}",          "",      "90% of peak reward"),
        ("avg_ep_len",       f"{m['avg_ep_len']:.1f}",         " steps","Mean steps/episode"),
        ("variance",         f"{m['variance']:.1f}",           "",      "Policy stability"),
        ("std_dev",          f"{m['std_dev']:.1f}",            "",      "Reward Std Dev"),
        ("training_time",    f"{m['training_time']:.1f}",      "s",     "Wall-clock time"),
    ]
    row1 = st.columns(8)
    for col, (key, val, unit, desc) in zip(row1, CARD_DEFS):
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{METRICS_CATALOGUE[key]['label']}</div>"
                f"<div class='metric-value'>{val}<span class='metric-unit'>{unit}</span></div>"
                f"<div class='metric-desc'>{desc}</div></div>",
                unsafe_allow_html=True,
            )

    CARD_DEFS2 = [
        ("max_reward",       f"{m['max_reward']:.1f}",         "",      "Best single episode"),
        ("min_reward",       f"{m['min_reward']:.1f}",         "",      "Worst single episode"),
        ("median_reward",    f"{m['median_reward']:.1f}",      "",      "Median (robust avg)"),
        ("reward_iqr",       f"{m['reward_iqr']:.1f}",         "",      "IQR – spread"),
        ("first_avg_reward", f"{m['first_avg_reward']:.2f}",   "",      "First 100 eps avg"),
        ("final_avg_reward", f"{m['final_avg_reward']:.2f}",   "",      "Last 100 eps avg"),
        ("improvement_rate", f"{m['improvement_rate']:.1f}",   "%",     "First→last 100 gain"),
    ]
    row2 = st.columns(7)
    for col, (key, val, unit, desc) in zip(row2, CARD_DEFS2):
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{METRICS_CATALOGUE[key]['label']}</div>"
                f"<div class='metric-value'>{val}<span class='metric-unit'>{unit}</span></div>"
                f"<div class='metric-desc'>{desc}</div></div>",
                unsafe_allow_html=True,
            )

    if "eval_avg_reward" in m:
        st.markdown(
            f"<div style='background:#1a3a2a;border:1px solid #3fb950;"
            f"border-radius:8px;padding:10px 15px;margin-top:6px;'>"
            f"<span style='font-family:monospace;font-size:0.75rem;color:#3fb950;'>"
            f"GREEDY EVAL (100 eps, ε=0)  ·  "
            f"Avg Reward: <b>{m['eval_avg_reward']:.2f}</b>  ·  "
            f"Success: <b>{m['eval_success_rate']:.1f}%</b>  ·  "
            f"Avg Length: <b>{m['eval_avg_ep_len']:.1f} steps</b>"
            f"</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Training Overview (6 Panels)</div>", unsafe_allow_html=True)
    td_err = latest.get("td_errors") or latest.get("losses") or []
    fig_ov = make_training_overview(m, latest["env"], latest["algo"], color, td_err)
    st.pyplot(fig_ov, use_container_width=True)
    plt.close(fig_ov)

    if latest.get("epsilon_hist"):
        st.markdown("<div class='section-title'>Epsilon Decay Schedule</div>", unsafe_allow_html=True)
        fig_eps = make_epsilon_decay(latest["epsilon_hist"], latest["env"], latest["algo"])
        st.pyplot(fig_eps, use_container_width=True)
        plt.close(fig_eps)

    if latest.get("Q") is not None and latest["env"] in ("CliffWalking-v1",):
        st.markdown("<div class='section-title'> Q-table & Policy Visualisation</div>", unsafe_allow_html=True)
        fig_qt = make_qtable_heatmap(latest["Q"], latest["env"], latest["algo"])
        if fig_qt:
            st.pyplot(fig_qt, use_container_width=True)
            plt.close(fig_qt)

        qt_stats = qtable_stats(latest["Q"])
        st.markdown(
            f"<div style='font-family:monospace;font-size:0.74rem;color:#8b949e;'>"
            f"Q-table stats  ·  max={qt_stats['max_q']:.3f}  "
            f"min={qt_stats['min_q']:.3f}  "
            f"mean={qt_stats['mean_q']:.3f}  "
            f"std={qt_stats['std_q']:.3f}  "
            f"policy entropy={qt_stats['policy_entropy']:.3f}"
            f"</div>",
            unsafe_allow_html=True,
        )

    if latest.get("snapshots"):
        st.markdown("<div class='section-title'> Q-value Evolution During Training</div>", unsafe_allow_html=True)
        fig_ev = make_qtable_evolution(latest["snapshots"], latest["env"], latest["algo"])
        if fig_ev:
            st.pyplot(fig_ev, use_container_width=True)
            plt.close(fig_ev)

    td_data = latest.get("td_errors") or latest.get("losses") or []
    if td_data and any(v != 0 for v in td_data):
        st.markdown("<div class='section-title'> TD Error Decay</div>", unsafe_allow_html=True)
        fig_td = make_td_error_curve(td_data, latest["env"], latest["algo"])
        if fig_td:
            st.pyplot(fig_td, use_container_width=True)
            plt.close(fig_td)

    if len(results) > 1:
        st.markdown("<div class='section-title'> Multi-run Comparison</div>", unsafe_allow_html=True)

        rows = []
        for r in results:
            mm = r["metrics"]
            rows.append({
                "Environment":       r["env"],
                "Algorithm":         r["algo"],
                "Avg Reward":        round(mm["avg_reward"],        2),
                "Final 100 Avg":     round(mm["final_avg_reward"],  2),
                "Success %":         round(mm["success_rate"],      1),
                "Conv. ep":          mm["conv_speed"],
                "Avg Ep Len":        round(mm["avg_ep_len"],        1),
                "Variance":          round(mm["variance"],          1),
                "IQR":               round(mm["reward_iqr"],        1),
                "Improvement %":     round(mm["improvement_rate"],  1),
                "Train Time (s)":    round(mm["training_time"],     2),
                "Greedy Avg":        round(mm.get("eval_avg_reward", 0), 2),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("<div class='section-title'>Smoothed Reward Curves – All Runs</div>", unsafe_allow_html=True)
        fig_oc = make_overlay_curves(results)
        st.pyplot(fig_oc, use_container_width=True)
        plt.close(fig_oc)

        st.markdown("<div class='section-title'>Metric Comparison – Bar Charts</div>", unsafe_allow_html=True)
        fig_cb = make_comparison_bars(results)
        if fig_cb:
            st.pyplot(fig_cb, use_container_width=True)
            plt.close(fig_cb)

        st.markdown("<div class='section-title'>Normalised Metric Radar</div>", unsafe_allow_html=True)
        fig_rd = make_metric_radar(results)
        if fig_rd:
            st.pyplot(fig_rd, use_container_width=True)
            plt.close(fig_rd)

        st.markdown("<div class='section-title'>Episode Length Distribution (Box Plot)</div>", unsafe_allow_html=True)
        fig_bx = make_episode_length_dist(results)
        if fig_bx:
            st.pyplot(fig_bx, use_container_width=True)
            plt.close(fig_bx)

else:
    st.markdown(
        "<div style='text-align:center;padding:65px 20px;color:#484f58;'>"
        "<div style='font-size:3rem;'>🤖</div>"
        "<div style='font-family:monospace;font-size:0.94rem;margin-top:12px;'>"
        "Select an environment and algorithm in the sidebar,<br>"
        "then click <b style='color:#00f0ff;'> Train Agent</b>"
        "</div>"
        "<div style='font-size:0.78rem;margin-top:8px;'>"
        "15 metrics · Q-table heatmaps · policy arrows · contour plots · "
        "multi-run radar charts appear here."
        "</div></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='footer'>"
    "Train and Evaluate your RL agent <br>"
    "Python + Streamlit + NumPy + Gymnasium  ·  15 Metrics"
    "</div>",
    unsafe_allow_html=True,
)
