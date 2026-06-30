# -*- coding: utf-8 -*-
"""
ppo_convergence.py — raccolta metriche di convergenza PPO + grafici dedicati.

Due cose:

1) `train_loop_rich(algo, n_iterations, policy_id, ...)`
   Drop-in replacement di mappo_trainer.train_loop che, oltre a
   reward/timesteps/elapsed, estrae a OGNI iterazione le metriche interne del
   learner PPO (policy loss, vf loss, entropy, KL, explained variance,
   total loss, e — se presente — bc_kl/bc_kl_beta del Fix B) e i timer
   (sampling vs learner update). Restituisce la stessa lista-di-dict di prima,
   con chiavi extra. Retro-compatibile: i grafici vecchi continuano a leggere
   `episode_reward_mean`.

2) Funzioni di plotting (PNG+PDF, stesso stile delle altre figure del paper)
   da chiamare in analysis_step8.make_paper_charts:
     - plot_ppo_convergence_panel : 6 pannelli (reward, policy loss, vf loss,
       entropy, KL+bc_kl, explained var) per vanilla e warm su stesso asse iter
     - plot_training_time_breakdown : barre impilate dei tempi (demo gen,
       BC train, PPO vanilla, PPO warm, eval) + tabella sotto
     - plot_ppo_throughput : timesteps/s e sampling-vs-update time per iter

Tutte usano [unità tra parentesi quadre] e griglia tenue come le altre.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Palette coerente con le altre figure del paper
# ----------------------------------------------------------------------------
COLORS = {
    "ppo_vanilla": "#E67E22",   # orange (come negli altri chart)
    "ppo_bc":      "#C0392B",   # red (PPO BC warmstart)
    "milp":        "#2E86C1",
    "bc":          "#27AE60",
    "random":      "#7F8C8D",
}
GRID_KW = dict(alpha=0.18, linewidth=0.6)


def _save(fig, out_dir: str, name: str, dpi_png: int = 300, dpi_pdf: int = 500):
    png_dir = os.path.join(out_dir, "png")
    pdf_dir = os.path.join(out_dir, "pdf")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    fig.savefig(os.path.join(png_dir, f"{name}.png"), dpi=dpi_png, bbox_inches="tight")
    fig.savefig(os.path.join(pdf_dir, f"{name}.pdf"), dpi=dpi_pdf, bbox_inches="tight",
                format="pdf")
    plt.close(fig)


# ============================================================================
# 1) Rich training loop
# ============================================================================

def _get(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def train_loop_rich(
    algo,
    n_iterations: int,
    policy_id: str = "shared_policy",
    log_every: int = 10,
    log_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Come train_loop, ma raccoglie anche le metriche interne del learner.

    Ogni entry contiene (oltre alle vecchie chiavi):
      policy_loss, vf_loss, vf_explained_var, entropy, mean_kl_loss,
      total_loss, curr_entropy_coeff, bc_kl, bc_kl_beta,
      sampling_time_s, learner_update_time_s, timesteps_throughput
    """
    metrics: List[Dict[str, Any]] = []
    t0 = time.time()
    for i in range(n_iterations):
        result = algo.train()
        env_runners = result.get("env_runners", {})
        learners = result.get("learners", result.get("learner", {})) or {}
        lp = learners.get(policy_id, {})
        if not lp and learners:
            # fall back to the first non-meta module key
            for k, v in learners.items():
                if isinstance(v, dict) and not k.startswith("__"):
                    lp = v
                    break
        timers = result.get("timers", {})

        mean_rew = _get(env_runners, "episode_return_mean",
                        default=result.get("episode_reward_mean",
                                           result.get("episode_return_mean")))
        min_rew = _get(env_runners, "episode_return_min",
                       default=result.get("episode_reward_min"))
        max_rew = _get(env_runners, "episode_return_max",
                       default=result.get("episode_reward_max"))
        ts = result.get("num_env_steps_sampled_lifetime",
                        result.get("timesteps_total", 0))
        eps = _get(env_runners, "num_episodes", default=result.get("episodes_total", 0))

        def lf(key):  # learner float, safe
            v = lp.get(key)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        entry = {
            "iteration": i,
            "episode_reward_mean": mean_rew,
            "episode_reward_min": min_rew,
            "episode_reward_max": max_rew,
            "episodes_total": eps,
            "timesteps_total": ts,
            "elapsed_seconds": time.time() - t0,
            # --- learner internals ---
            "policy_loss": lf("policy_loss"),
            "vf_loss": lf("vf_loss"),
            "vf_explained_var": lf("vf_explained_var"),
            "entropy": lf("entropy"),
            "mean_kl_loss": lf("mean_kl_loss"),
            "total_loss": lf("total_loss"),
            "curr_entropy_coeff": lf("curr_entropy_coeff"),
            "bc_kl": lf("bc_kl"),
            "bc_kl_beta": lf("bc_kl_beta"),
            # --- timers ---
            "sampling_time_s": float(timers.get("env_runner_sampling_timer", 0.0) or 0.0),
            "learner_update_time_s": float(timers.get("learner_update_timer", 0.0) or 0.0),
        }
        metrics.append(entry)
        if log_callback is not None:
            log_callback(entry)
        if verbose and ((i + 1) % log_every == 0 or i == 0):
            mr_s = f"{mean_rew:.2f}" if mean_rew is not None else "None"
            pl_s = f"{entry['policy_loss']:.3f}" if entry['policy_loss'] is not None else "—"
            kl_s = f"{entry['bc_kl']:.4f}" if entry['bc_kl'] is not None else "—"
            print(f"  iter {i+1:3d}: mean_rew={mr_s}  pol_loss={pl_s}  "
                  f"bc_kl={kl_s}  ts={ts}  elapsed={entry['elapsed_seconds']:.1f}s")
    return metrics


# ============================================================================
# 2) Convergence panel (6 subplots)
# ============================================================================

def _series(metrics: List[Dict[str, Any]], key: str):
    xs, ys = [], []
    for m in metrics:
        v = m.get(key)
        if v is not None:
            xs.append(m["iteration"] + 1)
            ys.append(v)
    return np.array(xs), np.array(ys)


def plot_ppo_convergence_panel(
    out_dir: str,
    vanilla_metrics: List[Dict[str, Any]],
    warm_metrics: Optional[List[Dict[str, Any]]] = None,
    name: str = "ppo_convergence_panel",
):
    """6-pannelli di convergenza: reward, policy loss, vf loss, entropy,
    KL (PPO mean_kl + bc_kl anchor), explained variance."""
    panels = [
        ("episode_reward_mean", "Mean episode return [EUR]", False),
        ("policy_loss",         "Policy surrogate loss [-]", False),
        ("vf_loss",             "Value function loss [-]",  True),
        ("entropy",             "Policy entropy [nats]",     False),
        ("bc_kl",               "KL anchor to BC [nats]",    False),
        ("vf_explained_var",    "VF explained variance [-]", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()
    for ax, (key, ylabel, logy) in zip(axes, panels):
        xv, yv = _series(vanilla_metrics, key)
        if len(xv):
            ax.plot(xv, yv, color=COLORS["ppo_vanilla"], lw=1.8, marker="o",
                    ms=2.5, label="PPO vanilla")
        if warm_metrics is not None:
            xw, yw = _series(warm_metrics, key)
            if len(xw):
                ax.plot(xw, yw, color=COLORS["ppo_bc"], lw=1.8, marker="o",
                        ms=2.5, label="PPO (BC warmstart)")
            # overlay the PPO native KL on the KL panel for context
            if key == "bc_kl":
                xk, yk = _series(warm_metrics, "mean_kl_loss")
                if len(xk):
                    ax.plot(xk, yk, color="#7D3C98", lw=1.2, ls="--",
                            label="PPO native KL (warm)")
                # secondary axis: beta schedule
                xb, yb = _series(warm_metrics, "bc_kl_beta")
                if len(xb):
                    ax2 = ax.twinx()
                    ax2.plot(xb, yb, color="#555555", lw=1.0, ls=":",
                             label=r"$\beta$ schedule")
                    ax2.set_ylabel(r"anchor weight $\beta$ [-]", fontsize=9)
                    ax2.tick_params(labelsize=8)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("PPO training iteration [-]", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, **GRID_KW)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("PPO training convergence diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out_dir, name)


# ============================================================================
# 3) Training time breakdown
# ============================================================================

def plot_training_time_breakdown(
    out_dir: str,
    timing: Dict[str, float],
    ppo_vanilla_seconds: Optional[float] = None,
    ppo_warm_seconds: Optional[float] = None,
    name: str = "training_time_breakdown",
):
    """Barra impilata orizzontale dei tempi di pipeline + annotazioni.

    `timing` è il dict di metrics.json: demo_generation_seconds,
    bc_training_seconds, evaluation_seconds, total_seconds.
    I tempi PPO, se passati, vengono mostrati come segmenti separati
    (estratti dall'elapsed dell'ultima iterazione di ciascun training).
    """
    segs = []
    segs.append(("MILP demo generation", timing.get("demo_generation_seconds", 0.0), "#2E86C1"))
    segs.append(("BC training",           timing.get("bc_training_seconds", 0.0),     "#27AE60"))
    if ppo_vanilla_seconds:
        segs.append(("PPO vanilla training", float(ppo_vanilla_seconds), COLORS["ppo_vanilla"]))
    if ppo_warm_seconds:
        segs.append(("PPO BC-warm training", float(ppo_warm_seconds), COLORS["ppo_bc"]))
    segs.append(("Evaluation (all policies)", timing.get("evaluation_seconds", 0.0), "#8E44AD"))

    fig, ax = plt.subplots(figsize=(12, 3.2))
    left = 0.0
    total = sum(s[1] for s in segs)
    for label, val, color in segs:
        if val <= 0:
            continue
        ax.barh(0, val, left=left, color=color, edgecolor="white", height=0.6, label=label)
        if val / max(total, 1e-9) > 0.04:
            ax.text(left + val / 2, 0, f"{val/60:.1f} min", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        left += val
    ax.set_xlim(0, total * 1.02)
    ax.set_yticks([])
    ax.set_xlabel("Wall-clock time [s]", fontsize=11)
    ax.set_title(f"End-to-end pipeline timing  (total {total/60:.1f} min)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", **GRID_KW)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    fig.tight_layout()
    _save(fig, out_dir, name)


# ============================================================================
# 4) Throughput / sampling-vs-update
# ============================================================================

def plot_ppo_throughput(
    out_dir: str,
    vanilla_metrics: List[Dict[str, Any]],
    warm_metrics: Optional[List[Dict[str, Any]]] = None,
    name: str = "ppo_throughput",
):
    """Due pannelli: (a) cumulative wall-clock vs iteration; (b) per-iter time
    split sampling vs learner update (mean over run, bar)."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 4.5))

    # (a) cumulative time
    for mets, col, lab in [(vanilla_metrics, COLORS["ppo_vanilla"], "PPO vanilla"),
                           (warm_metrics, COLORS["ppo_bc"], "PPO (BC warmstart)")]:
        if not mets:
            continue
        x, y = _series(mets, "elapsed_seconds")
        if len(x):
            axL.plot(x, y / 60.0, color=col, lw=1.8, marker="o", ms=2.5, label=lab)
    axL.set_xlabel("PPO training iteration [-]", fontsize=10)
    axL.set_ylabel("Cumulative wall-clock [min]", fontsize=10)
    axL.set_title("Training time vs iteration", fontsize=12, fontweight="bold")
    axL.grid(True, **GRID_KW); axL.legend(fontsize=9)

    # (b) sampling vs update split (mean)
    runs = [("vanilla", vanilla_metrics, COLORS["ppo_vanilla"]),
            ("BC-warm", warm_metrics, COLORS["ppo_bc"])]
    labels, samp, upd = [], [], []
    for lab, mets, _ in runs:
        if not mets:
            continue
        s = np.mean([m["sampling_time_s"] for m in mets if m.get("sampling_time_s")])
        u = np.mean([m["learner_update_time_s"] for m in mets if m.get("learner_update_time_s")])
        labels.append(lab); samp.append(s); upd.append(u)
    if labels:
        xpos = np.arange(len(labels))
        axR.bar(xpos, samp, 0.5, label="env sampling", color="#2E86C1")
        axR.bar(xpos, upd, 0.5, bottom=samp, label="learner update", color="#C0392B")
        axR.set_xticks(xpos); axR.set_xticklabels(labels, fontsize=10)
        axR.set_ylabel("Mean time per iteration [s]", fontsize=10)
        axR.set_title("Per-iteration time breakdown", fontsize=12, fontweight="bold")
        axR.grid(True, axis="y", **GRID_KW); axR.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, name)
