"""
Post-pipeline analysis: terminal behavioural reports and paper-ready charts.

Consumes the `trajectories` dict produced by run_full_pipeline (Step 8) and
exposes two top-level functions:

  - print_market_behaviour_report(trajectories, fleet, output_dir=None):
      formatted multi-page terminal report showing, per policy:
        * average hourly bidding profile (24-hour clock)
        * service utilization (% of hours bidding each service above threshold)
        * action bin histogram per axis
        * SOC range and average
        * total throughput and capacity factor
        * per-service revenue and cost decomposition
      The same content is also dumped to behaviour_report.txt in output_dir.

  - make_paper_charts(trajectories, output_dir, result=None):
      writes six PNG charts to output_dir:
        1. daily_profits.png  - per-day profit comparison across policies
        2. hourly_action_profile.png  - mean action fraction by hour-of-day,
           one subplot per axis (charge/discharge/FCR/aFRR/mFRR)
        3. soc_trajectory.png  - SOC mean (with min/max band) over the test
           window, one panel per policy
        4. service_revenue_breakdown.png  - stacked bar of revenue
           components (arbitrage / flex / -degradation) per policy
        5. cumulative_profit.png  - cumulative env-realised reward over time
        6. action_bin_histogram.png  - histogram of bin selection per axis,
           one subplot per axis, one bar series per policy
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Use a non-interactive backend so this works in headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Policy ordering and colours kept consistent across all charts
_POLICY_ORDER = ["milp", "bc", "ppo_bc", "ppo_vanilla", "random"]
_POLICY_LABELS = {
    "milp": "MILP",
    "bc": "BC policy",
    "ppo_bc": "PPO (BC warmstart)",
    "ppo_vanilla": "PPO vanilla",
    "random": "Random baseline",
}
_POLICY_COLORS = {
    "milp": "#1f77b4",       # blue
    "bc": "#2ca02c",         # green
    "ppo_bc": "#d62728",     # red
    "ppo_vanilla": "#ff7f0e",  # orange
    "random": "#7f7f7f",     # grey
}
_AXIS_LABELS = ["charge", "discharge", "FCR", "aFRR", "mFRR"]
_AXIS_LABELS_DIRECTIONAL = ["charge", "discharge", "FCR",
                             "aFRR up", "aFRR dn", "mFRR up", "mFRR dn"]


def _present_policies(trajectories: Dict[str, Any]) -> list:
    """Return the subset of _POLICY_ORDER that exists in the trajectory dict."""
    return [p for p in _POLICY_ORDER
            if p in trajectories and trajectories[p] is not None]


# ============================================================================
# Terminal report
# ============================================================================

def print_market_behaviour_report(
    trajectories: Dict[str, Dict[str, Any]],
    fleet,
    output_dir: Optional[Path] = None,
) -> None:
    """Print and (optionally) write a detailed per-policy behavioural report.

    Parameters
    ----------
    trajectories : dict of policy_name -> trajectory_dict
        As produced by run_full_pipeline. Each trajectory_dict has
        actions_mean_per_hour, action_bin_counts, soc_*, *_per_hour, etc.
    fleet : MultiBatteryParameters
        Used to compute capacity factor and total throughput limits.
    output_dir : Path, optional
        If provided, the full report is also written to
        `output_dir / behaviour_report.txt`.
    """
    policies = _present_policies(trajectories)
    if not policies:
        print("(no trajectories to report)")
        return

    fleet_total_capacity_mwh = float(sum(b.capacity_mwh for b in fleet.batteries))
    fleet_total_power_mw = float(sum(b.max_power_mw for b in fleet.batteries))
    N = fleet.n_batteries

    lines = []
    def out(s=""):
        lines.append(s)
        print(s)

    out("=" * 78)
    out("MARKET BEHAVIOUR REPORT")
    out("=" * 78)
    out(f"Fleet: N={N} BESS, total capacity {fleet_total_capacity_mwh:.1f} MWh, "
        f"total power {fleet_total_power_mw:.1f} MW")
    n_hours = None
    for p in policies:
        t = trajectories[p]
        if n_hours is None:
            n_hours = len(t["actions_mean_per_hour"])
            break
    if n_hours is not None:
        out(f"Test window: {n_hours} hours = {n_hours // 24} days")
    out("")

    for p in policies:
        t = trajectories[p]
        label = _POLICY_LABELS[p]
        out("-" * 78)
        out(f"  {label}")
        out("-" * 78)

        # ---- Hourly bidding profile averaged over the test window ----
        actions = t["actions_mean_per_hour"]   # (n_hours, n_axes) in [0,1]
        n_hours, n_axes = actions.shape
        axis_labels = (_AXIS_LABELS_DIRECTIONAL if n_axes == 7 else _AXIS_LABELS)
        n_days = n_hours // 24
        actions_24h = actions[:n_days * 24].reshape(n_days, 24, n_axes).mean(axis=0)
        out("  Mean bidding by hour of day (% of P_max):")
        # Header row dynamically built from axis_labels (truncate to fit ~5 chars)
        header_cells = "  ".join(f"{lab[:6]:>6s}" for lab in axis_labels)
        out(f"    hour | {header_cells}")
        for h in range(24):
            row = actions_24h[h] * 100
            cells = "  ".join(f"{row[k]:>6.1f}" for k in range(n_axes))
            out(f"    {h:>4d} | {cells}")

        # ---- Service utilization (% of hours bidding > 5% of P_max) ----
        threshold = 0.05
        util = (actions > threshold).mean(axis=0) * 100
        out("")
        out("  Service utilization (% of hours bidding > 5% of P_max):")
        for k, lab in enumerate(axis_labels):
            out(f"    {lab:<12s}: {util[k]:>5.1f}%")

        # ---- Action bin histogram per axis (in percent) ----
        bin_counts = t["action_bin_counts"]   # (n_axes, 11)
        total_per_axis = bin_counts.sum(axis=1, keepdims=True)
        hist = (bin_counts / np.maximum(total_per_axis, 1)) * 100
        out("")
        out("  Action bin distribution per axis (% of all (BESS, hour) decisions):")
        n_bins_rep = bin_counts.shape[1]
        header = "    axis         | " + "  ".join(f"bin{b}" for b in range(n_bins_rep))
        out(header)
        for k, label in enumerate(axis_labels):
            row = "  ".join(f"{hist[k, b]:>5.1f}" for b in range(n_bins_rep))
            out(f"    {label:<12s} | {row}")

        # ---- SOC range ----
        soc_mean = t["soc_mean"]
        soc_min = t["soc_min"]
        soc_max = t["soc_max"]
        out("")
        out(f"  SOC across fleet: mean {soc_mean.mean():.3f}, "
            f"observed range [{soc_min.min():.3f}, {soc_max.max():.3f}]")

        # ---- Throughput and capacity factor ----
        # Throughput per hour aggregated = (charge_frac + discharge_frac) * P_max * 1h
        # (approximate; ignores flex activation throughput which lives in env state)
        hourly_throughput_mwh = (
            (actions[:, 0] + actions[:, 1]) * fleet_total_power_mw
        )
        total_throughput = float(hourly_throughput_mwh.sum())
        avg_daily_throughput = total_throughput / max(n_days, 1)
        # Capacity factor of bidding: average bid power as fraction of P_max
        bidding_load = actions[:, 2:5].sum(axis=1).mean()  # average flex bidding
        out("")
        out(f"  Total charge+discharge throughput: {total_throughput:,.0f} MWh "
            f"({avg_daily_throughput:,.0f} MWh/day on average)")
        out(f"  Average flex bidding (mean of FCR+aFRR+mFRR fraction): "
            f"{bidding_load*100:.1f}% of P_max")

        # ---- Revenue and degradation ----
        arb = float(t["arbitrage_per_hour"].sum())
        flex = float(t["flex_per_hour"].sum())
        deg = float(t["deg_per_hour"].sum())
        net = arb + flex - deg
        out("")
        out(f"  Revenue and cost decomposition (test window):")
        out(f"    arbitrage:    {arb:>14,.0f} EUR")
        out(f"    flex revenue: {flex:>14,.0f} EUR")
        out(f"    degradation:  {deg:>14,.0f} EUR")
        out(f"    NET PROFIT:   {net:>14,.0f} EUR")
        out("")

    out("=" * 78)

    # Persist to file if requested
    if output_dir is not None:
        out_path = Path(output_dir) / "behaviour_report.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Saved: {out_path}")


# ============================================================================
# Charts
# ============================================================================

def make_paper_charts(
    trajectories: Dict[str, Dict[str, Any]],
    output_dir: Path,
    result=None,
    ppo_training_metrics: Optional[Dict[str, list]] = None,
) -> None:
    """Generate the six paper charts as PNG files in output_dir.

    Parameters
    ----------
    trajectories : dict of policy_name -> trajectory_dict
    output_dir : Path
        Directory where to write the PNGs.
    result : FullPipelineResult, optional
        If supplied, used to enrich charts with the headline numbers and the
        per-day MILP/BC daily profits.
    ppo_training_metrics : dict, optional
        Mapping {"ppo_vanilla": [...], "ppo_bc": [...]} where each value is
        a list of per-iteration mean rewards. Used for the learning curves
        chart. If None, that chart is skipped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create png/ and pdf/ subdirectories: each chart is saved in both formats.
    png_dir = output_dir / "png"
    pdf_dir = output_dir / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig, name: str, dpi: int = 140, bbox_inches: Optional[str] = None):
        """Save figure as both PNG (in png/) and PDF (in pdf/)."""
        kwargs = {"dpi": dpi}
        if bbox_inches is not None:
            kwargs["bbox_inches"] = bbox_inches
        fig.savefig(png_dir / f"{name}.png", **kwargs)
        # PDFs are vector: dpi has no effect, drop it to avoid warnings
        kwargs.pop("dpi", None)
        fig.savefig(pdf_dir / f"{name}.pdf", **kwargs)
        plt.close(fig)

    policies = _present_policies(trajectories)
    if not policies:
        print("(no trajectories: skipping charts)")
        return

    # ---- 1. Daily profits across the test window ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for p in policies:
        t = trajectories[p]
        hourly_net = (t["arbitrage_per_hour"] + t["flex_per_hour"]
                      - t["deg_per_hour"])
        n_hours = len(hourly_net)
        n_days = n_hours // 24
        daily = hourly_net[:n_days * 24].reshape(n_days, 24).sum(axis=1)
        ax.plot(np.arange(1, n_days + 1), daily,
                label=_POLICY_LABELS[p], color=_POLICY_COLORS[p],
                linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Day of test window [days]")
    ax.set_ylabel("Daily net profit [EUR]")
    ax.set_title("Daily net profit by policy on test window")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "daily_profits", dpi=120)

    # ---- 2. Hourly bidding profile by hour-of-day, one subplot per axis ----
    # Detect n_axes from any present trajectory
    sample_actions = trajectories[policies[0]]["actions_mean_per_hour"]
    n_axes = sample_actions.shape[1]
    axis_labels = (_AXIS_LABELS_DIRECTIONAL if n_axes == 7 else _AXIS_LABELS)
    fig, axes = plt.subplots(1, n_axes, figsize=(4 * n_axes, 4), sharey=True)
    if n_axes == 1:
        axes = [axes]
    for k, ax_k in enumerate(axes):
        for p in policies:
            t = trajectories[p]
            actions = t["actions_mean_per_hour"]
            n_hours = len(actions)
            n_days = n_hours // 24
            profile = actions[:n_days * 24].reshape(n_days, 24, n_axes).mean(axis=0)[:, k]
            ax_k.plot(np.arange(24), profile * 100,
                      label=_POLICY_LABELS[p], color=_POLICY_COLORS[p],
                      linewidth=1.5)
        ax_k.set_xlabel("Hour of day [h]")
        ax_k.set_title(axis_labels[k])
        ax_k.grid(True, alpha=0.3)
        ax_k.set_xticks([0, 6, 12, 18, 23])
    axes[0].set_ylabel("Mean bidding [% of P_max]")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Bidding profile by hour of day (averaged across test days)",
                 y=1.02)
    fig.tight_layout()
    _save(fig, "hourly_action_profile", dpi=120, bbox_inches="tight")

    # ---- 3. SOC trajectory over time, one panel per policy ----
    fig, axes = plt.subplots(len(policies), 1, figsize=(12, 2.5 * len(policies)),
                              sharex=True)
    if len(policies) == 1:
        axes = [axes]
    for ax_p, p in zip(axes, policies):
        t = trajectories[p]
        hours = np.arange(len(t["soc_mean"]))
        ax_p.fill_between(hours, t["soc_min"], t["soc_max"],
                           alpha=0.25, color=_POLICY_COLORS[p],
                           label="min-max across fleet")
        ax_p.plot(hours, t["soc_mean"], color=_POLICY_COLORS[p],
                   linewidth=1.0, label="mean across fleet")
        ax_p.set_ylabel(f"SOC [-]\n{_POLICY_LABELS[p]}")
        ax_p.set_ylim(0, 1)
        ax_p.grid(True, alpha=0.3)
        ax_p.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Hour of test window [h]")
    fig.suptitle("Fleet-wide SOC trajectory by policy", y=0.995)
    fig.tight_layout()
    _save(fig, "soc_trajectory", dpi=120)

    # ---- 4. Revenue breakdown by service (per policy, stacked bar) ----
    # Composition: arbitrage + FCR + aFRR (split by direction if directional)
    # + mFRR (idem). NO degradation subtracted: this chart shows GROSS revenue
    # composition. Net profit appears in cumulative_profit and daily_profits.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = np.arange(len(policies))
    bar_w = 0.55

    # Auto-detect directional vs legacy from presence of rev_afrr_up arrays
    sample_t = trajectories[policies[0]]
    is_directional_rev = (
        "rev_afrr_up_per_hour" in sample_t
        and sample_t["rev_afrr_up_per_hour"].size > 0
    )

    # Color palette: arbitrage neutral blue, FCR teal, aFRR shades of green,
    # mFRR shades of orange. Up = darker, dn = lighter for each direction.
    if is_directional_rev:
        components = [
            ("Arbitrage",  "arbitrage_per_hour",  "#1f77b4"),
            ("FCR",        "rev_fcr_per_hour",    "#17becf"),
            ("aFRR up",    "rev_afrr_up_per_hour", "#2ca02c"),
            ("aFRR dn",    "rev_afrr_dn_per_hour", "#98df8a"),
            ("mFRR up",    "rev_mfrr_up_per_hour", "#ff7f0e"),
            ("mFRR dn",    "rev_mfrr_dn_per_hour", "#ffbb78"),
        ]
    else:
        components = [
            ("Arbitrage", "arbitrage_per_hour", "#1f77b4"),
            ("FCR",       "rev_fcr_per_hour",   "#17becf"),
            ("aFRR",      "rev_afrr_per_hour",  "#2ca02c"),
            ("mFRR",      "rev_mfrr_per_hour",  "#ff7f0e"),
        ]

    # Sum each component per policy
    sums = {p: {} for p in policies}
    for p in policies:
        t = trajectories[p]
        for label, key, _ in components:
            arr = t.get(key)
            sums[p][label] = float(arr.sum()) if arr is not None and arr.size > 0 else 0.0

    # Stacked bars: separate positive and negative stacks so arbitrage<0
    # does not visually break the chart (random can have negative arbitrage)
    pos_bottom = np.zeros(len(policies))
    neg_bottom = np.zeros(len(policies))
    for label, _, color in components:
        vals = np.array([sums[p][label] for p in policies], dtype=np.float64)
        pos_vals = np.where(vals >= 0, vals, 0.0)
        neg_vals = np.where(vals < 0, vals, 0.0)
        ax.bar(xs, pos_vals, bar_w, bottom=pos_bottom,
                label=label, color=color, edgecolor="white", linewidth=0.5)
        ax.bar(xs, neg_vals, bar_w, bottom=neg_bottom,
                color=color, edgecolor="white", linewidth=0.5)
        pos_bottom += pos_vals
        neg_bottom += neg_vals

    # Annotate gross total revenue (sum of arbitrage + all flex, no degradation)
    for i, p in enumerate(policies):
        gross = sum(sums[p].values())
        top = pos_bottom[i] + max(0, abs(neg_bottom[i]) * 0.02)
        ax.text(xs[i], top + 50, f"{gross:,.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([_POLICY_LABELS[p] for p in policies],
                       rotation=12, ha="right")
    ax.set_ylabel("Revenue [EUR]")
    ax.set_title("Gross revenue breakdown by policy and service (test window total)")
    ax.legend(loc="upper right", fontsize=9, ncol=2 if is_directional_rev else 1,
              framealpha=0.95)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    _save(fig, "service_revenue_breakdown", dpi=120)

    # ---- 5. Cumulative profit over time ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for p in policies:
        t = trajectories[p]
        hourly_net = (t["arbitrage_per_hour"] + t["flex_per_hour"]
                      - t["deg_per_hour"])
        cum = np.cumsum(hourly_net)
        ax.plot(np.arange(len(cum)) / 24.0, cum,
                label=_POLICY_LABELS[p], color=_POLICY_COLORS[p],
                linewidth=1.5)
    ax.set_xlabel("Day of test window [days]")
    ax.set_ylabel("Cumulative net profit [EUR]")
    ax.set_title("Cumulative env-realised profit by policy")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "cumulative_profit", dpi=120)

    # ---- 6. Action bin histogram per axis ----
    n_axes_hist = trajectories[policies[0]]["action_bin_counts"].shape[0]
    axis_labels_hist = (_AXIS_LABELS_DIRECTIONAL if n_axes_hist == 7 else _AXIS_LABELS)
    fig, axes = plt.subplots(1, n_axes_hist, figsize=(4 * n_axes_hist, 4))
    if n_axes_hist == 1:
        axes = [axes]
    n_bins_hist = trajectories[policies[0]]["action_bin_counts"].shape[1]
    bins = np.arange(n_bins_hist)
    bar_width = 0.8 / len(policies)
    for k, ax_k in enumerate(axes):
        for ip, p in enumerate(policies):
            counts = trajectories[p]["action_bin_counts"][k]
            total = max(counts.sum(), 1)
            ax_k.bar(bins + ip * bar_width - 0.4, counts / total * 100,
                      bar_width, label=_POLICY_LABELS[p],
                      color=_POLICY_COLORS[p], alpha=0.85)
        ax_k.set_xlabel(f"Bin [0=zero, {n_bins_hist - 1}=full P_max]")
        ax_k.set_title(axis_labels_hist[k])
        ax_k.set_xticks(bins)
        ax_k.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel("Frequency [% of decisions]")
    axes[-1].legend(loc="best", fontsize=7)
    fig.suptitle("Action bin distribution by policy (per axis)", y=1.02)
    fig.tight_layout()
    _save(fig, "action_bin_histogram", dpi=120, bbox_inches="tight")

    # ---- (Optional) 7. PPO learning curves ----
    if ppo_training_metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        for tag, rewards in ppo_training_metrics.items():
            if not rewards:
                continue
            ax.plot(np.arange(1, len(rewards) + 1), rewards,
                    label=_POLICY_LABELS.get(tag, tag),
                    color=_POLICY_COLORS.get(tag, None),
                    linewidth=1.5, marker="o", markersize=3)
        ax.set_xlabel("PPO training iteration [-]")
        ax.set_ylabel("Mean episode return [EUR]")
        ax.set_title("PPO learning curves on train window")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, "ppo_learning_curves", dpi=120)

    # ---- 8. SOH comparison across policies ----
    # Line plot of mean SOH over the test window, with min/max band per policy.
    # Highlights which policy degrades the fleet the most.
    if all("soh_mean" in trajectories[p] for p in policies):
        fig, ax = plt.subplots(figsize=(11, 6))
        for p in policies:
            t = trajectories[p]
            n_hours = len(t["soh_mean"])
            days = np.arange(n_hours) / 24.0
            soh_pct = t["soh_mean"] * 100.0
            soh_min_pct = t["soh_min"] * 100.0
            soh_max_pct = t["soh_max"] * 100.0
            ax.fill_between(days, soh_min_pct, soh_max_pct,
                            alpha=0.18, color=_POLICY_COLORS[p],
                            linewidth=0)
            ax.plot(days, soh_pct, color=_POLICY_COLORS[p],
                    linewidth=2.0, label=_POLICY_LABELS[p])
            # Annotate final SOH
            if len(days) > 0:
                ax.annotate(f"{soh_pct[-1]:.3f}%",
                            xy=(days[-1], soh_pct[-1]),
                            xytext=(6, 0), textcoords="offset points",
                            fontsize=8, color=_POLICY_COLORS[p],
                            va="center", fontweight="bold")
        ax.set_xlabel("Test window [days]", fontsize=11)
        ax.set_ylabel("Battery State of Health [%]", fontsize=11)
        ax.set_title("SOH degradation across policies "
                     "(mean across fleet; shaded = min/max range)",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="lower left", fontsize=10, framealpha=0.92)
        ax.grid(True, alpha=0.35, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        _save(fig, "soh_comparison", dpi=140, bbox_inches="tight")

    # ---- 9. Inference benchmark: MILP vs DRL ----
    # Use the FullPipelineResult timings if available; otherwise run a small
    # synthetic benchmark for a few fleet sizes to compare scaling.
    inference_data = _collect_inference_timings(result)
    if inference_data is not None:
        fleet_sizes = inference_data["fleet_sizes"]
        milp_ms = inference_data["milp_ms"]
        drl_ms = inference_data["drl_ms"]

        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(fleet_sizes))
        width = 0.36
        bars_m = ax.bar(x - width / 2, milp_ms, width,
                         label="MILP daily solve",
                         color="#1f77b4", edgecolor="black", linewidth=0.5)
        bars_d = ax.bar(x + width / 2, drl_ms, width,
                         label="DRL per-decision forward pass",
                         color="#d62728", edgecolor="black", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"N={n}" for n in fleet_sizes], fontsize=11)
        ax.set_xlabel("Fleet size [BESS]", fontsize=11)
        ax.set_ylabel("Inference time per decision [ms, log scale]",
                      fontsize=11)
        ax.set_title("MILP vs DRL inference cost across fleet sizes",
                     fontsize=12, fontweight="bold")
        for i, (m, d) in enumerate(zip(milp_ms, drl_ms)):
            if d > 0:
                ratio = m / d
                ax.text(i, max(m, d) * 1.6,
                        f"{ratio:.0f}\u00d7", ha="center", fontsize=10,
                        fontweight="bold", color="#2ca02c")
        for bars in (bars_m, bars_d):
            for b in bars:
                h = b.get_height()
                ax.annotate(f"{h:.2f}" if h < 10 else f"{h:.0f}",
                            xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.92)
        # Fainter grid lines per Lorenzo's request
        ax.grid(True, alpha=0.12, axis="y", which="major", linestyle="--",
                linewidth=0.6)
        ax.grid(True, alpha=0.06, axis="y", which="minor", linestyle=":",
                linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        _save(fig, "inference_benchmark", dpi=140, bbox_inches="tight")

    # ---- 10. Carpet plot: market activity + congestion relationship ----
    _make_carpet_plot(trajectories, policies, _save)

    # ---- 11. Pareto frontier: profit vs degradation ----
    _make_pareto_profit_degradation(trajectories, policies, _save)

    # ---- 12. BC training curves: loss + val accuracy per axis ----
    _make_bc_training_curves(result, _save)

    # ---- 13. BC vs MILP action confusion matrix ----
    _make_bc_confusion_matrix(result, _save)

    # ---- 14. Spread aFRR vs PUN (hour-of-day price profile) ----
    _make_price_spread_hourly(trajectories, _save)

    # ---- 15. ECDF of daily profits per policy ----
    _make_daily_profit_ecdf(trajectories, policies, _save)

    # ---- 16. SOC distribution per policy (violin) ----
    _make_soc_distribution(trajectories, policies, _save)

    # ---- 17. Service mix donut per policy ----
    _make_service_mix_donut(trajectories, policies, _save)

    # ---- 18. Hour-of-day radar of bid intensity ----
    _make_hourly_bid_radar(trajectories, policies, _save)

    n_png = len(list(png_dir.glob('*.png')))
    n_pdf = len(list(pdf_dir.glob('*.pdf')))
    print(f"Saved {n_png} PNG chart(s) to {png_dir}")
    print(f"Saved {n_pdf} PDF chart(s) to {pdf_dir}")


def _make_carpet_plot(trajectories: Dict[str, Dict[str, Any]],
                       policies: list,
                       save_fn) -> None:
    """Carpet plots: hour-of-day (y-axis) × day-of-window (x-axis) heatmaps,
    one row per service plus a top row showing PUN price (congestion proxy).
    Columns = policies. Reveals temporal alignment between market congestion
    and the policies' bidding response.

    Three colormaps used:
      - PUN row: divergent RdBu_r centered on the test-window mean PUN.
        Red cells = price ABOVE mean (scarcity / upward-needed regime).
        Blue cells = price BELOW mean (excess / downward-needed regime).
      - Service rows: sequential (one per service family) mapping bid in MW.

    If trajectory has the directional bid arrays, plots 6 service rows
    (FCR, aFRR up, aFRR dn, mFRR up, mFRR dn). Otherwise 3 (FCR, aFRR, mFRR).
    """
    if not policies:
        return

    sample = trajectories[policies[0]]
    is_directional = ("bid_afrr_up_mw_per_hour" in sample
                       and sample["bid_afrr_up_mw_per_hour"].size > 0)

    if is_directional:
        service_rows = [
            ("FCR",     "bid_fcr_mw_per_hour",     "viridis"),
            ("aFRR up", "bid_afrr_up_mw_per_hour", "Greens"),
            ("aFRR dn", "bid_afrr_dn_mw_per_hour", "Greens"),
            ("mFRR up", "bid_mfrr_up_mw_per_hour", "Oranges"),
            ("mFRR dn", "bid_mfrr_dn_mw_per_hour", "Oranges"),
        ]
    else:
        service_rows = [
            ("FCR",  "bid_fcr_mw_per_hour",  "viridis"),
            ("aFRR", "bid_afrr_mw_per_hour", "Greens"),
            ("mFRR", "bid_mfrr_mw_per_hour", "Oranges"),
        ]

    n_rows = 1 + len(service_rows)  # PUN + services
    n_cols = len(policies)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 1.6 * n_rows + 1.0),
        sharex=True, sharey=True,
        gridspec_kw=dict(hspace=0.35, wspace=0.20),
    )
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Establish global shared bid scale per service across policies for a fair
    # visual comparison. Max bid value across policies sets the colormap top.
    max_per_service = {}
    for label, key, _ in service_rows:
        max_val = 0.0
        for p in policies:
            arr = trajectories[p].get(key)
            if arr is not None and arr.size > 0:
                max_val = max(max_val, float(np.max(arr)))
        max_per_service[key] = max(max_val, 0.1)  # avoid 0-only colormap

    # Shared PUN colormap range
    all_prices = []
    for p in policies:
        arr = trajectories[p].get("price_per_hour")
        if arr is not None and arr.size > 0:
            all_prices.extend(arr.tolist())
    pun_mean = float(np.mean(all_prices)) if all_prices else 0.0
    pun_abs_dev = float(np.max(np.abs(np.array(all_prices) - pun_mean))) if all_prices else 1.0
    pun_abs_dev = max(pun_abs_dev, 1.0)

    for col, p in enumerate(policies):
        traj = trajectories[p]
        # --- PUN row (congestion proxy) ---
        ax_pun = axes[0, col]
        prices = traj.get("price_per_hour")
        if prices is not None and prices.size > 0:
            n_hours = len(prices)
            n_days = n_hours // 24
            grid = prices[:n_days * 24].reshape(n_days, 24).T  # (24, n_days)
            im = ax_pun.imshow(
                grid, aspect="auto", origin="lower",
                cmap="RdBu_r", interpolation="nearest",
                vmin=pun_mean - pun_abs_dev, vmax=pun_mean + pun_abs_dev,
                extent=[0.5, n_days + 0.5, -0.5, 23.5],
            )
            if col == n_cols - 1:
                cbar = fig.colorbar(im, ax=ax_pun, fraction=0.035, pad=0.03)
                cbar.set_label("PUN [EUR/MWh]", fontsize=8)
                cbar.ax.tick_params(labelsize=7)
        ax_pun.set_title(f"{_POLICY_LABELS[p]}", fontsize=11, fontweight="bold")
        if col == 0:
            ax_pun.set_ylabel("PUN\n(congestion)\nHour [h]", fontsize=9)

        # --- Service rows ---
        for row_idx, (svc_label, key, cmap_name) in enumerate(service_rows, start=1):
            ax_svc = axes[row_idx, col]
            arr = traj.get(key)
            if arr is None or arr.size == 0:
                ax_svc.set_visible(False)
                continue
            n_hours = len(arr)
            n_days = n_hours // 24
            grid = arr[:n_days * 24].reshape(n_days, 24).T  # (24, n_days)
            im = ax_svc.imshow(
                grid, aspect="auto", origin="lower",
                cmap=cmap_name, interpolation="nearest",
                vmin=0.0, vmax=max_per_service[key],
                extent=[0.5, n_days + 0.5, -0.5, 23.5],
            )
            if col == n_cols - 1:
                cbar = fig.colorbar(im, ax=ax_svc, fraction=0.035, pad=0.03)
                cbar.set_label("Bid [MW]", fontsize=8)
                cbar.ax.tick_params(labelsize=7)
            if col == 0:
                ax_svc.set_ylabel(f"{svc_label}\nHour [h]", fontsize=9)
            ax_svc.set_yticks([0, 6, 12, 18, 23])

        if col == 0:
            for r in range(n_rows):
                axes[r, 0].set_yticks([0, 6, 12, 18, 23])
        # Bottom row gets xlabel
        axes[-1, col].set_xlabel("Day of test window [days]", fontsize=10)

    fig.suptitle(
        "Carpet plot: PUN congestion vs per-service bidding across policies\n"
        "(rows: PUN price + bidding intensity per service; cols: policies)",
        fontsize=12, fontweight="bold", y=0.995,
    )
    save_fn(fig, "carpet_market_activity", dpi=140, bbox_inches="tight")

    # ---- 11. Carpet plot: revenue earned per service (MILP only by default) ----
    # Helps see WHERE in (day, hour) space the MILP makes its money.
    if is_directional:
        rev_rows = [
            ("FCR",     "rev_fcr_per_hour",     "viridis"),
            ("aFRR up", "rev_afrr_up_per_hour", "Greens"),
            ("aFRR dn", "rev_afrr_dn_per_hour", "Greens"),
            ("mFRR up", "rev_mfrr_up_per_hour", "Oranges"),
            ("mFRR dn", "rev_mfrr_dn_per_hour", "Oranges"),
        ]
    else:
        rev_rows = [
            ("FCR",  "rev_fcr_per_hour",  "viridis"),
            ("aFRR", "rev_afrr_per_hour", "Greens"),
            ("mFRR", "rev_mfrr_per_hour", "Oranges"),
        ]

    # Display revenue carpets for MILP, BC, and PPO policies present
    revenue_policies = [p for p in policies if p in ("milp", "bc", "ppo_vanilla", "ppo_bc")]
    if not revenue_policies:
        return

    n_rows_r = len(rev_rows)
    n_cols_r = len(revenue_policies)

    fig, axes = plt.subplots(
        n_rows_r, n_cols_r,
        figsize=(4.5 * n_cols_r, 1.6 * n_rows_r + 1.0),
        sharex=True, sharey=True,
        gridspec_kw=dict(hspace=0.35, wspace=0.20),
    )
    if n_rows_r == 1:
        axes = np.array([axes])
    if n_cols_r == 1:
        axes = axes.reshape(-1, 1)

    # Shared revenue scale per service across these policies
    max_per_rev = {}
    for label, key, _ in rev_rows:
        max_val = 0.0
        for p in revenue_policies:
            arr = trajectories[p].get(key)
            if arr is not None and arr.size > 0:
                max_val = max(max_val, float(np.max(arr)))
        max_per_rev[key] = max(max_val, 1.0)

    for col, p in enumerate(revenue_policies):
        traj = trajectories[p]
        for row_idx, (svc_label, key, cmap_name) in enumerate(rev_rows):
            ax_svc = axes[row_idx, col]
            arr = traj.get(key)
            if arr is None or arr.size == 0:
                ax_svc.set_visible(False)
                continue
            n_hours = len(arr)
            n_days = n_hours // 24
            grid = arr[:n_days * 24].reshape(n_days, 24).T
            im = ax_svc.imshow(
                grid, aspect="auto", origin="lower",
                cmap=cmap_name, interpolation="nearest",
                vmin=0.0, vmax=max_per_rev[key],
                extent=[0.5, n_days + 0.5, -0.5, 23.5],
            )
            if col == n_cols_r - 1:
                cbar = fig.colorbar(im, ax=ax_svc, fraction=0.035, pad=0.03)
                cbar.set_label("Revenue [EUR]", fontsize=8)
                cbar.ax.tick_params(labelsize=7)
            if col == 0:
                ax_svc.set_ylabel(f"{svc_label}\nHour [h]", fontsize=9)
            ax_svc.set_yticks([0, 6, 12, 18, 23])
            if row_idx == 0:
                ax_svc.set_title(_POLICY_LABELS[p], fontsize=11, fontweight="bold")

        axes[-1, col].set_xlabel("Day of test window [days]", fontsize=10)

    fig.suptitle(
        "Carpet plot: per-service revenue across policies\n"
        "(rows: service; cols: policies; cell = revenue earned in that hour-day cell)",
        fontsize=12, fontweight="bold", y=0.995,
    )
    save_fn(fig, "carpet_revenue_by_service", dpi=140, bbox_inches="tight")


# ============================================================================
# Inference benchmark helper
# ============================================================================

def _collect_inference_timings(result) -> Optional[Dict[str, list]]:
    """Run a small benchmark of MILP vs DRL forward-pass per fleet size.

    The MILP cost is measured by solving one day with N batteries; the DRL
    cost is measured by a single BC forward pass on a (1, OBS_DIM) tensor.
    Both are repeated to amortise warm-up jitter. Returns None on failure.
    """
    try:
        import time as _time
        import numpy as _np
        import torch as _torch
        from datetime import datetime as _dt
        from milp_single import BatteryParameters as _BP
        from milp_fleet import (
            MultiBatteryParameters as _MBP,
            MultiBESSMILPOptimizer as _MILP)
        from market_data import (
            build_yearly_service_catalog as _bcat)
        from bc import BCPolicyNet as _BCN
        from marl_env import OBS_DIM as _OBS_DIM

        fleet_sizes = [3, 10, 50]
        milp_ms = []
        drl_ms = []

        # Build a tiny 24-hour catalog & price stub once
        catalog = _bcat(_dt(2024, 4, 1), _dt(2024, 4, 2))
        prices = _np.linspace(50, 150, 24).tolist()

        for N in fleet_sizes:
            bess = [_BP(capacity_mwh=4.0, max_power_mw=2.0,
                         degradation_cost_per_mwh=25000.0, cycle_life=6000)
                    for _ in range(N)]
            fleet = _MBP(bess)
            # MILP timing (median of 2 to dampen JIT spikes)
            samples = []
            for _ in range(2):
                opt = _MILP(fleet)
                t0 = _time.perf_counter()
                opt.optimize(24, prices, catalog[:24])
                samples.append((_time.perf_counter() - t0) * 1000.0)
            milp_ms.append(min(samples))

            # DRL forward-pass timing: amortised over N agents x 24 hours,
            # report per-decision ms. Batch the N obs in one forward call.
            net = _BCN()
            net.eval()
            obs = _torch.randn(N, _OBS_DIM)
            with _torch.no_grad():
                # warm-up
                for _ in range(2):
                    _ = net(obs)
                t0 = _time.perf_counter()
                for _ in range(24):
                    _ = net(obs)
                elapsed_ms = (_time.perf_counter() - t0) * 1000.0
            # per-decision = elapsed / (24 hours * N agents)
            drl_ms.append(elapsed_ms / (24 * N))

        return {"fleet_sizes": fleet_sizes,
                "milp_ms": milp_ms, "drl_ms": drl_ms}
    except Exception as exc:
        print(f"(inference benchmark skipped: {exc})")
        return None

# ============================================================================
# Additional paper-ready charts
# ============================================================================

def _make_pareto_profit_degradation(trajectories: Dict[str, Dict[str, Any]],
                                      policies: list, save_fn) -> None:
    """Scatter chart of total net profit vs total degradation cost per policy.
    Top-left quadrant = high profit with low degradation (Pareto-optimal).
    Annotate each point with policy label; plot Pareto frontier line if applicable.
    """
    if not policies:
        return
    deg = []
    prof = []
    labels = []
    colors = []
    for p in policies:
        t = trajectories[p]
        deg_total = float(t["deg_per_hour"].sum())
        net_total = float((t["arbitrage_per_hour"] + t["flex_per_hour"]
                            - t["deg_per_hour"]).sum())
        deg.append(deg_total)
        prof.append(net_total)
        labels.append(_POLICY_LABELS[p])
        colors.append(_POLICY_COLORS[p])
    deg = np.array(deg); prof = np.array(prof)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(deg, prof, s=180, c=colors, edgecolor="black", linewidth=1.2,
                zorder=3)
    for x, y, lbl in zip(deg, prof, labels):
        ax.annotate(lbl, (x, y), xytext=(8, 8), textcoords="offset points",
                     fontsize=10, fontweight="bold")

    # Pareto front (non-dominated points: lower deg, higher profit)
    order = np.argsort(deg)
    front_x, front_y = [], []
    max_prof = -np.inf
    for idx in order:
        if prof[idx] >= max_prof:
            front_x.append(deg[idx]); front_y.append(prof[idx])
            max_prof = prof[idx]
    if len(front_x) >= 2:
        ax.plot(front_x, front_y, "--", color="#2ca02c", alpha=0.6,
                 linewidth=1.5, label="Pareto front (non-dominated)", zorder=2)
        ax.legend(loc="lower right", fontsize=9)

    ax.set_xlabel("Total degradation cost [EUR]", fontsize=11)
    ax.set_ylabel("Total net profit [EUR]", fontsize=11)
    ax.set_title("Profit vs degradation trade-off across policies",
                  fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fn(fig, "pareto_profit_vs_degradation", dpi=140, bbox_inches="tight")


def _make_bc_training_curves(result, save_fn) -> None:
    """Two-panel: BC train loss (left) and val accuracy per axis (right)
    versus epoch. Useful to verify BC convergence per axis (some axes
    are slower to learn — typically discharge and aFRR_dn)."""
    history = getattr(result, "bc_history", None)
    if not history:
        return
    epochs = np.arange(1, len(history) + 1)
    train_losses = [h.get("train_loss", np.nan) for h in history]
    val_accs_per_axis = np.array([h.get("val_acc_per_axis", []) for h in history])
    if val_accs_per_axis.size == 0:
        return
    n_axes = val_accs_per_axis.shape[1]
    axis_labels = (_AXIS_LABELS_DIRECTIONAL if n_axes == 7 else _AXIS_LABELS)
    axis_colors = ["#1f77b4", "#ff7f0e", "#17becf",
                   "#2ca02c", "#98df8a", "#ff7f0e", "#ffbb78"][:n_axes]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax_l = axes[0]
    ax_l.plot(epochs, train_losses, color="#d62728", linewidth=2,
              marker="o", markersize=4)
    ax_l.set_xlabel("Epoch [-]", fontsize=11)
    ax_l.set_ylabel("Cross-entropy training loss [-]", fontsize=11)
    ax_l.set_title("BC training loss", fontsize=11, fontweight="bold")
    ax_l.grid(True, alpha=0.3, linestyle="--")
    ax_l.spines["top"].set_visible(False); ax_l.spines["right"].set_visible(False)

    ax_r = axes[1]
    for k in range(n_axes):
        ax_r.plot(epochs, val_accs_per_axis[:, k] * 100.0,
                   linewidth=1.8, marker="o", markersize=3.5,
                   color=axis_colors[k], label=axis_labels[k])
    ax_r.set_xlabel("Epoch [-]", fontsize=11)
    ax_r.set_ylabel("Validation accuracy per axis [%]", fontsize=11)
    ax_r.set_title("BC per-axis validation accuracy", fontsize=11, fontweight="bold")
    ax_r.set_ylim(0, 105)
    ax_r.grid(True, alpha=0.3, linestyle="--")
    ax_r.legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.95)
    ax_r.spines["top"].set_visible(False); ax_r.spines["right"].set_visible(False)

    fig.suptitle("Behavioral Cloning training dynamics", fontsize=12,
                  fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fn(fig, "bc_training_curves", dpi=140, bbox_inches="tight")


def _make_bc_confusion_matrix(result, save_fn) -> None:
    """Per-axis confusion matrix: MILP expert bin vs BC predicted bin.
    Diagonal = perfect imitation. Off-diagonal patterns reveal systematic
    BC errors (e.g. under-bidding, direction confusion).
    """
    milp_actions = getattr(result, "train_demo_actions", None)
    bc_preds = getattr(result, "bc_train_predictions", None)
    if milp_actions is None or bc_preds is None:
        return
    if milp_actions.shape != bc_preds.shape:
        return
    n_axes = milp_actions.shape[1]
    axis_labels = (_AXIS_LABELS_DIRECTIONAL if n_axes == 7 else _AXIS_LABELS)

    n_bins = 11
    # Lay out: 2 rows, ceil(n_axes/2) cols (for 7 axes -> 4 cols, last empty)
    ncols = (n_axes + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(3.6 * ncols, 7.2))
    axes = np.atleast_2d(axes)

    for k in range(n_axes):
        row, col = divmod(k, ncols)
        ax = axes[row, col]
        # Build confusion matrix: rows = MILP bin, cols = BC bin
        cm = np.zeros((n_bins, n_bins), dtype=np.int64)
        for m, b in zip(milp_actions[:, k], bc_preds[:, k]):
            if 0 <= m < n_bins and 0 <= b < n_bins:
                cm[int(m), int(b)] += 1
        # Normalize per row (each MILP class) for color
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / np.maximum(row_sums, 1), 0)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        # Add diagonal accuracy annotation
        diag = np.trace(cm) / max(cm.sum(), 1)
        ax.text(0.02, 0.98, f"diag acc {diag*100:.1f}%",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
        ax.set_title(axis_labels[k], fontsize=10)
        ax.set_xticks(range(0, n_bins, 2))
        ax.set_yticks(range(0, n_bins, 2))
        if col == 0:
            ax.set_ylabel("MILP bin [-]", fontsize=9)
        if row == 1 or k == n_axes - 1:
            ax.set_xlabel("BC predicted bin [-]", fontsize=9)

    # Hide unused subplots
    for k in range(n_axes, 2 * ncols):
        row, col = divmod(k, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("BC vs MILP action confusion matrices (row-normalised)",
                  fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    # Colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
    cbar.set_label("Row-normalised frequency [-]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    save_fn(fig, "bc_confusion_matrix", dpi=140, bbox_inches="tight")


def _make_price_spread_hourly(trajectories: Dict[str, Dict[str, Any]],
                                 save_fn) -> None:
    """Hour-of-day average prices: PUN vs each flex service energy price.
    Visualises which service has the highest marginal value per hour.
    Uses the MILP trajectory (any policy gives same prices: they are
    market-determined). Skip if directional prices are all zero (synthetic).
    """
    if "milp" not in trajectories:
        return
    t = trajectories["milp"]
    has_dir = ("price_afrr_up_per_hour" in t
                and t["price_afrr_up_per_hour"].size > 0
                and t["price_afrr_up_per_hour"].max() > 0)
    has_legacy = ("price_afrr_per_hour" in t
                   and t["price_afrr_per_hour"].size > 0
                   and t["price_afrr_per_hour"].max() > 0)
    if not (has_dir or has_legacy):
        return

    def by_hour(arr):
        n_h = len(arr); n_d = n_h // 24
        grid = arr[:n_d*24].reshape(n_d, 24)
        return grid.mean(axis=0)

    pun_h = by_hour(t["price_per_hour"])
    hours = np.arange(24)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(hours, pun_h, color="#1f77b4", linewidth=2.5, marker="o",
             label="PUN (day-ahead)", zorder=5)

    if has_dir:
        aup = by_hour(t["price_afrr_up_per_hour"])
        adn = by_hour(t["price_afrr_dn_per_hour"])
        mup = by_hour(t["price_mfrr_up_per_hour"])
        mdn = by_hour(t["price_mfrr_dn_per_hour"])
        ax.plot(hours, aup, color="#2ca02c", linewidth=1.8, marker="s",
                 markersize=5, label="aFRR up energy price")
        ax.plot(hours, adn, color="#98df8a", linewidth=1.8, marker="s",
                 markersize=5, label="aFRR dn energy price")
        ax.plot(hours, mup, color="#ff7f0e", linewidth=1.8, marker="^",
                 markersize=5, label="mFRR up energy price")
        ax.plot(hours, mdn, color="#ffbb78", linewidth=1.8, marker="^",
                 markersize=5, label="mFRR dn energy price")
    else:
        a = by_hour(t["price_afrr_per_hour"])
        m = by_hour(t["price_mfrr_per_hour"])
        ax.plot(hours, a, color="#2ca02c", linewidth=1.8, marker="s",
                 markersize=5, label="aFRR energy price")
        ax.plot(hours, m, color="#ff7f0e", linewidth=1.8, marker="^",
                 markersize=5, label="mFRR energy price")

    ax.set_xlabel("Hour of day [h]", fontsize=11)
    ax.set_ylabel("Energy price [EUR/MWh]", fontsize=11)
    ax.set_title("Hour-of-day price profile: PUN vs flexibility services",
                  fontsize=12, fontweight="bold")
    ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
    ax.legend(loc="best", fontsize=9, framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fn(fig, "price_profile_hourly", dpi=140, bbox_inches="tight")


def _make_daily_profit_ecdf(trajectories: Dict[str, Dict[str, Any]],
                              policies: list, save_fn) -> None:
    """ECDF of daily net profit per policy. Lower-leftish curves = poor;
    upper-rightish = good. Slope reflects variance (steep = consistent).
    """
    if not policies:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for p in policies:
        t = trajectories[p]
        n_hours = len(t["arbitrage_per_hour"])
        n_days = n_hours // 24
        net = (t["arbitrage_per_hour"] + t["flex_per_hour"] - t["deg_per_hour"])
        daily = net[:n_days*24].reshape(n_days, 24).sum(axis=1)
        if daily.size == 0:
            continue
        xs = np.sort(daily)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", color=_POLICY_COLORS[p],
                 linewidth=2.0, label=_POLICY_LABELS[p])

    ax.set_xlabel("Daily net profit [EUR]", fontsize=11)
    ax.set_ylabel("Empirical CDF (fraction of days \u2264 x) [-]", fontsize=11)
    ax.set_title("ECDF of daily net profit across policies\n"
                  "(steeper slope = more consistent revenue)",
                  fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fn(fig, "daily_profit_ecdf", dpi=140, bbox_inches="tight")


def _make_soc_distribution(trajectories: Dict[str, Dict[str, Any]],
                             policies: list, save_fn) -> None:
    """Violin plot of fleet-mean SOC distribution per policy.
    Visualises which policy tends to operate near boundaries vs centrally.
    """
    if not policies:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    data = []
    labels = []
    colors = []
    for p in policies:
        soc = trajectories[p].get("soc_mean")
        if soc is None or soc.size == 0:
            continue
        data.append(np.asarray(soc, dtype=np.float64))
        labels.append(_POLICY_LABELS[p])
        colors.append(_POLICY_COLORS[p])
    if not data:
        return
    parts = ax.violinplot(data, positions=np.arange(len(data)),
                            widths=0.8, showmeans=True, showmedians=False,
                            showextrema=True)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c); body.set_edgecolor("black"); body.set_alpha(0.7)
    for key in ("cmeans", "cmaxes", "cmins", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Fleet-mean SOC [-]", fontsize=11)
    ax.set_ylim(0, 1)
    ax.axhline(0.1, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
                label="SOC_min")
    ax.axhline(0.9, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
                label="SOC_max")
    ax.set_title("SOC operating regime per policy", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fn(fig, "soc_distribution", dpi=140, bbox_inches="tight")


def _make_service_mix_donut(trajectories: Dict[str, Dict[str, Any]],
                              policies: list, save_fn) -> None:
    """One donut chart per policy, showing the composition of FLEX revenue
    across services (FCR / aFRR_up / aFRR_dn / mFRR_up / mFRR_dn).
    Highlights each policy's market-segment specialization.
    """
    if not policies:
        return
    sample = trajectories[policies[0]]
    has_dir = ("rev_afrr_up_per_hour" in sample
                and sample["rev_afrr_up_per_hour"].size > 0)
    if has_dir:
        slices = [
            ("FCR",     "rev_fcr_per_hour",     "#17becf"),
            ("aFRR up", "rev_afrr_up_per_hour", "#2ca02c"),
            ("aFRR dn", "rev_afrr_dn_per_hour", "#98df8a"),
            ("mFRR up", "rev_mfrr_up_per_hour", "#ff7f0e"),
            ("mFRR dn", "rev_mfrr_dn_per_hour", "#ffbb78"),
        ]
    else:
        slices = [
            ("FCR",  "rev_fcr_per_hour",  "#17becf"),
            ("aFRR", "rev_afrr_per_hour", "#2ca02c"),
            ("mFRR", "rev_mfrr_per_hour", "#ff7f0e"),
        ]

    n_pol = len(policies)
    fig, axes = plt.subplots(1, n_pol, figsize=(4 * n_pol, 4.5))
    if n_pol == 1:
        axes = [axes]
    for ax, p in zip(axes, policies):
        traj = trajectories[p]
        vals = []
        labels = []
        colors = []
        for label, key, color in slices:
            arr = traj.get(key)
            v = float(arr.sum()) if arr is not None and arr.size > 0 else 0.0
            if v > 0:
                vals.append(v); labels.append(label); colors.append(color)
        if not vals:
            ax.text(0.5, 0.5, "no flex\nrevenue",
                     ha="center", va="center", transform=ax.transAxes)
            ax.set_title(_POLICY_LABELS[p], fontsize=11, fontweight="bold")
            ax.axis("off")
            continue
        total = sum(vals)
        wedges, _texts, autotxt = ax.pie(
            vals, labels=labels, colors=colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 4 else "",
            startangle=90, wedgeprops=dict(width=0.42, edgecolor="white"),
            pctdistance=0.78, textprops=dict(fontsize=9),
        )
        for t in autotxt:
            t.set_color("black"); t.set_fontsize(8); t.set_fontweight("bold")
        ax.set_title(f"{_POLICY_LABELS[p]}\nTotal flex: {total:,.0f} EUR",
                      fontsize=10, fontweight="bold")

    fig.suptitle("Flex revenue composition per policy",
                  fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fn(fig, "service_mix_donut", dpi=140, bbox_inches="tight")


def _make_hourly_bid_radar(trajectories: Dict[str, Dict[str, Any]],
                             policies: list, save_fn) -> None:
    """Radar plot: 24-spoke (one per hour-of-day) of average total bid
    intensity (% of P_max). Reveals each policy's daily 'rhythm'.
    """
    if not policies:
        return
    sample = trajectories[policies[0]]
    n_axes_act = sample["actions_mean_per_hour"].shape[1]
    # Sum bid axes (skip charge/discharge which are arbitrage, not flex)
    # Indices: 0=charge, 1=discharge, 2=FCR, 3+=flex services
    flex_axes = list(range(2, n_axes_act))

    fig, ax = plt.subplots(figsize=(8.5, 8.5),
                            subplot_kw=dict(projection="polar"))
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    for p in policies:
        t = trajectories[p]
        am = t["actions_mean_per_hour"]
        n_h = am.shape[0]; n_d = n_h // 24
        if n_d <= 0:
            continue
        # actions_mean_per_hour is already in [0, 1] (fraction of P_max)
        grid = am[:n_d*24].reshape(n_d, 24, n_axes_act)
        # Sum across flex axes, then average across days
        flex_total_per_hour = grid[:, :, flex_axes].sum(axis=-1).mean(axis=0)
        # Express as % of P_max times n_flex_axes; for 5 flex axes max = 500%
        pct = flex_total_per_hour * 100.0
        values_closed = np.concatenate([pct, [pct[0]]])
        ax.plot(angles_closed, values_closed,
                 color=_POLICY_COLORS[p], linewidth=2.0,
                 label=_POLICY_LABELS[p])
        ax.fill(angles_closed, values_closed,
                 color=_POLICY_COLORS[p], alpha=0.10)

    ax.set_xticks(angles)
    ax.set_xticklabels([f"{h}h" for h in range(24)], fontsize=8)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Daily bidding rhythm per policy\n"
                  "(total flex bid as % of P_max, averaged over test days)",
                  fontsize=11, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05),
               fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fn(fig, "hourly_bid_radar", dpi=140, bbox_inches="tight")