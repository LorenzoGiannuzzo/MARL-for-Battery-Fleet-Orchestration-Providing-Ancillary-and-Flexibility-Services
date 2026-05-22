"""
Publication-grade plotting module for the BESS PPO vs MILP comparison pipeline.

Design notes:
    - All figures are saved as both .pdf (vector, for publications) and .png
      (raster preview at 300 dpi). PDFs are the primary deliverable.
    - A consistent matplotlib rcParams style is applied globally via `apply_style()`.
    - Color semantics are fixed across all figures via the COLORS dictionary,
      so MILP is always one color, PPO another, FCR/aFRR/mFRR always the same
      hue across plots. This is critical for paper readability.
    - Each figure function takes a pandas DataFrame (or dict) and a Path, and
      returns the list of files written. No global state, easy to test.
    - Figures use the single-column journal aspect ratio (~3.5" wide) when
      reasonable, double-column (~7") when extra space is needed.

The DataFrame schema expected by these functions is the one produced by
`main.run_annual_comparison()`: see `main.py` for the full column list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Style and color palette
# ---------------------------------------------------------------------------

COLORS: Dict[str, str] = {
    # Algorithms
    "MILP":         "#1f4e79",  # deep blue
    "PPO":          "#c0392b",  # crimson
    # Flexibility services
    "FCR":          "#2e7d32",  # forest green
    "aFRR":         "#f39c12",  # amber
    "mFRR":         "#8e44ad",  # purple
    # Other components
    "arbitrage":    "#5b9bd5",  # mid blue
    "degradation":  "#7f7f7f",  # neutral gray
    # Auxiliary
    "price":        "#34495e",  # slate
    "soc":          "#16a085",  # teal
    "reference":    "#bdc3c7",  # light gray (grid, ref lines)
}

# Figure widths in inches (typical journal columns)
FIG_WIDTH_SINGLE = 3.5
FIG_WIDTH_ONE_HALF = 5.0
FIG_WIDTH_DOUBLE = 7.0


def apply_style() -> None:
    """Apply a consistent publication-style matplotlib rcParams configuration."""
    plt.rcParams.update(
        {
            # Fonts (use serif for paper style; falls back if Times is unavailable)
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            # Spines: remove top and right for a cleaner look
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            # Grid
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
            # Lines and markers
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            # Ticks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            # Saving
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,   # TrueType (editable in vector editors)
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> List[Path]:
    """Save a figure as both PDF (primary) and PNG (preview)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return [pdf_path, png_path]


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def plot_cumulative_profit(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Cumulative net profit over the horizon for MILP and PPO."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, 3.0))
    ax.plot(df["day"], df["milp_cumulative"] / 1000.0,
            label="MILP", color=COLORS["MILP"], linewidth=1.5)
    ax.plot(df["day"], df["ppo_cumulative"] / 1000.0,
            label="PPO", color=COLORS["PPO"], linewidth=1.5)
    where_milp = (df["milp_cumulative"] >= df["ppo_cumulative"]).values
    where_ppo = ~where_milp
    if where_milp.any():
        ax.fill_between(df["day"],
                        df["ppo_cumulative"] / 1000.0,
                        df["milp_cumulative"] / 1000.0,
                        where=where_milp, color=COLORS["MILP"], alpha=0.08,
                        interpolate=True, label="MILP advantage")
    if where_ppo.any():
        ax.fill_between(df["day"],
                        df["ppo_cumulative"] / 1000.0,
                        df["milp_cumulative"] / 1000.0,
                        where=where_ppo, color=COLORS["PPO"], alpha=0.08,
                        interpolate=True, label="PPO advantage")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Cumulative net profit (k\u20ac)")
    ax.set_title("Cumulative net profit: MILP vs PPO")
    ax.legend(loc="upper left", ncol=2)
    ax.set_xlim(df["day"].min(), df["day"].max())
    return save_figure(fig, out_dir, "01_cumulative_profit")


def plot_daily_profit_distribution(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Side-by-side daily profit distributions: histogram + boxplot."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_DOUBLE, 3.0),
                             gridspec_kw={"width_ratios": [3, 1]})

    # Left: histogram
    ax = axes[0]
    data_milp = df["milp_net_profit"].values
    data_ppo = df["ppo_net_profit"].values
    lo = min(data_milp.min(), data_ppo.min())
    hi = max(data_milp.max(), data_ppo.max())
    bins = np.linspace(lo, hi, 35)
    ax.hist(data_milp, bins=bins, alpha=0.55, color=COLORS["MILP"],
            label="MILP", edgecolor="white", linewidth=0.3)
    ax.hist(data_ppo, bins=bins, alpha=0.55, color=COLORS["PPO"],
            label="PPO", edgecolor="white", linewidth=0.3)
    ax.axvline(np.mean(data_milp), color=COLORS["MILP"], linestyle="--", linewidth=1.0)
    ax.axvline(np.mean(data_ppo), color=COLORS["PPO"], linestyle="--", linewidth=1.0)
    ax.set_xlabel("Daily net profit (\u20ac)")
    ax.set_ylabel("Number of days")
    ax.set_title("Daily profit distribution")
    ax.legend(loc="upper right")

    # Right: boxplot
    ax = axes[1]
    bp = ax.boxplot([data_milp, data_ppo],
                    labels=["MILP", "PPO"],
                    patch_artist=True, widths=0.55,
                    medianprops={"color": "black", "linewidth": 1.0},
                    flierprops={"marker": "o", "markersize": 2,
                                "markerfacecolor": "gray",
                                "markeredgecolor": "gray", "alpha": 0.5})
    for patch, c in zip(bp["boxes"], [COLORS["MILP"], COLORS["PPO"]]):
        patch.set_facecolor(c); patch.set_alpha(0.6); patch.set_edgecolor("black")
    ax.set_ylabel("Daily net profit (\u20ac)")
    ax.set_title("Box summary")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    return save_figure(fig, out_dir, "02_daily_profit_distribution")


def plot_revenue_breakdown(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Stacked bar of annual revenue components for MILP and PPO."""
    components = {
        "Arbitrage":  (df["milp_arbitrage"].sum(),    df["ppo_arbitrage"].sum()),
        "FCR":        (df.get("milp_fcr_revenue",  pd.Series([0])).sum(),
                       df["ppo_fcr_revenue"].sum()),
        "aFRR":       (df.get("milp_afrr_revenue", pd.Series([0])).sum(),
                       df["ppo_afrr_revenue"].sum()),
        "mFRR":       (df.get("milp_mfrr_revenue", pd.Series([0])).sum(),
                       df["ppo_mfrr_revenue"].sum()),
        "Degradation": (-df["milp_degradation"].sum(), -df["ppo_degradation"].sum()),
    }
    comp_colors = {
        "Arbitrage":   COLORS["arbitrage"],
        "FCR":         COLORS["FCR"],
        "aFRR":        COLORS["aFRR"],
        "mFRR":        COLORS["mFRR"],
        "Degradation": COLORS["degradation"],
    }

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_ONE_HALF, 3.5))
    x = np.array([0, 1], dtype=float)
    width = 0.55
    pos_bottom = np.zeros(2); neg_bottom = np.zeros(2)
    for label, (vm, vp) in components.items():
        vals = np.array([vm, vp]) / 1000.0
        bottom = np.where(vals >= 0, pos_bottom, neg_bottom)
        ax.bar(x, vals, width, bottom=bottom,
               label=label, color=comp_colors[label],
               edgecolor="white", linewidth=0.5, alpha=0.9)
        pos_bottom = np.where(vals >= 0, pos_bottom + vals, pos_bottom)
        neg_bottom = np.where(vals < 0, neg_bottom + vals, neg_bottom)

    # Net profit markers
    net_milp = df["milp_cumulative"].iloc[-1] / 1000.0
    net_ppo = df["ppo_cumulative"].iloc[-1] / 1000.0
    ax.plot([-0.3, 0.3], [net_milp, net_milp], color="black", linewidth=1.8)
    ax.plot([0.7, 1.3], [net_ppo, net_ppo], color="black", linewidth=1.8)
    ax.text(0.0, net_milp, f"  net={net_milp:.1f}k",
            va="bottom", ha="left", fontsize=8)
    ax.text(1.0, net_ppo, f"  net={net_ppo:.1f}k",
            va="bottom", ha="left", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(["MILP", "PPO"])
    ax.set_ylabel("Annual revenue (k\u20ac)")
    ax.set_title("Revenue breakdown by source")
    ax.legend(loc="upper right", fontsize=7, ncol=1)
    fig.tight_layout()
    return save_figure(fig, out_dir, "03_revenue_breakdown")


def plot_profit_difference(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Daily profit difference (MILP - PPO) over time, with cumulative gap."""
    diff = df["milp_net_profit"] - df["ppo_net_profit"]
    cum_diff = diff.cumsum()

    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH_DOUBLE, 4.0),
                             sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    ax = axes[0]
    colors_bar = np.where(diff >= 0, COLORS["MILP"], COLORS["PPO"])
    ax.bar(df["day"], diff, color=colors_bar, width=1.0, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Daily (MILP - PPO) (\u20ac)")
    ax.set_title("Daily profit gap and cumulative gap")
    milp_wins = int((diff > 0).sum())
    ppo_wins  = int((diff < 0).sum())
    ax.text(0.02, 0.95,
            f"MILP wins: {milp_wins} days\nPPO wins:  {ppo_wins} days",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    ax = axes[1]
    cum_pos = (cum_diff >= 0).values
    cum_neg = ~cum_pos
    if cum_pos.any():
        ax.fill_between(df["day"], 0, cum_diff / 1000.0,
                        where=cum_pos, color=COLORS["MILP"], alpha=0.45,
                        interpolate=True, label="cum (MILP - PPO) > 0")
    if cum_neg.any():
        ax.fill_between(df["day"], 0, cum_diff / 1000.0,
                        where=cum_neg, color=COLORS["PPO"], alpha=0.45,
                        interpolate=True, label="cum (MILP - PPO) < 0")
    ax.plot(df["day"], cum_diff / 1000.0, color="black", linewidth=1.0)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Cumulative gap (k\u20ac)")
    if cum_pos.any() or cum_neg.any():
        ax.legend(loc="upper left")
    fig.tight_layout()
    return save_figure(fig, out_dir, "04_profit_difference")


def plot_execution_time(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Per-day execution time for both algorithms, log scale, plus ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_DOUBLE, 2.8),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    ax.semilogy(df["day"], df["milp_exec_time"],
                color=COLORS["MILP"], label="MILP", linewidth=0.9)
    ax.semilogy(df["day"], df["ppo_exec_time"],
                color=COLORS["PPO"],  label="PPO",  linewidth=0.9)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Execution time (s, log scale)")
    ax.set_title("Daily execution time per 24-hour horizon")
    ax.legend(loc="best")

    ax = axes[1]
    ratio = df["milp_exec_time"] / df["ppo_exec_time"].clip(lower=1e-6)
    ax.boxplot(ratio.values, vert=True, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=COLORS["MILP"], alpha=0.5),
               medianprops=dict(color="black", linewidth=1.0),
               flierprops=dict(marker="o", markersize=2, alpha=0.5))
    ax.set_yscale("log")
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xticks([1]); ax.set_xticklabels(["ratio"])
    ax.set_ylabel("MILP / PPO time ratio")
    ax.set_title(f"Median ratio: {ratio.median():.1f}x")
    fig.tight_layout()
    return save_figure(fig, out_dir, "05_execution_time")


def plot_soh_evolution(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    """State-of-health proxy over the annual horizon (cumulative throughput)."""
    # Simple linear SOH model consistent with the marginal degradation cost:
    # SOH(t) = 1 - throughput(t) / (2 * cycle_life * capacity_mwh)
    # We do not know capacity/cycle_life inside the dataframe explicitly,
    # but we can normalize by the largest throughput to plot a relative trend.
    capacity_mwh = 4.0
    cycle_life = 6000
    denom = 2.0 * cycle_life * capacity_mwh
    soh_milp = 1.0 - df["milp_throughput_cum"] / denom
    soh_ppo = 1.0 - df["ppo_throughput_cum"] / denom

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, 3.0))
    ax.plot(df["day"], soh_milp * 100, color=COLORS["MILP"],
            label=f"MILP (final SOH={soh_milp.iloc[-1]*100:.2f}%)", linewidth=1.5)
    ax.plot(df["day"], soh_ppo * 100, color=COLORS["PPO"],
            label=f"PPO (final SOH={soh_ppo.iloc[-1]*100:.2f}%)", linewidth=1.5)
    ax.axhline(80.0, color=COLORS["reference"], linestyle="--", linewidth=0.8,
               label="End-of-warranty (80%)")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("State of health (%)")
    ax.set_title("Cumulative cycle aging (linear LCOS model)")
    ax.legend(loc="lower left")
    return save_figure(fig, out_dir, "06_soh_evolution")


def plot_profit_scatter(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Scatter of daily MILP vs PPO profit with diagonal reference."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_ONE_HALF, FIG_WIDTH_ONE_HALF))
    ax.scatter(df["ppo_net_profit"], df["milp_net_profit"],
               c=df["price_volatility"], cmap="viridis",
               s=12, alpha=0.75, edgecolors="white", linewidth=0.3)
    lo = min(df["ppo_net_profit"].min(), df["milp_net_profit"].min())
    hi = max(df["ppo_net_profit"].max(), df["milp_net_profit"].max())
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.7, linestyle="--",
            label="MILP = PPO")
    ax.set_xlabel("PPO daily profit (\u20ac)")
    ax.set_ylabel("MILP daily profit (\u20ac)")
    ax.set_title("Per-day profit comparison")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=df["price_volatility"].min(),
                                                  vmax=df["price_volatility"].max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Daily price volatility (\u20ac/MWh)", fontsize=8)
    fig.tight_layout()
    return save_figure(fig, out_dir, "07_profit_scatter")


def plot_representative_day(rep_day: Optional[Dict[str, Any]],
                            out_dir: Path) -> List[Path]:
    """SOC and power trajectory on a representative 24-hour day, MILP vs PPO.

    Expects a dict with keys:
        day, prices, milp_soc, milp_power, milp_reservations (dict by service),
        ppo_soc, ppo_power, ppo_reservations (dict by service).
    If `rep_day` is None, the function returns [].
    """
    if rep_day is None:
        return []

    hours = np.arange(24)
    prices = np.asarray(rep_day["prices"])
    milp_soc = np.asarray(rep_day["milp_soc"])
    milp_power = np.asarray(rep_day["milp_power"])
    ppo_soc = np.asarray(rep_day["ppo_soc"])
    ppo_power = np.asarray(rep_day["ppo_power"])

    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH_DOUBLE, 5.5), sharex=True)

    # Top: price profile
    ax = axes[0]
    ax.plot(hours, prices, color=COLORS["price"], linewidth=1.2,
            marker="o", markersize=3, label="PUN price")
    ax.set_ylabel("Price (\u20ac/MWh)")
    ax.set_title(f"Representative day {rep_day.get('day', '?')}: "
                 f"price, SOC and power")
    ax.legend(loc="upper right")

    # Middle: SOC trajectories
    ax = axes[1]
    ax.plot(np.arange(len(milp_soc)), np.asarray(milp_soc) * 100,
            color=COLORS["MILP"], linewidth=1.5, label="MILP")
    ax.plot(np.arange(len(ppo_soc)), np.asarray(ppo_soc) * 100,
            color=COLORS["PPO"],  linewidth=1.5, label="PPO")
    ax.axhline(10.0, color=COLORS["reference"], linestyle=":", linewidth=0.7)
    ax.axhline(90.0, color=COLORS["reference"], linestyle=":", linewidth=0.7)
    ax.set_ylabel("SOC (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="best")

    # Bottom: net arbitrage power
    ax = axes[2]
    ax.bar(hours - 0.2, milp_power, width=0.4,
           color=COLORS["MILP"], label="MILP", alpha=0.8)
    ax.bar(hours + 0.2, ppo_power, width=0.4,
           color=COLORS["PPO"],  label="PPO", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Arbitrage power (MW)\n+: discharge / -: charge")
    ax.set_xlabel("Hour of day")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.legend(loc="best")
    fig.tight_layout()
    return save_figure(fig, out_dir, "08_representative_day")


def plot_cross_distribution_boxplot(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Daily-profit distribution per (algorithm, error distribution).

    The DataFrame must contain a column `error_distribution`. If the column
    is missing or has a single unique value, the figure is skipped.
    """
    if "error_distribution" not in df.columns:
        return []
    dists = sorted(df["error_distribution"].unique())
    if len(dists) < 2:
        return []

    # Build paired data: for each distribution, two box collections (MILP, PPO)
    milp_data = [df.loc[df["error_distribution"] == d, "milp_net_profit"].values
                 for d in dists]
    ppo_data  = [df.loc[df["error_distribution"] == d, "ppo_net_profit"].values
                 for d in dists]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, 3.2))
    width = 0.35
    positions = np.arange(len(dists), dtype=float)

    bp1 = ax.boxplot(milp_data, positions=positions - width / 2,
                     widths=width, patch_artist=True,
                     medianprops={"color": "black", "linewidth": 1.0},
                     flierprops={"marker": "o", "markersize": 2,
                                 "markerfacecolor": "gray",
                                 "markeredgecolor": "gray", "alpha": 0.4})
    bp2 = ax.boxplot(ppo_data, positions=positions + width / 2,
                     widths=width, patch_artist=True,
                     medianprops={"color": "black", "linewidth": 1.0},
                     flierprops={"marker": "o", "markersize": 2,
                                 "markerfacecolor": "gray",
                                 "markeredgecolor": "gray", "alpha": 0.4})
    for b in bp1["boxes"]:
        b.set_facecolor(COLORS["MILP"]); b.set_alpha(0.6); b.set_edgecolor("black")
    for b in bp2["boxes"]:
        b.set_facecolor(COLORS["PPO"]);  b.set_alpha(0.6); b.set_edgecolor("black")

    ax.set_xticks(positions)
    ax.set_xticklabels(dists, rotation=0)
    ax.set_xlabel("Forecast-error distribution")
    ax.set_ylabel("Daily net profit (\u20ac)")
    ax.set_title("Daily profit by algorithm and forecast-error distribution")
    # Custom legend (boxplot patches do not produce one)
    milp_patch = mpatches.Patch(facecolor=COLORS["MILP"], alpha=0.6,
                                edgecolor="black", label="MILP")
    ppo_patch  = mpatches.Patch(facecolor=COLORS["PPO"],  alpha=0.6,
                                edgecolor="black", label="PPO")
    ax.legend(handles=[milp_patch, ppo_patch], loc="best")
    fig.tight_layout()
    return save_figure(fig, out_dir, "09_cross_distribution_boxplot")


def plot_robustness_summary(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    """Annual cumulative profit per (algorithm, distribution), with mean and CV.

    Produces a grouped bar chart with the annual cumulative profit for MILP
    and PPO on each distribution, and annotates the coefficient of variation
    (CV = std / mean) of annual profit across distributions for each algorithm.
    Lower CV means the algorithm is more robust to forecast-error structure.
    """
    if "error_distribution" not in df.columns:
        return []
    dists = sorted(df["error_distribution"].unique())
    if len(dists) < 2:
        return []

    milp_annual = []
    ppo_annual = []
    for d in dists:
        sub = (df[df["error_distribution"] == d]
               .sort_values("day").reset_index(drop=True))
        milp_annual.append(sub["milp_cumulative"].iloc[-1])
        ppo_annual.append(sub["ppo_cumulative"].iloc[-1])
    milp_annual = np.asarray(milp_annual) / 1000.0
    ppo_annual = np.asarray(ppo_annual) / 1000.0

    def _cv(x: np.ndarray) -> float:
        m = float(np.mean(x))
        return float(np.std(x) / abs(m)) if abs(m) > 1e-9 else float("inf")

    cv_milp = _cv(milp_annual)
    cv_ppo = _cv(ppo_annual)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, 3.4))
    positions = np.arange(len(dists), dtype=float)
    width = 0.36

    ax.bar(positions - width / 2, milp_annual, width=width,
           color=COLORS["MILP"], alpha=0.85, edgecolor="white", linewidth=0.5,
           label=f"MILP (CV={cv_milp:.3f})")
    ax.bar(positions + width / 2, ppo_annual, width=width,
           color=COLORS["PPO"], alpha=0.85, edgecolor="white", linewidth=0.5,
           label=f"PPO (CV={cv_ppo:.3f})")

    # Mean reference lines
    ax.axhline(np.mean(milp_annual), color=COLORS["MILP"], linestyle="--",
               linewidth=0.7, alpha=0.7)
    ax.axhline(np.mean(ppo_annual), color=COLORS["PPO"], linestyle="--",
               linewidth=0.7, alpha=0.7)

    # Value labels on bars
    for x, v in zip(positions - width / 2, milp_annual):
        ax.text(x, v, f"{v:.1f}k", ha="center", va="bottom", fontsize=7)
    for x, v in zip(positions + width / 2, ppo_annual):
        ax.text(x, v, f"{v:.1f}k", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(positions)
    ax.set_xticklabels(dists)
    ax.set_xlabel("Forecast-error distribution")
    ax.set_ylabel("Annual cumulative net profit (k\u20ac)")
    ax.set_title("Robustness of annual profit across forecast-error distributions")
    ax.legend(loc="best")
    fig.tight_layout()
    return save_figure(fig, out_dir, "10_robustness_summary")


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def generate_all_plots(df: pd.DataFrame,
                       out_dir: Path,
                       representative_day: Optional[Dict[str, Any]] = None,
                       primary_distribution: Optional[str] = None,
                       verbose: bool = True) -> Dict[str, List[Path]]:
    """Generate every figure used by the comparison pipeline.

    Args:
        df: long-format DataFrame produced by `main.run_annual_comparison()`.
            May contain an `error_distribution` column with multiple values.
        out_dir: directory in which to write PDFs and PNGs.
        representative_day: optional dict with 24-hour snapshots for plot 08.
        primary_distribution: name of the distribution from which to draw the
            single-distribution plots (01 to 08). Defaults to the first
            distribution found in `df` if not provided.
        verbose: whether to print a summary line at the end.

    Returns:
        Mapping figure-name -> list of file paths.
    """
    apply_style()
    out_dir = Path(out_dir)

    # Choose the primary distribution for single-distribution plots
    has_dist_col = "error_distribution" in df.columns
    if has_dist_col:
        if primary_distribution is None or primary_distribution not in df["error_distribution"].unique():
            primary_distribution = sorted(df["error_distribution"].unique())[0]
        df_primary = (df[df["error_distribution"] == primary_distribution]
                      .sort_values("day").reset_index(drop=True))
    else:
        df_primary = df.copy()
        primary_distribution = None

    figures = {
        "cumulative_profit":           plot_cumulative_profit(df_primary, out_dir),
        "daily_profit_distribution":   plot_daily_profit_distribution(df_primary, out_dir),
        "revenue_breakdown":           plot_revenue_breakdown(df_primary, out_dir),
        "profit_difference":           plot_profit_difference(df_primary, out_dir),
        "execution_time":              plot_execution_time(df_primary, out_dir),
        "soh_evolution":               plot_soh_evolution(df_primary, out_dir),
        "profit_scatter":              plot_profit_scatter(df_primary, out_dir),
        "representative_day":          plot_representative_day(representative_day, out_dir),
        # Cross-distribution figures (skipped automatically if only one dist)
        "cross_distribution_boxplot":  plot_cross_distribution_boxplot(df, out_dir),
        "robustness_summary":          plot_robustness_summary(df, out_dir),
    }
    if verbose:
        n_files = sum(len(v) for v in figures.values())
        if primary_distribution is not None:
            print(f"  Saved {n_files} files under {out_dir} "
                  f"(single-dist plots from '{primary_distribution}')")
        else:
            print(f"  Saved {n_files} files under {out_dir}")
    return figures