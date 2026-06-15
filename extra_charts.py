# -*- coding: utf-8 -*-
"""
extra_charts.py — figure aggiuntive per il paper, generate dopo il production run.

Cinque famiglie di figure, tutte salvate in PNG (300 dpi) e PDF (vettoriale)
sotto results/<run_id>/png/ e results/<run_id>/pdf/.

  1. Monthly stacked revenue per policy
     4 sub-pannelli (uno per policy), barre stacked per mese:
     arbitrage / FCR / aFRR_dn / mFRR (e altri se presenti).
     Replica lo stile Mortimer/Sandelic ma con le tue policy a confronto.

  2. 2D histogram activation duration vs intensity, per servizio
     Per ogni servizio direzionale + FCR, scatter di "eventi" di attivazione
     (sequenze consecutive di bid > soglia) sull'asse durata-intensita,
     con colore = frequenza. Stile Sandelic 2022.

  3. SOC annual carpet per policy
     Heatmap 24h x 366gg con SOC come colore, una figura con 5 sub-pannelli
     verticali (MILP / BC / PPO_warm / PPO_vanilla / Random).

  4. Net revenue annual carpet per policy
     Heatmap 24h x 366gg con NET revenue oraria come colore.

  5. Scatter 2D profit-vs-degradation per policy, con throughput come colore
     Ogni punto = un giorno del test. Sostituisce il 3D che era stato
     proposto (e che renderebbe peggio della versione 2D con encoding).

UTILIZZO
--------
Le trajectories sono salvate da pipeline_step8 quando si usa
`record_trajectory=True` (gia' attivo). Ma il run di Lorenzo NON salva
trajectories a disco di default; sono solo in memoria nel `result` object.

Per generare queste figure SENZA rilanciare il pipeline, due opzioni:
 (a) modificare run_paper_experiment.py per aggiungere un torch.save delle
     trajectories (vedi commento in fondo); poi questo script le carica;
 (b) ri-aggangiarsi al `result` direttamente importando le funzioni e
     passandogli `result.trajectories`. Per questo lo script espone una
     funzione `make_extra_charts(trajectories, output_dir)`.

Per (a):
  python extra_charts.py results/<run_id>

Per (b): chiamato come callback alla fine di run_paper_experiment.py.
"""

from __future__ import annotations

import os
import sys
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ----------------------------------------------------------------------------
# Palette coerente con il resto dei chart del paper
# ----------------------------------------------------------------------------
COLORS = {
    "milp":        "#2E86C1",
    "bc":          "#27AE60",
    "ppo_bc":      "#C0392B",
    "ppo_vanilla": "#E67E22",
    "random":      "#7F8C8D",
}
SERVICE_COLORS = {
    "arbitrage": "#3498DB",
    "FCR":       "#1ABC9C",
    "aFRR_up":   "#229954",
    "aFRR_dn":   "#7DCEA0",
    "mFRR_up":   "#E67E22",
    "mFRR_dn":   "#F5B041",
}
POLICY_LABELS = {
    "milp":        "MILP",
    "bc":          "BC policy",
    "ppo_bc":      "PPO (BC warmstart)",
    "ppo_vanilla": "PPO vanilla",
    "random":      "Random baseline",
}
GRID_KW = dict(alpha=0.18, linewidth=0.6)


def _save(fig, out_dir: Path, name: str, dpi_png: int = 300, dpi_pdf: int = 500):
    png_dir = out_dir / "png"
    pdf_dir = out_dir / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / f"{name}.png", dpi=dpi_png, bbox_inches="tight")
    fig.savefig(pdf_dir / f"{name}.pdf", dpi=dpi_pdf, bbox_inches="tight",
                format="pdf")
    plt.close(fig)
    print(f"  saved: {name}.png + {name}.pdf")


# ============================================================================
# Helpers to extract per-policy hourly arrays from the trajectory dict
# ============================================================================

def _safe(traj: Dict[str, Any], key: str, default_len: int) -> np.ndarray:
    """Get hourly array from trajectory, or zeros if missing."""
    arr = traj.get(key)
    if arr is None or len(arr) == 0:
        return np.zeros(default_len, dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)


def _hours_to_calendar(n_hours: int, start_date: datetime) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays of (day_index 0..N_days-1, hour_of_day 0..23, month 1..12)
    for each hour in the test window."""
    hours = np.arange(n_hours)
    day_idx = hours // 24
    hour_of_day = hours % 24
    months = np.array([(start_date + timedelta(hours=int(h))).month
                       for h in hours])
    return day_idx, hour_of_day, months


# ============================================================================
# Figure 1: Monthly stacked revenue per policy
# ============================================================================

def plot_monthly_revenue_stacked(
    trajectories: Dict[str, Dict[str, Any]],
    out_dir: Path,
    start_date: datetime,
    name: str = "monthly_revenue_stacked",
):
    """4-panel figure (one per policy) with bars stacked per month showing
    arbitrage + per-service revenue contributions. Mortimer/Sandelic style.
    Random baseline is omitted (cluttered and not meaningful).
    """
    policies = [k for k in ("milp", "bc", "ppo_bc", "ppo_vanilla")
                if k in trajectories]
    if not policies:
        print("  [monthly_revenue] no policies found, skipping")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes = axes.ravel()

    for ax, pol in zip(axes, policies):
        traj = trajectories[pol]
        n_hours = len(traj.get("arbitrage_per_hour", []))
        if n_hours == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes); continue
        _, _, months = _hours_to_calendar(n_hours, start_date)

        arb = _safe(traj, "arbitrage_per_hour", n_hours)
        rev_fcr = _safe(traj, "rev_fcr_per_hour", n_hours)
        rev_aup = _safe(traj, "rev_afrr_up_per_hour", n_hours)
        rev_adn = _safe(traj, "rev_afrr_dn_per_hour", n_hours)
        rev_mup = _safe(traj, "rev_mfrr_up_per_hour", n_hours)
        rev_mdn = _safe(traj, "rev_mfrr_dn_per_hour", n_hours)

        # Aggregate per calendar month
        monthly = {m: dict(arbitrage=0., FCR=0., aFRR_up=0., aFRR_dn=0.,
                            mFRR_up=0., mFRR_dn=0.)
                   for m in range(1, 13)}
        for h in range(n_hours):
            m = int(months[h])
            monthly[m]["arbitrage"] += float(arb[h])
            monthly[m]["FCR"]       += float(rev_fcr[h])
            monthly[m]["aFRR_up"]   += float(rev_aup[h])
            monthly[m]["aFRR_dn"]   += float(rev_adn[h])
            monthly[m]["mFRR_up"]   += float(rev_mup[h])
            monthly[m]["mFRR_dn"]   += float(rev_mdn[h])

        months_x = np.arange(1, 13)
        # convert EUR -> kEUR for readability
        bottoms = np.zeros(12)
        components = ["arbitrage", "FCR", "aFRR_dn", "aFRR_up",
                      "mFRR_dn", "mFRR_up"]
        for comp in components:
            vals = np.array([monthly[m][comp] / 1e3 for m in months_x])
            if np.all(np.abs(vals) < 1.0):  # skip empty series for clarity
                continue
            ax.bar(months_x, vals, bottom=bottoms,
                   color=SERVICE_COLORS[comp], edgecolor="white",
                   linewidth=0.4, label=comp.replace("_", " "))
            bottoms = bottoms + vals

        # Total line for context
        totals = bottoms.copy()
        ax.plot(months_x, totals, color="#333333", lw=0.0, marker="d",
                ms=4, label="total")

        ax.set_title(POLICY_LABELS[pol], fontsize=12, fontweight="bold",
                     color=COLORS[pol])
        ax.set_xticks(months_x)
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                            rotation=0, fontsize=9)
        ax.set_xlabel("Month of year [-]", fontsize=10)
        ax.set_ylabel("Monthly revenue [kEUR]", fontsize=10)
        ax.grid(True, axis="y", **GRID_KW)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                  ncol=4, frameon=False)

    fig.suptitle("Monthly revenue composition by policy",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, name)


# ============================================================================
# Figure 2: 2D histogram activation duration vs intensity, per service
# ============================================================================

def _extract_events(bid_series: np.ndarray, p_max_fleet: float,
                    threshold_frac: float = 0.05) -> List[Tuple[float, float]]:
    """Extract activation events as (duration_h, mean_intensity_frac) tuples.

    An 'event' is a maximal run of consecutive hours where bid > threshold.
    Intensity is the mean bid over the event normalised by p_max_fleet.
    """
    over = bid_series > (threshold_frac * p_max_fleet)
    events = []
    i = 0
    n = len(over)
    while i < n:
        if not over[i]:
            i += 1; continue
        j = i
        while j < n and over[j]:
            j += 1
        duration = float(j - i)
        intensity = float(bid_series[i:j].mean() / max(p_max_fleet, 1e-9))
        events.append((duration, intensity))
        i = j
    return events


def plot_activation_duration_vs_intensity(
    trajectories: Dict[str, Dict[str, Any]],
    out_dir: Path,
    p_max_fleet: float,
    name: str = "activation_duration_vs_intensity",
):
    """2D histograms of activation events for each policy x service.
    Mimics Sandelic 2022 Fig.3 but conditioned on policy."""
    policies = [k for k in ("milp", "bc", "ppo_bc")
                if k in trajectories]
    if not policies:
        print("  [activation_2d] no policies found, skipping")
        return

    services = [
        ("bid_fcr_mw_per_hour",     "FCR"),
        ("bid_afrr_dn_mw_per_hour", "aFRR dn"),
        ("bid_afrr_up_mw_per_hour", "aFRR up"),
    ]

    fig, axes = plt.subplots(len(policies), len(services),
                             figsize=(4.2 * len(services), 3.4 * len(policies)),
                             sharex=True, sharey=True)
    if len(policies) == 1:
        axes = axes.reshape(1, -1)

    duration_max = 24.0  # truncate very long events for readability
    for r, pol in enumerate(policies):
        traj = trajectories[pol]
        for c, (bid_key, svc_label) in enumerate(services):
            ax = axes[r, c]
            bids = _safe(traj, bid_key, 0)
            if len(bids) == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes); continue
            events = _extract_events(bids, p_max_fleet)
            if not events:
                ax.text(0.5, 0.5, "no activations", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#888")
                ax.set_xlim(0, duration_max); ax.set_ylim(0, 1)
            else:
                ds = np.array([min(d, duration_max) for d, _ in events])
                its = np.array([i for _, i in events])
                h, xe, ye = np.histogram2d(ds, its, bins=[24, 20],
                                            range=[[0, duration_max], [0, 1]])
                im = ax.imshow(h.T, origin="lower", aspect="auto",
                                extent=[0, duration_max, 0, 1],
                                cmap="viridis",
                                norm=LogNorm(vmin=1, vmax=max(2, h.max())))
                ax.text(0.97, 0.95, f"n={len(events)}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8, color="white",
                        bbox=dict(facecolor="#222", alpha=0.6, edgecolor="none",
                                  pad=2))
            if r == 0:
                ax.set_title(svc_label, fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{POLICY_LABELS[pol]}\nIntensity [frac of P_max]",
                              fontsize=9, color=COLORS[pol], fontweight="bold")
            if r == len(policies) - 1:
                ax.set_xlabel("Activation duration [h]", fontsize=10)
            ax.grid(True, **GRID_KW)
            ax.tick_params(labelsize=8)

    fig.suptitle("Activation event characterisation: duration vs intensity",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, name)


# ============================================================================
# Figure 3 & 4: Annual carpet plots (SOC, NET revenue)
# ============================================================================

def _carpet_array(series: np.ndarray, start_date: datetime) -> Tuple[np.ndarray, int]:
    """Reshape a length-N hourly series into a (24, N_days) array for imshow.
    Pads with NaN if N is not a multiple of 24."""
    n = len(series)
    n_days = (n + 23) // 24
    pad = n_days * 24 - n
    if pad > 0:
        series = np.concatenate([series, np.full(pad, np.nan)])
    return series.reshape(n_days, 24).T, n_days  # (24, N_days)


def plot_annual_carpets(
    trajectories: Dict[str, Dict[str, Any]],
    out_dir: Path,
    start_date: datetime,
    metric: str,  # 'soc' or 'net'
    name: Optional[str] = None,
):
    """5-panel vertical carpet plot, one per policy.
       metric='soc' -> use 'soc_mean' (range 0..1)
       metric='net' -> use arb+flex-deg hourly NET (EUR)
    """
    policies = [k for k in ("milp", "bc", "ppo_bc", "ppo_vanilla", "random")
                if k in trajectories]
    if not policies:
        return

    fig, axes = plt.subplots(len(policies), 1,
                             figsize=(14, 1.8 * len(policies)),
                             sharex=True)
    if len(policies) == 1:
        axes = [axes]

    # Compute global value range for shared color scale
    all_vals = []
    for pol in policies:
        traj = trajectories[pol]
        if metric == "soc":
            s = _safe(traj, "soc_mean", 0)
        elif metric == "net":
            arb = _safe(traj, "arbitrage_per_hour", 0)
            flex = _safe(traj, "flex_per_hour", 0)
            deg = _safe(traj, "deg_per_hour", 0)
            if len(arb) == 0:
                s = np.array([])
            else:
                s = arb + flex - deg
        if len(s):
            all_vals.append(s)
    if not all_vals:
        plt.close(fig); return
    all_concat = np.concatenate(all_vals)
    if metric == "soc":
        vmin, vmax = 0.0, 1.0
        cmap = "viridis"
        cbar_label = "SOC [-]"
    else:
        # Use percentile clipping so outliers don't dominate the colormap
        vmin = float(np.nanpercentile(all_concat, 2))
        vmax = float(np.nanpercentile(all_concat, 98))
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
        cmap = "RdBu_r"
        cbar_label = "Hourly NET revenue [EUR]"

    last_im = None
    for ax, pol in zip(axes, policies):
        traj = trajectories[pol]
        if metric == "soc":
            s = _safe(traj, "soc_mean", 0)
        else:
            arb = _safe(traj, "arbitrage_per_hour", 0)
            flex = _safe(traj, "flex_per_hour", 0)
            deg = _safe(traj, "deg_per_hour", 0)
            if len(arb) == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes); continue
            s = arb + flex - deg
        if len(s) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes); continue
        grid, n_days = _carpet_array(s, start_date)
        im = ax.imshow(grid, origin="lower", aspect="auto",
                       extent=[0, n_days, 0, 24],
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        last_im = im
        ax.set_ylabel(f"{POLICY_LABELS[pol]}\nHour [h]",
                      fontsize=9, color=COLORS[pol], fontweight="bold")
        ax.set_yticks([0, 6, 12, 18, 23])
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Day of test window [days]", fontsize=10)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, orientation="vertical",
                            fraction=0.018, pad=0.015)
        cbar.set_label(cbar_label, fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    title = ("Fleet-mean SOC over the test year" if metric == "soc"
             else "Hourly NET revenue carpet")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    if name is None:
        name = f"annual_carpet_{metric}"
    _save(fig, out_dir, name)


# ============================================================================
# Figure 5: 2D scatter profit vs degradation, throughput as colour
# ============================================================================

def plot_profit_vs_degradation_with_throughput(
    trajectories: Dict[str, Dict[str, Any]],
    out_dir: Path,
    name: str = "profit_vs_deg_throughput_2d",
):
    """For each policy, scatter daily revenue against daily degradation cost;
    encode throughput as marker size (and indirectly via density). Replaces
    the proposed 3D plot with a more readable 2D representation."""
    policies = [k for k in ("milp", "bc", "ppo_bc", "ppo_vanilla", "random")
                if k in trajectories]
    if not policies:
        return

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for pol in policies:
        traj = trajectories[pol]
        arb = _safe(traj, "arbitrage_per_hour", 0)
        flex = _safe(traj, "flex_per_hour", 0)
        deg = _safe(traj, "deg_per_hour", 0)
        bid_dis_proxy = (
            _safe(traj, "bid_afrr_dn_mw_per_hour", len(arb))
            + _safe(traj, "bid_afrr_up_mw_per_hour", len(arb))
            + _safe(traj, "bid_fcr_mw_per_hour", len(arb))
        )
        if len(arb) == 0:
            continue

        # Aggregate per day
        n_days = (len(arb) + 23) // 24
        daily_revenue = []
        daily_deg = []
        daily_throughput = []
        for d in range(n_days):
            s, e = d * 24, min((d + 1) * 24, len(arb))
            daily_revenue.append(float((arb[s:e] + flex[s:e]).sum()))
            daily_deg.append(float(deg[s:e].sum()))
            daily_throughput.append(float(bid_dis_proxy[s:e].sum()))

        daily_revenue = np.array(daily_revenue)
        daily_deg = np.array(daily_deg)
        daily_throughput = np.array(daily_throughput)
        # marker size proportional to throughput, capped for readability
        size_norm = 60 * (daily_throughput - daily_throughput.min()) / max(
            (daily_throughput.max() - daily_throughput.min()), 1e-9) + 8
        ax.scatter(daily_deg, daily_revenue, s=size_norm,
                   color=COLORS[pol], alpha=0.45, edgecolor="white",
                   linewidth=0.3, label=POLICY_LABELS[pol])

    ax.set_xlabel("Daily degradation cost [EUR/day]", fontsize=11)
    ax.set_ylabel("Daily revenue (arbitrage + flex) [EUR/day]", fontsize=11)
    ax.set_title("Daily profit vs degradation, marker size = activation throughput",
                 fontsize=12, fontweight="bold")
    ax.grid(True, **GRID_KW)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    _save(fig, out_dir, name)


# ============================================================================
# Public entry points
# ============================================================================

def make_extra_charts(
    trajectories: Dict[str, Dict[str, Any]],
    out_dir,
    start_date: datetime,
    p_max_fleet: float,
):
    """Generate all 5 extra charts in one call. Use this from
    run_paper_experiment.py after run_full_pipeline() returns."""
    out_dir = Path(out_dir)
    print(f"\n[extra_charts] generating into {out_dir}/png + /pdf")
    plot_monthly_revenue_stacked(trajectories, out_dir, start_date)
    plot_activation_duration_vs_intensity(trajectories, out_dir, p_max_fleet)
    plot_annual_carpets(trajectories, out_dir, start_date, metric="soc")
    plot_annual_carpets(trajectories, out_dir, start_date, metric="net")
    plot_profit_vs_degradation_with_throughput(trajectories, out_dir)
    print(f"[extra_charts] done.")


def main_cli():
    """Standalone CLI: load trajectories from a pickle and generate charts.

    Expects results/<run_id>/trajectories.pkl (need to dump it from
    run_paper_experiment.py — see the README block at the top of this file).
    """
    if len(sys.argv) < 2:
        print("usage: python extra_charts.py results/<run_id>")
        print("       (requires trajectories.pkl in that folder)")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    pkl = run_dir / "trajectories.pkl"
    if not pkl.exists():
        print(f"ERROR: {pkl} not found. Add this snippet to run_paper_experiment.py")
        print("right after run_full_pipeline() returns:")
        print()
        print("    import pickle")
        print("    if result.trajectories is not None:")
        print("        with open(out_dir / 'trajectories.pkl', 'wb') as f:")
        print("            pickle.dump({")
        print("                'trajectories': result.trajectories,")
        print("                'start_date': test_start,")
        print("                'p_max_fleet': sum(b.max_power_mw for b in fleet.batteries),")
        print("            }, f)")
        sys.exit(1)
    with open(pkl, "rb") as f:
        bundle = pickle.load(f)
    make_extra_charts(
        bundle["trajectories"], run_dir,
        bundle["start_date"], bundle["p_max_fleet"],
    )


if __name__ == "__main__":
    main_cli()
