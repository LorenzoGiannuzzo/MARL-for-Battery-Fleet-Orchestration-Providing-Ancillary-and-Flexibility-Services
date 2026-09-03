# -*- coding: utf-8 -*-
"""
aggregation_figures.py — publication figures for the aggregation study.

Reads the JSON written by aggregation_value.py and emits every figure twice,
as PNG for drafts and slides and as PDF for the camera-ready. The PDF is
vector, so a reviewer zooming in sees clean type instead of the pixelated
equations the EI.A reviewers complained about.

The style is set for a two-column IEEE page: a single-column figure is 3.5 in
wide and a full-width one 7.16 in, and the fonts are sized so that text inside
the figure matches the body text once placed. Figures drawn at default
matplotlib size and then scaled down in LaTeX are why so many papers have
captions larger than their axis labels.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE two-column geometry.
COL_W, FULL_W = 3.5, 7.16

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
})

# Colour-blind safe, and distinguishable in greyscale print, which still
# matters for conference proceedings.
C_SOLO, C_AGG = "#4C72B0", "#DD8452"
C_CLASS = {"small_4h": "#4C72B0", "medium_2h": "#DD8452",
           "large_1h": "#55A868"}
NICE = {"small_4h": "0.5 MW / 4 h", "medium_2h": "1.0 MW / 2 h",
        "large_1h": "2.5 MW / 1 h"}


def save(fig, outdir: str, name: str):
    """Every figure as PNG and PDF, from the same draw."""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, format=ext)
        paths.append(p)
    plt.close(fig)
    print(f"  {name}.png / .pdf")
    return paths


def _by_class_shares(study: Dict[str, Any]):
    """Per class: (share of band supplied, share of revenue received)."""
    out: Dict[str, List[float]] = {}
    for u in study["units"]:
        d = out.setdefault(u["label"], [0.0, 0.0])
        d[0] += u.get("band_share", 0.0)
        d[1] += u.get("power_share", 0.0)
    return {k: tuple(v) for k, v in out.items()}


def _by_class_rules(study: Dict[str, Any], keys: List[str]):
    """Per class, the total under each allocation rule."""
    out: Dict[str, Dict[str, float]] = {}
    for u in study["units"]:
        d = out.setdefault(u["label"], {k: 0.0 for k in keys})
        for k in keys:
            d[k] += u.get(k, 0.0)
    return out


def _by_class(study: Dict[str, Any]):
    """Aggregate the per-unit results into the three portfolio classes."""
    out: Dict[str, Dict[str, float]] = {}
    for u in study["units"]:
        d = out.setdefault(u["label"], {"solo": 0.0, "agg": 0.0, "n": 0,
                                        "power": u["power_mw"]})
        d["solo"] += u["standalone_eur"]
        d["agg"] += u["aggregated_eur"]
        d["n"] += 1
    return out


# ---------------------------------------------------------------------------

def fig_gain_by_class(studies: List[Dict[str, Any]], outdir: str):
    """THE headline figure: who gains from aggregation.

    Grouped bars per class rather than a single portfolio total, because the
    portfolio total hides the finding. The story is not that aggregation adds
    value — everyone expects that — it is that the value accrues almost
    entirely to the units the threshold excludes.
    """
    fig, axes = plt.subplots(1, len(studies), figsize=(FULL_W, 2.4),
                             sharey=True, squeeze=False)
    for ax, st in zip(axes[0], studies):
        cls = _by_class(st)
        labels = [k for k in ("small_4h", "medium_2h", "large_1h")
                  if k in cls]
        x = np.arange(len(labels))
        solo = [cls[k]["solo"] / 1e3 for k in labels]
        agg = [cls[k]["agg"] / 1e3 for k in labels]

        ax.bar(x - 0.2, solo, 0.4, label="standalone", color=C_SOLO)
        ax.bar(x + 0.2, agg, 0.4, label="aggregated", color=C_AGG)
        for i, k in enumerate(labels):
            s, a = cls[k]["solo"], cls[k]["agg"]
            if abs(s) > 1e-9:
                ax.annotate(f"{100*(a/s-1):+.0f}%",
                            (i + 0.2, a / 1e3), ha="center", va="bottom",
                            fontsize=6.5, xytext=(0, 2),
                            textcoords="offset points")
        ax.set_xticks(x)
        ax.set_xticklabels([NICE.get(k, k) for k in labels])
        ax.set_title(st["year_label"])
    axes[0][0].set_ylabel("annual profit [k EUR]")
    axes[0][0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return save(fig, outdir, "fig1_gain_by_class")


def fig_revenue_composition(studies: List[Dict[str, Any]], outdir: str):
    """WHY the small units gain: they are locked out of services.

    Stacked composition makes the mechanism visible. Standalone, the sub-MW
    units earn arbitrage only, because the 1 MW threshold denies them every
    ancillary product. Aggregation is what opens that revenue stream.
    """
    fig, axes = plt.subplots(1, len(studies), figsize=(FULL_W, 2.4),
                             sharey=True, squeeze=False)
    for ax, st in zip(axes[0], studies):
        arms = ["standalone", "aggregated"]
        arb = [st[a]["arbitrage_eur"] / 1e3 for a in arms]
        flex = [st[a]["flexibility_eur"] / 1e3 for a in arms]
        deg = [-st[a]["degradation_eur"] / 1e3 for a in arms]
        x = np.arange(2)
        ax.bar(x, arb, 0.5, label="arbitrage", color=C_SOLO)
        ax.bar(x, flex, 0.5, bottom=arb, label="ancillary services",
               color=C_AGG)
        ax.bar(x, deg, 0.5, label="degradation", color="#C44E52")
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(arms)
        ax.set_title(st["year_label"])
    axes[0][0].set_ylabel("annual revenue [k EUR]")
    axes[0][0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return save(fig, outdir, "fig2_revenue_composition")


def fig_contribution_vs_payment(studies: List[Dict[str, Any]], outdir: str):
    """Share of band supplied against share of revenue received.

    This replaces an earlier scatter of gain against rated power, which used a
    symlog axis and pressed the +3%, +1% and -4% points into an
    indistinguishable smear around zero while there were only three distinct
    power values anyway — it restated figure 1 with less resolution.

    What this asks instead is whether the proportional split is FAIR: does a
    class receive the share of revenue that matches the share of reserve it
    actually provided? The diagonal is perfect correspondence. Points above it
    are over-rewarded relative to contribution, points below under-rewarded.
    """
    fig, ax = plt.subplots(figsize=(COL_W, 3.0))
    marks = {"2023": "o", "2024": "s"}
    lo, hi = 100, 0
    for st in studies:
        by = _by_class_shares(st)
        for lab, (band, power) in by.items():
            ax.scatter(100 * band, 100 * power, s=42,
                       color=C_CLASS.get(lab, "0.4"),
                       marker=marks.get(st["year_label"], "o"),
                       edgecolor="white", linewidth=0.6, zorder=3,
                       label=f"{NICE.get(lab, lab)} ({st['year_label']})")
            lo = min(lo, 100 * band, 100 * power)
            hi = max(hi, 100 * band, 100 * power)
    pad = 0.08 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, ls="--", lw=0.9, color="0.45", zorder=1)
    ax.annotate("equal share", (lim[1], lim[1]), xytext=(-4, -10),
                textcoords="offset points", ha="right", fontsize=6.5,
                color="0.45")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("share of reserve band supplied [%]")
    ax.set_ylabel("share of revenue received [%]")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    return save(fig, outdir, "fig3_contribution_vs_payment")


def fig_allocation_rules(studies: List[Dict[str, Any]], outdir: str):
    """The same optimizer result under three allocation conventions.

    The MILP maximises the JOINT profit and says nothing about who receives
    what. Splitting it is a separate decision, and it moves the headline
    number more than most modelling choices do: the sub-MW units go from
    +180% to +36% depending only on the rule. A study that reports one rule
    without saying so is implicitly picking a winner.
    """
    fig, axes = plt.subplots(1, len(studies), figsize=(FULL_W, 2.6),
                             sharey=True, squeeze=False)
    rules = [("standalone_eur", "standalone", "0.55"),
             ("aggregated_by_power_eur", "by rated power", C_SOLO),
             ("aggregated_by_band_eur", "by band supplied", C_AGG),
             ("aggregated_floor_eur", "with reserve floor", "#55A868")]
    for ax, st in zip(axes[0], studies):
        cls = _by_class_rules(st, [r[0] for r in rules])
        labels = [k for k in ("small_4h", "medium_2h", "large_1h") if k in cls]
        x = np.arange(len(labels))
        w = 0.2
        for n, (key, name, colour) in enumerate(rules):
            vals = [cls[k][key] / 1e3 for k in labels]
            ax.bar(x + (n - 1.5) * w, vals, w, label=name, color=colour)
        ax.set_xticks(x)
        ax.set_xticklabels([NICE.get(k, k) for k in labels])
        ax.set_title(st["year_label"])
    axes[0][0].set_ylabel("annual profit [k EUR]")
    axes[0][0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    return save(fig, outdir, "fig5_allocation_rules")


def fig_individual_rationality(studies: List[Dict[str, Any]], outdir: str):
    """How many units end up worse off than they were alone.

    Individual rationality is the minimum requirement in the cooperative-game
    literature: a member who earns less inside the portfolio than outside it
    simply leaves. The standalone arm is that benchmark, already computed, so
    the check is free.
    """
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    rules = [("aggregated_by_power_eur", "by rated\npower", C_SOLO),
             ("aggregated_by_band_eur", "by band\nsupplied", C_AGG),
             ("aggregated_floor_eur", "with reserve\nfloor", "#55A868")]
    x = np.arange(len(rules))
    w = 0.36
    for n, st in enumerate(studies):
        counts = []
        for key, _, _ in rules:
            counts.append(sum(1 for u in st["units"]
                              if u.get(key, 0) < u["standalone_eur"] - 1e-6))
        ax.bar(x + (n - 0.5) * w, counts, w, label=st["year_label"],
               color=[C_SOLO, C_AGG][n % 2])
        for xi, cnt in zip(x + (n - 0.5) * w, counts):
            ax.annotate(str(cnt), (xi, cnt), ha="center", va="bottom",
                        fontsize=7, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in rules])
    ax.set_ylabel("units below their\nstandalone value")
    ax.set_ylim(0, max(4, ax.get_ylim()[1] * 1.25))
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return save(fig, outdir, "fig6_individual_rationality")


def fig_service_mix(studies: List[Dict[str, Any]], outdir: str):
    """Which products the portfolio reaches, standalone versus aggregated."""
    keys: List[str] = []
    for st in studies:
        for a in ("standalone", "aggregated"):
            for k in (st[a].get("revenue_by_service") or {}):
                if k not in keys:
                    keys.append(k)
    if not keys:
        print("  (no per-service breakdown recorded, skipping fig4)")
        return []

    fig, axes = plt.subplots(1, len(studies), figsize=(FULL_W, 2.4),
                            sharey=True, squeeze=False)
    for ax, st in zip(axes[0], studies):
        x = np.arange(len(keys))
        solo = [(st["standalone"].get("revenue_by_service") or {}).get(k, 0.0)
                / 1e3 for k in keys]
        agg = [(st["aggregated"].get("revenue_by_service") or {}).get(k, 0.0)
               / 1e3 for k in keys]
        ax.bar(x - 0.2, solo, 0.4, label="standalone", color=C_SOLO)
        ax.bar(x + 0.2, agg, 0.4, label="aggregated", color=C_AGG)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=30, ha="right")
        ax.set_title(st["year_label"])
    axes[0][0].set_ylabel("annual revenue [k EUR]")
    axes[0][0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return save(fig, outdir, "fig4_service_mix")


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="aggregation_results",
                    help="directory holding aggregation_*.json")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.results, "aggregation_*.json")))
    if not files:
        print(f"no aggregation_*.json found in {args.results}")
        return 2
    studies = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            studies.append(json.load(fh))
    print(f"loaded {len(studies)} studies: "
          f"{', '.join(s['year_label'] for s in studies)}")

    print("writing figures (PNG + PDF):")
    fig_gain_by_class(studies, args.outdir)
    fig_revenue_composition(studies, args.outdir)
    fig_contribution_vs_payment(studies, args.outdir)
    fig_service_mix(studies, args.outdir)
    # These two need the per-unit allocation fields, which only runs from the
    # updated aggregation_value.py carry. Skipped with a clear message rather
    # than crashing on an older results directory.
    if any("aggregated_by_band_eur" in (st["units"][0] if st["units"] else {})
           for st in studies):
        fig_allocation_rules(studies, args.outdir)
        fig_individual_rationality(studies, args.outdir)
    else:
        print("  (no allocation fields in these results, skipping fig5/fig6 — "
              "re-run aggregation_value.py with the current version)")
    print(f"\nall figures in {args.outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())