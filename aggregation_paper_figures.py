# -*- coding: utf-8 -*-
"""
aggregation_paper_figures.py: the two figures of the PES IM manuscript, drawn from the
JSON files written by aggregation_value.py.

Fig. 1: hours of participation per product, averaged within each unit class, aggregated
case as filled bars and standalone case as open outlines.
Fig. 2: annual profit of each unit class under the three allocation conventions, next to
the standalone benchmark.

The figure sizes match the frames of the manuscript exactly (7.206 x 1.767 in and
6.897 x 2.383 in), so that the PNG can replace the embedded image without distortion.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.formatter.use_locale": False,
})

CLASSES = ("small_4h", "medium_2h", "large_1h")
NICE = {"small_4h": "0.5 MW / 4 h", "medium_2h": "1.0 MW / 2 h", "large_1h": "2.5 MW / 1 h"}
C_CLASS = {"small_4h": "#4C72B0", "medium_2h": "#DD8452", "large_1h": "#55A868"}
PRODUCTS = (("FCR", "FCR"), ("aFRR_up", "Upward aFRR"), ("aFRR_dn", "Downward aFRR"))

#Lorenzo Giannuzzo: frame sizes of the two images in the manuscript, in inches
FIG1_SIZE = (6588960 / 914400, 1615440 / 914400)
FIG2_SIZE = (6306636 / 914400, 2179320 / 914400)


def boxed_legend(fig, handles, labels, ncol):
    """Legend centred at the top of the figure, in a box with a black border."""
    leg = fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=True, fancybox=False,
                     edgecolor="black", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2)
    leg.get_frame().set_linewidth(0.6)
    return leg


def year_tag(ax, label):
    ax.text(0.01, 0.97, label, transform=ax.transAxes, ha="left", va="top", fontweight="bold")


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=400)
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def class_index(study):
    return {c: [i for i, u in enumerate(study["units"]) if u["label"] == c] for c in CLASSES}


def fig_participation_hours(studies, outdir):
    fig, axes = plt.subplots(1, len(studies), figsize=FIG1_SIZE, sharey=True, squeeze=False)
    width = 0.26
    for ax, st in zip(axes[0], studies):
        idx = class_index(st)
        x = np.arange(len(PRODUCTS))
        for n, c in enumerate(CLASSES):
            agg = [np.mean([st["detail"]["aggregated"]["hours_by_service"][k][i] for i in idx[c]]) for k, _ in PRODUCTS]
            solo = [np.mean([st["detail"]["standalone"]["hours_by_service"][k][i] for i in idx[c]]) for k, _ in PRODUCTS]
            pos = x + (n - 1) * width
            ax.bar(pos, agg, width, color=C_CLASS[c])
            #Lorenzo Giannuzzo: standalone hours as open outlines; a zero-height outline is not drawn
            ax.bar(pos, solo, width, facecolor="none", edgecolor="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([name for _, name in PRODUCTS])
        year_tag(ax, st["year_label"])
    axes[0][0].set_ylabel("Hours Per Year [h]")
    handles = [Patch(color=C_CLASS[c]) for c in CLASSES] + [Patch(facecolor="none", edgecolor="black", linewidth=0.7)]
    labels = [NICE[c] for c in CLASSES] + ["Standalone"]
    boxed_legend(fig, handles, labels, ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.84), pad=0.3, w_pad=0.8)
    save(fig, outdir, "paper_fig1_participation_hours")


def fig_allocation_rules(studies, outdir):
    rules = (("standalone_eur", "Standalone", "0.55"),
             ("aggregated_by_power_eur", "By Rated Power", "#4C72B0"),
             ("aggregated_by_band_eur", "By Band Supplied", "#DD8452"),
             ("aggregated_floor_eur", "With Reserve Floor", "#55A868"))
    fig, axes = plt.subplots(1, len(studies), figsize=FIG2_SIZE, sharey=True, squeeze=False)
    width = 0.2
    for ax, st in zip(axes[0], studies):
        idx = class_index(st)
        x = np.arange(len(CLASSES))
        for n, (key, _, colour) in enumerate(rules):
            vals = [sum(st["units"][i][key] for i in idx[c]) / 1e3 for c in CLASSES]
            ax.bar(x + (n - 1.5) * width, vals, width, color=colour)
        ax.set_xticks(x)
        ax.set_xticklabels([NICE[c] for c in CLASSES])
        year_tag(ax, st["year_label"])
    axes[0][0].set_ylabel("Annual Profit [k€]")
    handles = [Patch(color=colour) for _, _, colour in rules]
    boxed_legend(fig, handles, [name for _, name, _ in rules], ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=0.3, w_pad=0.8)
    save(fig, outdir, "paper_fig2_allocation_rules")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="aggregation_v3", help="directory holding aggregation_*.json")
    ap.add_argument("--outdir", default="figures_paper")
    args = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(args.results, "aggregation_*.json")))
    if not files:
        print(f"no aggregation_*.json found in {args.results}")
        return 2
    studies = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            studies.append(json.load(fh))
    print(f"loaded {len(studies)} studies: {', '.join(s['year_label'] for s in studies)}")
    fig_participation_hours(studies, args.outdir)
    fig_allocation_rules(studies, args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
