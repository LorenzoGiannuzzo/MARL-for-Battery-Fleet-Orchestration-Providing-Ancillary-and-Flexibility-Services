# Figure index

This folder contains all the figures produced by the comparison pipeline. Each figure is saved both as `.pdf` (vector, for the paper) and `.png` (raster preview at 300 dpi).

- Train distribution (PPO pretraining + retraining): **ornstein-uhlenbeck**
- Eval distributions (MILP and trained PPO benchmarked on each): **ornstein-uhlenbeck**
- Primary distribution for the representative-day plot: **ornstein-uhlenbeck**

## Organization

```
figures/
├── INDEX.md                              (this file)
├── per_distribution/                     (single-distribution analyses)
│   ├── ornstein-uhlenbeck/
│   │   ├── 01_cumulative_profit.{pdf,png}
│   │   ├── 02_daily_profit_distribution.{pdf,png}
│   │   ├── 03_revenue_breakdown.{pdf,png}
│   │   ├── 04_profit_difference.{pdf,png}
│   │   ├── 05_execution_time.{pdf,png}
│   │   ├── 06_soh_evolution.{pdf,png}
│   │   ├── 07_profit_scatter.{pdf,png}
│   │   └── 08_representative_day.{pdf,png}  (only for the primary distribution)
└── cross_distribution/                   (robustness analyses)
    ├── 09_cross_distribution_boxplot.{pdf,png}
    └── 10_robustness_summary.{pdf,png}
```

## What each figure shows

**Single-distribution plots** (under `per_distribution/<name>/`) describe the behavior of MILP and PPO on a single evaluation distribution. They are useful for the *deep-dive* section of the paper, where one specific scenario is described in detail.

- `01_cumulative_profit`: annual profit trajectory, with shaded regions for the algorithm with the lead. The classic abstract figure.
- `02_daily_profit_distribution`: histogram + box of daily net profit. Shows skewness, modality, and outliers.
- `03_revenue_breakdown`: stacked bar of arbitrage, FCR, aFRR, mFRR, degradation, side-by-side for MILP and PPO. Net profit is annotated explicitly.
- `04_profit_difference`: daily (MILP - PPO) bar with cumulative gap below. Useful for identifying *which days* the gap accumulates.
- `05_execution_time`: MILP vs PPO execution time per day on log scale, plus the time-ratio boxplot.
- `06_soh_evolution`: state-of-health proxy for both algorithms based on cumulative throughput.
- `07_profit_scatter`: per-day MILP vs PPO scatter with diagonal reference, color-coded by daily price volatility.
- `08_representative_day`: 24-hour snapshot of PUN price, SOC trajectories, and arbitrage power, on the median-volatility day. The *qualitative* result. Available only for the primary distribution to avoid clutter.

**Cross-distribution plots** (under `cross_distribution/`) describe the robustness of the two algorithms across the four forecast-error distributions. They are the key figures for the *robustness* section of the paper.

- `09_cross_distribution_boxplot`: grouped boxplot of daily profit for (algorithm, distribution).
- `10_robustness_summary`: grouped bar of annual cumulative profit, with the coefficient of variation (CV) of each algorithm annotated in the legend. **This is the headline figure for the robustness section.** Lower CV = more robust to forecast-error structure.

## Suggested mapping to paper figures

- Figure 1 (abstract / introduction teaser): `per_distribution/<primary>/01_cumulative_profit` + `cross_distribution/10_robustness_summary` as a two-panel.
- Figure 2 (qualitative): `per_distribution/<primary>/08_representative_day`.
- Figure 3 (economic detail): `per_distribution/<primary>/03_revenue_breakdown`.
- Figure 4 (robustness): `cross_distribution/09_cross_distribution_boxplot`.
- Figure 5 (computational comparison): `per_distribution/<primary>/05_execution_time`.
- Supplementary: everything else.