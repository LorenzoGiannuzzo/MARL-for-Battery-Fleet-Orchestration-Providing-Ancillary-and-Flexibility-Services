"""
Diagnostic plotting for forecast-error calibration.

Produces publication-grade figures that summarise the empirical residual
distribution versus parametric fits, for use in the paper's methodology
section.
"""

from __future__ import annotations

from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from forecast_error_calibration import RegimeCalibration


# Visual style: keep close to plotting.py conventions in the repo.
_STYLE = {
    'hist': dict(bins=80, color='#cccccc', edgecolor='#666666',
                 linewidth=0.3, density=True, alpha=0.85),
    'normal': dict(color='#1f77b4', linewidth=1.8, label='Normal fit'),
    'ou':     dict(color='#d62728', linewidth=1.8, label='OU stationary'),
    'uniform':dict(color='#2ca02c', linewidth=1.8, linestyle='--',
                   label='Uniform fit'),
}


def _stationary_ou_pdf(grid: np.ndarray, fit) -> np.ndarray:
    """Stationary distribution of OU = N(mu, sigma_inf^2) where
    sigma_inf^2 = sigma^2 / (2 * theta - theta^2) for the discrete EM
    formulation (equivalent to sigma^2 / (1 - phi^2) with phi = 1 - theta)."""
    phi = 1.0 - fit.theta
    var_inf = (fit.sigma ** 2) / max(1.0 - phi ** 2, 1e-12)
    sd_inf = np.sqrt(var_inf)
    return stats.norm.pdf(grid, loc=fit.mu, scale=sd_inf)


def _uniform_pdf(grid: np.ndarray, fit) -> np.ndarray:
    lo, hi = fit.loc - fit.half_width, fit.loc + fit.half_width
    pdf = np.where((grid >= lo) & (grid <= hi), 1.0 / (hi - lo), 0.0)
    return pdf


def plot_regime_diagnostic(residuals: np.ndarray, cal: RegimeCalibration,
                           ax_hist, ax_qq, ax_acf,
                           xlim_quantile: float = 0.005,
                           acf_max_lag: int = 48) -> None:
    """Render the three diagnostic panels for one regime."""
    # ----- Histogram with overlay -----
    lo, hi = np.quantile(residuals, [xlim_quantile, 1 - xlim_quantile])
    grid = np.linspace(lo, hi, 600)
    ax_hist.hist(residuals, range=(lo, hi), **_STYLE['hist'])
    ax_hist.plot(grid, stats.norm.pdf(grid, cal.normal.mu, cal.normal.sigma),
                 **_STYLE['normal'])
    ax_hist.plot(grid, _stationary_ou_pdf(grid, cal.ornstein_uhlenbeck),
                 **_STYLE['ou'])
    ax_hist.plot(grid, _uniform_pdf(grid, cal.uniform),
                 **_STYLE['uniform'])
    ax_hist.set_xlim(lo, hi)
    ax_hist.set_xlabel('Relative forecast error (MGP - MI) / MI')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title(f"{cal.name}: empirical vs fitted")
    ax_hist.legend(loc='upper right', frameon=False, fontsize=8)
    ax_hist.text(
        0.02, 0.95,
        f"n = {cal.n_hours:,}\n"
        f"mean = {cal.mean:+.4f}\n"
        f"std  = {cal.std:.4f}\n"
        f"skew = {cal.skewness:.2f}\n"
        f"kurt = {cal.kurtosis_excess:.1f}",
        transform=ax_hist.transAxes, ha='left', va='top', fontsize=8,
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#999999', alpha=0.9))

    # ----- Q-Q plot vs Normal -----
    # Standardised empirical quantiles vs Normal(0,1) theoretical quantiles.
    z = (residuals - cal.normal.mu) / cal.normal.sigma
    z_sorted = np.sort(z)
    probs = (np.arange(1, len(z_sorted) + 1) - 0.5) / len(z_sorted)
    theoretical = stats.norm.ppf(probs)
    ax_qq.scatter(theoretical, z_sorted, s=2, alpha=0.4, color='#444444')
    # 45 degree reference line for the standardised QQ
    lims = [theoretical.min(), theoretical.max()]
    ax_qq.plot(lims, lims, color='#1f77b4', linewidth=1.3, linestyle='-')
    # Clip y to show tail departure without zooming away
    ax_qq.set_xlim(lims)
    ax_qq.set_ylim(np.percentile(z_sorted, 0.5),
                   np.percentile(z_sorted, 99.5))
    ax_qq.set_xlabel('Normal theoretical quantile')
    ax_qq.set_ylabel('Empirical quantile (standardised)')
    ax_qq.set_title('Q-Q vs Normal (heavy-tail diagnostic)')

    # ----- ACF -----
    lags = np.arange(1, acf_max_lag + 1)
    centred = residuals - residuals.mean()
    var = (centred ** 2).mean()
    acf = np.array([
        (centred[:-k] * centred[k:]).mean() / var for k in lags
    ])
    ax_acf.bar(lags, acf, color='#666666', edgecolor='none', width=0.8)
    # 95% CI under H0 of white noise (Bartlett): ±1.96 / sqrt(n)
    ci = 1.96 / np.sqrt(len(residuals))
    ax_acf.axhline(+ci, color='#1f77b4', linestyle=':', linewidth=1)
    ax_acf.axhline(-ci, color='#1f77b4', linestyle=':', linewidth=1)
    ax_acf.axhline(0, color='black', linewidth=0.5)
    ax_acf.set_xlabel('Lag (hours)')
    ax_acf.set_ylabel('Autocorrelation')
    ax_acf.set_title(f'ACF — ρ₁ = {cal.lag1_autocorr:.3f}')
    ax_acf.set_xlim(0, acf_max_lag + 1)


def plot_calibration_panel(residuals_by_regime: Dict[str, np.ndarray],
                           cals: Dict[str, RegimeCalibration],
                           output_path: str | Path,
                           figsize=(12, 6.5)) -> Path:
    """Produce a 2 row x 3 col figure summarising the calibration."""
    n_rows = len(cals)
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize, constrained_layout=True)
    if n_rows == 1:
        axes = axes.reshape(1, 3)
    for i, (name, cal) in enumerate(cals.items()):
        x = residuals_by_regime[name]
        plot_regime_diagnostic(x, cal,
                               ax_hist=axes[i, 0],
                               ax_qq=axes[i, 1],
                               ax_acf=axes[i, 2])
    fig.suptitle(
        "Empirical forecast-error calibration on PUN MGP vs MI-A1 (Italian market)",
        fontsize=11)
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    # Also save a vector copy if the user asks for PDF/SVG via extension
    if output_path.suffix.lower() == '.png':
        pdf_path = output_path.with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return output_path
