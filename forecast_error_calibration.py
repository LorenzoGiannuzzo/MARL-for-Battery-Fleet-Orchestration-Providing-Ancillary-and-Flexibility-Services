"""
Empirical calibration of forecast-error distributions on Italian PUN data.

Computes the empirical day-ahead vs intra-day price residual from PUN MGP and
MI-A1 hourly series, then fits the four parametric distributions used by the
existing ForecastErrorGenerator (uniform, normal, Ornstein-Uhlenbeck,
fixed-bias) by maximum-likelihood / method-of-moments. Output parameters are
directly pluggable into the env's error generator.

The empirical error is defined as a **relative** quantity to match the
existing API `forecast = true_price * (1 + error)`:

    error_t = (forecast_t - true_t) / true_t = (MGP_t - MI_t) / MI_t

Rows where the realized MI price is below `min_price_eur_mwh` (default 5)
are excluded because the relative error becomes numerically unstable.

A regime split is supported via the `regime_assignment` callback so the
calibration can be reported separately for gas-crisis 2022 vs normalized
2023-2024 (the two regimes differ substantially in error magnitude and
autocorrelation).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Optional, Tuple

import json
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================================
# Fit containers
# ============================================================================

@dataclass
class NormalFit:
    """Parameters of N(mu, sigma) fit on relative residuals."""
    mu: float
    sigma: float
    log_lik: float
    n: int

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.normal(self.mu, self.sigma, size=size)


@dataclass
class UniformFit:
    """Parameters of U(low, high) fit on relative residuals.

    Reported as the symmetric MLE on the centered series, i.e. low = -a,
    high = +a where a = max(|x - mean(x)|). We also report the empirical
    location 'loc' so the distribution is U(loc - a, loc + a).
    """
    loc: float
    half_width: float
    log_lik: float
    n: int

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return self.loc + rng.uniform(-self.half_width, self.half_width, size=size)


@dataclass
class OrnsteinUhlenbeckFit:
    """Discrete-time AR(1) fit, parameterised as the Euler-Maruyama OU.

        X_{t+1} = X_t + theta * (mu - X_t) + sigma * eps_t

    Equivalently AR(1) in the form X_{t+1} = phi * X_t + c + eps'
    with phi = 1 - theta and c = theta * mu.

    NB: the existing `OrnsteinUhlenbeckNoise` in drl_flexibility_analysis.py
    hardcodes mu = 0. To use these fitted parameters with a non-zero mean,
    extend that class to accept mu as a constructor argument.
    """
    theta: float       # mean-reversion rate (1/h)
    mu: float          # long-run mean (relative units)
    sigma: float       # innovation volatility per sqrt(h)
    lag1_autocorr: float
    log_lik: float
    n: int

    def sample_series(self, rng: np.random.Generator, n: int,
                      x0: Optional[float] = None) -> np.ndarray:
        x = self.mu if x0 is None else x0
        out = np.empty(n, dtype=float)
        for i in range(n):
            x = x + self.theta * (self.mu - x) + self.sigma * rng.standard_normal()
            out[i] = x
        return out


@dataclass
class FixedBiasFit:
    """Constant bias = mean(empirical relative error)."""
    bias: float
    rmse: float        # information only: std around the bias
    n: int


@dataclass
class RegimeCalibration:
    """All four fits on a single regime."""
    name: str
    n_hours: int
    mean: float
    std: float
    skewness: float
    kurtosis_excess: float
    lag1_autocorr: float
    normal: NormalFit
    uniform: UniformFit
    ornstein_uhlenbeck: OrnsteinUhlenbeckFit
    fixed_bias: FixedBiasFit

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop log_lik=NaN entries (will be set explicitly later) and round
        # for readability when dumping to JSON.
        return _round_floats(d, ndigits=6)


# ============================================================================
# Calibrator
# ============================================================================

class ForecastErrorCalibrator:
    """Fit forecast-error distributions to PUN MGP vs MI-A1 empirical residuals.

    Parameters
    ----------
    min_price_eur_mwh : float, default 5.0
        Drop rows where |MI| < this threshold (relative-error definition
        becomes unstable for near-zero realized prices).
    """

    def __init__(self, min_price_eur_mwh: float = 5.0):
        self.min_price = float(min_price_eur_mwh)

    # ----- residual construction -----

    def compute_residuals(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with columns ['mgp', 'mi', 'err_abs', 'err_rel']
        restricted to rows where both prices are present and |MI| >= threshold."""
        if 'pun_mgp' not in merged.columns or 'pun_mi' not in merged.columns:
            raise KeyError("merged DataFrame must contain 'pun_mgp' and 'pun_mi'")
        df = merged[['pun_mgp', 'pun_mi']].dropna().copy()
        df.columns = ['mgp', 'mi']
        df = df[df['mi'].abs() >= self.min_price]
        df['err_abs'] = df['mgp'] - df['mi']
        df['err_rel'] = df['err_abs'] / df['mi']
        return df

    # ----- individual fits -----

    @staticmethod
    def _fit_normal(x: np.ndarray) -> NormalFit:
        mu = float(np.mean(x))
        sigma = float(np.std(x, ddof=1))
        ll = float(np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma)))
        return NormalFit(mu=mu, sigma=sigma, log_lik=ll, n=len(x))

    @staticmethod
    def _fit_uniform(x: np.ndarray) -> UniformFit:
        # Symmetric uniform centred at empirical mean, half-width = max|x-mu|.
        # This is the MLE for the symmetric uniform family and avoids the
        # pathological zero-likelihood that the non-symmetric MLE produces
        # when fitting to (min, max).
        loc = float(np.mean(x))
        half = float(np.max(np.abs(x - loc)))
        # Log-likelihood of symmetric uniform = -n * log(2 * half_width).
        ll = float(-len(x) * np.log(2 * half)) if half > 0 else -np.inf
        return UniformFit(loc=loc, half_width=half, log_lik=ll, n=len(x))

    @staticmethod
    def _fit_ou(x: np.ndarray) -> OrnsteinUhlenbeckFit:
        # AR(1) regression: X_{t+1} = phi * X_t + c + eps
        # Then theta = 1 - phi, mu = c / theta, sigma = std(eps).
        x_lag = x[:-1]
        x_next = x[1:]
        # OLS slope and intercept via closed form
        x_lag_mean = x_lag.mean()
        x_next_mean = x_next.mean()
        cov = np.mean((x_lag - x_lag_mean) * (x_next - x_next_mean))
        var = np.mean((x_lag - x_lag_mean) ** 2)
        phi = float(cov / var) if var > 0 else 0.0
        c = float(x_next_mean - phi * x_lag_mean)
        residuals = x_next - (phi * x_lag + c)
        sigma_eps = float(np.std(residuals, ddof=2))
        # Convert AR(1) -> OU parametrisation
        theta = 1.0 - phi
        # Guard against pathological phi >= 1 (non-stationary) and phi <= -1
        # (oscillatory): in those cases the OU interpretation breaks down, but
        # we still return numbers and let the caller interpret.
        if abs(theta) < 1e-12:
            mu = float(np.mean(x))  # degenerate: random walk; use sample mean
        else:
            mu = c / theta
        ll = float(np.sum(stats.norm.logpdf(residuals, loc=0.0, scale=sigma_eps)))
        return OrnsteinUhlenbeckFit(
            theta=float(theta), mu=float(mu), sigma=float(sigma_eps),
            lag1_autocorr=float(phi), log_lik=ll, n=len(x))

    @staticmethod
    def _fit_fixed_bias(x: np.ndarray) -> FixedBiasFit:
        bias = float(np.mean(x))
        rmse = float(np.sqrt(np.mean((x - bias) ** 2)))
        return FixedBiasFit(bias=bias, rmse=rmse, n=len(x))

    # ----- regime-level driver -----

    def fit_regime(self, residuals: pd.DataFrame, name: str) -> RegimeCalibration:
        """Fit all four distributions on the 'err_rel' column."""
        x = residuals['err_rel'].to_numpy()
        if len(x) < 100:
            raise ValueError(f"Too few residuals to fit regime '{name}': n={len(x)}")
        # Empirical moments (informational, not used by the fits themselves)
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1))
        skew = float(stats.skew(x))
        kurt = float(stats.kurtosis(x))  # excess kurtosis (Normal = 0)
        lag1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        return RegimeCalibration(
            name=name,
            n_hours=len(x),
            mean=mean, std=std, skewness=skew, kurtosis_excess=kurt,
            lag1_autocorr=lag1,
            normal=self._fit_normal(x),
            uniform=self._fit_uniform(x),
            ornstein_uhlenbeck=self._fit_ou(x),
            fixed_bias=self._fit_fixed_bias(x),
        )

    def fit_by_regime(
        self, merged: pd.DataFrame,
        regime_assignment: Callable[[pd.DatetimeIndex], pd.Series],
    ) -> Dict[str, RegimeCalibration]:
        """Compute residuals and fit each regime separately.

        `regime_assignment` takes a DatetimeIndex and returns a Series of
        regime labels aligned to that index. Rows whose label is None or NaN
        are dropped.
        """
        residuals = self.compute_residuals(merged)
        labels = regime_assignment(residuals.index)
        residuals = residuals.assign(regime=labels.values).dropna(subset=['regime'])
        out: Dict[str, RegimeCalibration] = {}
        for name, grp in residuals.groupby('regime', sort=False):
            out[str(name)] = self.fit_regime(grp, str(name))
        return out


# ============================================================================
# Default regime split: 2022 (gas crisis) vs 2023-2024 (normalized)
# ============================================================================

def default_regime_assignment(index: pd.DatetimeIndex) -> pd.Series:
    """Assign each timestamp to a market-regime label.

    Returns 'gas_crisis_2022' for years <= 2022, 'normalized_2023_2024'
    for years >= 2023. None for anything else (caller drops NaN rows).
    """
    out = pd.Series(index=index, dtype=object)
    out[index.year == 2022] = 'gas_crisis_2022'
    out[index.year.isin([2023, 2024])] = 'normalized_2023_2024'
    return out


# ============================================================================
# Reporting helpers
# ============================================================================

def calibrations_to_table(cals: Dict[str, RegimeCalibration]) -> pd.DataFrame:
    """Flatten the regime calibrations to a publication-style table."""
    rows = []
    for name, c in cals.items():
        rows.append({
            'regime': name,
            'n_hours': c.n_hours,
            'empirical_mean': c.mean,
            'empirical_std': c.std,
            'empirical_skew': c.skewness,
            'empirical_kurt_excess': c.kurtosis_excess,
            'empirical_lag1_rho': c.lag1_autocorr,
            'normal_mu': c.normal.mu,
            'normal_sigma': c.normal.sigma,
            'normal_log_lik': c.normal.log_lik,
            'uniform_loc': c.uniform.loc,
            'uniform_half_width': c.uniform.half_width,
            'uniform_log_lik': c.uniform.log_lik,
            'ou_theta': c.ornstein_uhlenbeck.theta,
            'ou_mu': c.ornstein_uhlenbeck.mu,
            'ou_sigma': c.ornstein_uhlenbeck.sigma,
            'ou_log_lik': c.ornstein_uhlenbeck.log_lik,
            'fixed_bias': c.fixed_bias.bias,
        })
    return pd.DataFrame(rows).set_index('regime')


def save_calibrations_json(cals: Dict[str, RegimeCalibration], path: str) -> None:
    out = {name: c.to_dict() for name, c in cals.items()}
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)


# ============================================================================
# Utility
# ============================================================================

def _round_floats(obj, ndigits: int = 6):
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    if isinstance(obj, float):
        return round(obj, ndigits)
    return obj
