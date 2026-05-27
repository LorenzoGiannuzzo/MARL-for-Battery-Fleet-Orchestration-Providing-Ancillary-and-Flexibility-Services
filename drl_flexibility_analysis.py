"""
------------------------------------------------------------------------------------------------------------------------
BATTERY ENERGY STORAGE SYSTEM (BESS) OPTIMIZATION - PPO ONLY
PPO with Multiple Error Distributions + Monthly Retraining + PDF Export (DPI=500)
------------------------------------------------------------------------------------------------------------------------
Author: Lorenzo Giannuzzo
Affiliation: Politecnico di Torino - DENERG - Energy Center Lab

VERSION: PPO-ONLY
- 4 error distributions: uniform, normal, ornstein-uhlenbeck, fixed-bias
- Monthly retraining with ACTUAL forecast errors
- All plots saved in PNG (300 DPI) and PDF (500 DPI)
- Battery degradation model integrated
- Heatmap visualizations
- SOH tracking
------------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gymnasium as gym
import torch
import traceback

from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces
from collections import deque

# =================== CONFIGURATION ====================
TEST_FIRST_MONTH_ONLY = False
NOISE_TYPES_TO_TEST = ['uniform', 'normal', 'ornstein-uhlenbeck']
FORCE_RETRAIN = False

# ==================== BATTERY PARAMETERS ====================
FORECAST_ERROR_MIN = -0.15
FORECAST_ERROR_MAX = 0.15
BATTERY_CAPACITY = 4.0
BATTERY_POWER = 2.0
BATTERY_EFFICIENCY = 0.95
C_RATE = 0.5
SOC_MIN = 0.1
SOC_MAX = 0.9

# ==================== DEGRADATION PARAMETERS ====================
DEGRADATION_COST_PER_MWH = 25000  # EUR/MWh of throughput

# ==================== PPO PARAMETERS ====================
PPO_TOTAL_TIMESTEPS = 500000
PPO_LEARNING_RATE = 1e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_HIDDEN_DIM_1 = 1024
PPO_HIDDEN_DIM_2 = 512
PPO_ENT_COEF = 0.01
PPO_CLIP_RANGE = 0.2
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5

# ==================== ACTION SPACE ====================
N_DISCRETE_ACTIONS = 21

# ==================== REWARD PARAMETERS ====================
SOC_VIOLATION_PENALTY = 1000.0

# ==================== ENVIRONMENT ====================
FORECAST_HORIZON = 24

# ==================== MONTHLY RETRAINING PARAMETERS ====================
MONTHLY_RETRAINING_ENABLED = True
MONTHLY_RETRAINING_LOOKBACK = 60
MONTHLY_RETRAINING_TIMESTEPS = 500000
MONTHLY_RETRAINING_FREQUENCY = 30

TRAIN_FILE = '20231101_20231231_PUN.xlsx'
TEST_FILE = '20240101_20241231_PUN.xlsx'

np.random.seed(42)
torch.manual_seed(42)


# ==================== DEGRADATION MODEL ====================
def degradation(cycle_num):
    """
    Battery degradation function (remaining capacity %)
    Polynomial model based on CUMULATIVE cycle count
    """
    capacity_remaining = (
            -0.00000000000000000000000000000005613 * cycle_num ** 9 +
            0.000000000000000000000000003121 * cycle_num ** 8 -
            0.00000000000000000000006353 * cycle_num ** 7 +
            0.000000000000000000663 * cycle_num ** 6 -
            0.000000000000003987 * cycle_num ** 5 +
            0.00000000001435 * cycle_num ** 4 -
            0.0000000307 * cycle_num ** 3 +
            0.00003746 * cycle_num ** 2 -
            0.0277 * cycle_num + 100
    )
    return max(0, capacity_remaining)


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for time-correlated forecast errors.

    Discrete-time Euler-Maruyama update:

        X_{t+1} = X_t + theta * (mu - X_t) * dt + sigma * sqrt(dt) * eps_t

    where eps_t ~ N(0, 1). Equivalent to AR(1) with phi = 1 - theta*dt and
    intercept theta*mu*dt; the stationary mean is mu, the stationary
    variance is sigma^2 / (2*theta) (continuous limit) or
    sigma^2 / (1 - phi^2) for the discrete version.

    Parameters
    ----------
    theta : float, default 0.15
        Mean reversion rate (1/h).
    mu : float, default 0.0
        Long-run mean of the process. NB: prior to Block 2 (May 2026) this
        was hardcoded to 0; it is now exposed so empirically calibrated
        biases (e.g. +0.025 in gas-crisis 2022) can be honoured.
    sigma : float, default 0.2
        Innovation volatility per sqrt(h).
    dt : float, default 1.0
        Integration time step. Keep at 1.0 for hourly markets.
    clip_low, clip_high : float
        Output clipping bounds. Default matches the hardcoded
        FORECAST_ERROR_MIN/MAX (-0.15, +0.15) used historically by the
        non-OU branches; pass wider bounds when using empirically
        calibrated sigma values that legitimately exceed 0.15.
    """

    def __init__(self, theta=0.15, sigma=0.2, dt=1.0, mu=0.0,
                 clip_low=FORECAST_ERROR_MIN, clip_high=FORECAST_ERROR_MAX):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.x = mu  # start at the long-run mean (not 0) to avoid an
                     # initial transient when mu != 0.

    def sample(self):
        """Generate next correlated error sample."""
        dx = (self.theta * (self.mu - self.x) * self.dt
              + self.sigma * np.sqrt(self.dt) * np.random.randn())
        self.x += dx
        return np.clip(self.x, self.clip_low, self.clip_high)

    def reset(self):
        """Reset process to its initial state (long-run mean)."""
        self.x = self.mu


class ForecastErrorGenerator:
    """Stochastic forecast error generator for price predictions.

    Generates relative price errors that are applied multiplicatively:

        forecast_price = true_price * (1 + error)

    Four distribution families are supported: 'uniform', 'normal',
    'ornstein-uhlenbeck', and 'fixed-bias'.

    Default parameters (backward-compatible) use the historical hardcoded
    values: U(-0.15, +0.15) for uniform, N(0, 0.05) for normal (= MAX/3
    clipped to [-0.15, +0.15]), OU(theta=0.15, mu=0, sigma=0.2), and
    +10% for fixed-bias.

    To use empirically calibrated parameters (e.g. fit on the Italian PUN
    MGP vs MI-A1 residuals via Block 1), pass a `params` dict whose keys
    match the distribution branch:

        - 'uniform':           {'loc': float, 'half_width': float}
        - 'normal':            {'mu': float, 'sigma': float, 'clip_low': float,
                                'clip_high': float}
        - 'ornstein-uhlenbeck': {'theta': float, 'mu': float, 'sigma': float,
                                'clip_low': float, 'clip_high': float}
        - 'fixed-bias':        {'bias': float}

    Or, more conveniently, use the classmethod `from_calibration_json`.
    """

    # Wide clipping bounds used when the calibration produces a sigma that
    # exceeds the legacy +/-15% range. Matches roughly +/-5 std of the OU
    # stationary distribution for the 2022 gas-crisis regime.
    DEFAULT_WIDE_CLIP_LOW = -1.0
    DEFAULT_WIDE_CLIP_HIGH = +1.0

    def __init__(self, error_type='uniform', params=None):
        self.error_type = error_type
        self.params = dict(params) if params else {}
        self.ou_noise = None
        if error_type == 'ornstein-uhlenbeck':
            p = self.params
            self.ou_noise = OrnsteinUhlenbeckNoise(
                theta=p.get('theta', 0.15),
                mu=p.get('mu', 0.0),
                sigma=p.get('sigma', 0.2),
                clip_low=p.get('clip_low', FORECAST_ERROR_MIN),
                clip_high=p.get('clip_high', FORECAST_ERROR_MAX),
            )
        # Legacy attribute kept for backward compatibility; only used by the
        # 'fixed-bias' branch when no params are provided.
        self.fixed_bias = self.params.get('bias', 0.10)

    @classmethod
    def from_calibration_json(cls, json_path: str, regime: str,
                              error_type: str = 'ornstein-uhlenbeck',
                              wide_clip: bool = True):
        """Build a generator from a Block-1 calibrations.json file.

        Parameters
        ----------
        json_path : str
            Path to the JSON file produced by `forecast_error_calibration.
            save_calibrations_json`.
        regime : str
            Regime key inside the JSON (e.g. 'normalized_2023_2024',
            'gas_crisis_2022').
        error_type : str
            Which of the four branches to instantiate. The corresponding
            parameter block is read from the JSON.
        wide_clip : bool, default True
            If True, use wide (+/-100%) clipping bounds instead of the
            legacy +/-15%. Recommended whenever the calibrated sigma is
            comparable to or larger than 0.05, which is the case for
            Italian data.
        """
        import json as _json
        with open(json_path, 'r') as f:
            doc = _json.load(f)
        if regime not in doc:
            raise KeyError(f"regime '{regime}' not in {json_path} "
                           f"(available: {list(doc)})")
        block = doc[regime]
        params = {}
        if error_type == 'uniform':
            params = dict(block['uniform'])  # {'loc', 'half_width', ...}
        elif error_type == 'normal':
            params = {'mu': block['normal']['mu'],
                      'sigma': block['normal']['sigma']}
        elif error_type == 'ornstein-uhlenbeck':
            params = {'theta': block['ornstein_uhlenbeck']['theta'],
                      'mu': block['ornstein_uhlenbeck']['mu'],
                      'sigma': block['ornstein_uhlenbeck']['sigma']}
        elif error_type == 'fixed-bias':
            params = {'bias': block['fixed_bias']['bias']}
        else:
            raise ValueError(f"unknown error_type: {error_type}")
        if wide_clip:
            params.setdefault('clip_low',  cls.DEFAULT_WIDE_CLIP_LOW)
            params.setdefault('clip_high', cls.DEFAULT_WIDE_CLIP_HIGH)
        return cls(error_type=error_type, params=params)

    def generate_error(self):
        """Generate forecast error based on distribution type."""
        p = self.params
        if self.error_type == 'uniform':
            # Backward-compatible default: symmetric U(MIN, MAX) when no
            # params; otherwise U(loc - half, loc + half).
            if not p:
                return np.random.uniform(FORECAST_ERROR_MIN, FORECAST_ERROR_MAX)
            loc = p.get('loc', 0.0)
            half = p.get('half_width', FORECAST_ERROR_MAX)
            return float(loc + np.random.uniform(-half, +half))
        elif self.error_type == 'normal':
            if not p:
                error = np.random.normal(0.0, FORECAST_ERROR_MAX / 3.0)
                return float(np.clip(error, FORECAST_ERROR_MIN, FORECAST_ERROR_MAX))
            mu = p.get('mu', 0.0)
            sigma = p.get('sigma', FORECAST_ERROR_MAX / 3.0)
            lo = p.get('clip_low',  FORECAST_ERROR_MIN)
            hi = p.get('clip_high', FORECAST_ERROR_MAX)
            return float(np.clip(np.random.normal(mu, sigma), lo, hi))
        elif self.error_type == 'ornstein-uhlenbeck':
            return float(self.ou_noise.sample())
        elif self.error_type == 'fixed-bias':
            return float(p.get('bias', self.fixed_bias))
        else:
            return 0.0

    def apply_error(self, true_price):
        """Apply error to true price to generate forecast."""
        error = self.generate_error()
        forecast_price = true_price * (1 + error)
        return forecast_price, error

    def reset(self):
        """Reset error generator state."""
        if self.ou_noise is not None:
            self.ou_noise.reset()


class Battery:
    """Battery model with CUMULATIVE degradation tracking"""

    def __init__(self):
        self.capacity = BATTERY_CAPACITY
        self.nominal_capacity = BATTERY_CAPACITY
        self.max_power = min(BATTERY_POWER, BATTERY_CAPACITY * C_RATE)
        self.efficiency = BATTERY_EFFICIENCY
        self.soc = 0.5
        self.soc_min = SOC_MIN
        self.soc_max = SOC_MAX
        self.cumulative_throughput_kwh = 0.0
        self.equivalent_cycles = 0.0

    def step(self, action):
        """Execute battery action and update state"""
        action = np.clip(action, -self.max_power, self.max_power)

        # Charging mode
        if action > 0.01:
            if self.soc >= self.soc_max - 1e-6:
                return 0.0

            max_energy_storable = (self.soc_max - self.soc) * self.capacity
            max_power_from_grid = max_energy_storable / self.efficiency
            actual_power_from_grid = min(action, max_power_from_grid)

            if actual_power_from_grid > 0.01:
                energy_stored = actual_power_from_grid * self.efficiency
                self.soc += energy_stored / self.capacity
                self.soc = min(self.soc, self.soc_max)
                self.cumulative_throughput_kwh += energy_stored * 1000
                return actual_power_from_grid
            return 0.0

        # Discharging mode
        elif action < -0.01:
            if self.soc <= self.soc_min + 1e-6:
                return 0.0

            max_energy_available = (self.soc - self.soc_min) * self.capacity
            max_power_to_grid = max_energy_available * self.efficiency
            actual_power_to_grid = min(-action, max_power_to_grid)

            if actual_power_to_grid > 0.01:
                energy_from_battery = actual_power_to_grid / self.efficiency
                self.soc -= energy_from_battery / self.capacity
                self.soc = max(self.soc, self.soc_min)
                self.cumulative_throughput_kwh += energy_from_battery * 1000
                return -actual_power_to_grid
            return 0.0

        return 0.0

    def update_degradation(self):
        """Update capacity based on CUMULATIVE cycles"""
        self.equivalent_cycles = self.cumulative_throughput_kwh / (2 * self.nominal_capacity * 1000)
        capacity_percentage = degradation(self.equivalent_cycles)
        self.capacity = self.nominal_capacity * (capacity_percentage / 100.0)

    def get_soh(self):
        """Get State of Health percentage"""
        return (self.capacity / self.nominal_capacity) * 100.0

    def reset(self):
        """Reset battery to initial state"""
        self.soc = 0.5

    def copy(self):
        """Create deep copy of battery state"""
        b = Battery()
        b.soc = self.soc
        b.capacity = self.capacity
        b.equivalent_cycles = self.equivalent_cycles
        b.cumulative_throughput_kwh = self.cumulative_throughput_kwh
        return b


class BatteryTradingEnvClean(gym.Env):
    """PPO Environment - 26 features, simple reward with degradation cost"""

    def __init__(self, prices, error_generator):
        super(BatteryTradingEnvClean, self).__init__()
        self.prices = prices
        self.error_gen = error_generator
        self.battery = Battery()
        self.current_step = 0
        self.max_steps = len(prices)
        self.forecast_horizon = FORECAST_HORIZON

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.action_values = np.linspace(-BATTERY_POWER, BATTERY_POWER, N_DISCRETE_ACTIONS)

        n_features = 1 + FORECAST_HORIZON + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=2.0,
            shape=(n_features,),
            dtype=np.float32
        )

        self.episode_length = 480

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.battery.reset()
        self.error_gen.reset()
        self.current_step = 0

        if self.max_steps > 1000:
            self.current_step = np.random.randint(0, min(1000, self.max_steps - 500))

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        """Construct observation vector with SOC, forecast prices, and current price"""
        if self.current_step >= len(self.prices):
            return np.zeros(26, dtype=np.float32)

        soc = self.battery.soc
        current_price = self.prices[self.current_step]

        forecast_prices_norm = []
        for h in range(self.forecast_horizon):
            if self.current_step + h < len(self.prices):
                true_future_price = self.prices[self.current_step + h]
                forecast_price, _ = self.error_gen.apply_error(true_future_price)
                forecast_prices_norm.append(forecast_price / 200.0)
            else:
                forecast_prices_norm.append(0.5)

        current_price_norm = current_price / 200.0

        obs = np.array([
            soc,
            *forecast_prices_norm,
            current_price_norm
        ], dtype=np.float32)

        return obs

    def step(self, action_idx):
        """Execute action and return reward with degradation cost"""
        action = float(self.action_values[action_idx])
        true_price = self.prices[self.current_step]

        energy = self.battery.step(action)
        profit = -energy * true_price

        degradation_cost = abs(energy) * DEGRADATION_COST_PER_MWH / (2 * 6000)

        soc_penalty = 0.0
        if self.battery.soc > SOC_MAX:
            soc_penalty = -abs(self.battery.soc - SOC_MAX) * SOC_VIOLATION_PENALTY
        elif self.battery.soc < SOC_MIN:
            soc_penalty = -abs(self.battery.soc - SOC_MIN) * SOC_VIOLATION_PENALTY

        reward = profit - degradation_cost + soc_penalty

        self.current_step += 1
        done = self.current_step >= min(self.max_steps, self.episode_length)
        truncated = False

        if done:
            obs = self._get_observation() if self.current_step < self.max_steps else np.zeros(26, dtype=np.float32)
        else:
            obs = self._get_observation()

        info = {
            'energy': energy,
            'price': true_price,
            'profit': profit,
            'degradation_cost': degradation_cost,
            'soc': self.battery.soc
        }

        return obs, reward, done, truncated, info


class BatteryTradingEnvWithActualForecasts(gym.Env):
    """Environment that uses ACTUAL forecast errors from testing history"""

    def __init__(self, true_prices, forecast_history, offset=0):
        super(BatteryTradingEnvWithActualForecasts, self).__init__()
        self.true_prices = true_prices
        self.forecast_history = forecast_history
        self.offset = offset

        self.battery = Battery()
        self.current_step = 0
        self.max_steps = len(true_prices)
        self.forecast_horizon = FORECAST_HORIZON

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.action_values = np.linspace(-BATTERY_POWER, BATTERY_POWER, N_DISCRETE_ACTIONS)

        n_features = 1 + FORECAST_HORIZON + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=2.0,
            shape=(n_features,),
            dtype=np.float32
        )

        self.episode_length = 480
        self.forecast_lookup = {f['hour']: f['forecast_price'] for f in forecast_history}

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.battery.reset()
        self.current_step = 0

        if self.max_steps > 1000:
            self.current_step = np.random.randint(0, min(1000, self.max_steps - 500))

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        """Construct observation using actual forecast history"""
        if self.current_step >= len(self.true_prices):
            return np.zeros(26, dtype=np.float32)

        soc = self.battery.soc
        current_price = self.true_prices[self.current_step]

        forecast_prices_norm = []
        for h in range(self.forecast_horizon):
            if self.current_step + h < len(self.true_prices):
                global_hour = self.current_step + h + self.offset

                if global_hour in self.forecast_lookup:
                    forecast_price = self.forecast_lookup[global_hour]
                else:
                    forecast_price = self.true_prices[self.current_step + h]

                forecast_prices_norm.append(forecast_price / 200.0)
            else:
                forecast_prices_norm.append(0.5)

        current_price_norm = current_price / 200.0

        obs = np.array([
            soc,
            *forecast_prices_norm,
            current_price_norm
        ], dtype=np.float32)

        return obs

    def step(self, action_idx):
        """Execute action and return reward with degradation cost"""
        action = float(self.action_values[action_idx])
        true_price = self.true_prices[self.current_step]

        energy = self.battery.step(action)
        profit = -energy * true_price

        degradation_cost = abs(energy) * DEGRADATION_COST_PER_MWH / (2 * 6000)

        soc_penalty = 0.0
        if self.battery.soc > SOC_MAX:
            soc_penalty = -abs(self.battery.soc - SOC_MAX) * SOC_VIOLATION_PENALTY
        elif self.battery.soc < SOC_MIN:
            soc_penalty = -abs(self.battery.soc - SOC_MIN) * SOC_VIOLATION_PENALTY

        reward = profit - degradation_cost + soc_penalty

        self.current_step += 1
        done = self.current_step >= min(self.max_steps, self.episode_length)
        truncated = False

        if done:
            obs = self._get_observation() if self.current_step < self.max_steps else np.zeros(26, dtype=np.float32)
        else:
            obs = self._get_observation()

        info = {
            'energy': energy,
            'price': true_price,
            'profit': profit,
            'degradation_cost': degradation_cost,
            'soc': self.battery.soc
        }

        return obs, reward, done, truncated, info


class TrainingCallback(BaseCallback):
    """Training callback for monitoring PPO progress"""

    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.rollout_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Called at end of each rollout to print training statistics"""
        self.rollout_count += 1

        if self.rollout_count % 5 == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            else:
                mean_reward = 0

            reward_per_day = mean_reward / 20
            reward_per_year = reward_per_day * 365

            status = "LOSS" if mean_reward < 0 else "LEARNING" if mean_reward < 2000 else "GOOD" if mean_reward < 5000 else "EXCELLENT"

            print(f"  Rollout {self.rollout_count:3d} | "
                  f"R(20d): EUR {mean_reward:>8,.0f} | "
                  f"~ EUR {reward_per_year:>8,.0f}/yr | "
                  f"{status}")


def save_plot_as_pdf(fig, filename_base, results_folder, dpi=500):
    """Helper function to save plots as both PNG and PDF"""
    filename_png = os.path.join(results_folder, f'{filename_base}.png')
    fig.savefig(filename_png, dpi=300, bbox_inches='tight')

    filename_pdf = os.path.join(results_folder, f'{filename_base}.pdf')
    fig.savefig(filename_pdf, dpi=dpi, bbox_inches='tight', format='pdf')

    return filename_png, filename_pdf


def plot_error_types_comparison(results_folder):
    """Plot showing the different error types characteristics"""
    try:
        print(f"\n  Generating error types comparison plot...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        np.random.seed(42)
        hours = 168
        base_price = 100.0

        time = np.arange(hours)
        true_prices = base_price + 30 * np.sin(2 * np.pi * time / 24) + 10 * np.sin(2 * np.pi * time / 168)

        error_generators = {
            'uniform': ForecastErrorGenerator('uniform'),
            'normal': ForecastErrorGenerator('normal'),
            'ornstein-uhlenbeck': ForecastErrorGenerator('ornstein-uhlenbeck'),
            'fixed-bias': ForecastErrorGenerator('fixed-bias')
        }

        error_titles = {
            'uniform': 'UNIFORM ERROR\nRandom ±15% (each hour independent)',
            'normal': 'NORMAL ERROR\nGaussian σ=5% (concentrated near 0%)',
            'ornstein-uhlenbeck': 'ORNSTEIN-UHLENBECK ERROR\nTime-correlated (realistic persistence)',
            'fixed-bias': 'FIXED-BIAS ERROR\nDeterministic +10% (always overestimate)'
        }

        error_descriptions = {
            'uniform': 'Completely random errors\nEqual probability for all values in [-15%, +15%]',
            'normal': 'Gaussian distribution\nMost errors near 0%, rare extreme values',
            'ornstein-uhlenbeck': 'Autocorrelated errors\nIf hour N is +5%, hour N+1 likely similar',
            'fixed-bias': 'Constant bias\nAlways predicts 10% higher than true price'
        }

        plot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        error_types_list = ['uniform', 'normal', 'ornstein-uhlenbeck', 'fixed-bias']

        for idx, (pos, error_type) in enumerate(zip(plot_positions, error_types_list)):
            ax = axes[pos]

            error_gen = error_generators[error_type]
            error_gen.reset()

            forecast_prices = []
            errors_pct = []

            for t in range(hours):
                forecast_price, error = error_gen.apply_error(true_prices[t])
                forecast_prices.append(forecast_price)
                errors_pct.append(error * 100)

            ax.plot(time, true_prices, 'k-', linewidth=2.5, label='True Price', alpha=0.9)
            ax.plot(time, forecast_prices, 'r--', linewidth=2, label='Forecast Price', alpha=0.7)

            ax.fill_between(time, true_prices, forecast_prices, alpha=0.2, color='red')

            ax.set_xlabel('Hour', fontweight='bold', fontsize=11)
            ax.set_ylabel('Price (EUR/MWh)', fontweight='bold', fontsize=11)
            ax.set_title(error_titles[error_type], fontweight='bold', fontsize=12)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)

            mean_error = np.mean(errors_pct)
            std_error = np.std(errors_pct)
            min_error = np.min(errors_pct)
            max_error = np.max(errors_pct)

            stats_text = (f'Mean: {mean_error:+.1f}%\n'
                          f'Std: {std_error:.1f}%\n'
                          f'Range: [{min_error:+.1f}%, {max_error:+.1f}%]')

            ax.text(0.02, 0.98, stats_text,
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    fontsize=9, fontweight='bold')

            ax.text(0.98, 0.02, error_descriptions[error_type],
                    transform=ax.transAxes, verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                    fontsize=8, style='italic')

        plt.suptitle('Comparison of 4 Forecast Error Distributions\n'
                     'One Week Simulation (168 hours)',
                     fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        filename_png, filename_pdf = save_plot_as_pdf(fig, 'error_types_comparison', results_folder)

        plt.close()
        print(f"  Saved error types comparison (PNG): {filename_png}")
        print(f"  Saved error types comparison (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR plotting error types comparison: {e}")
        traceback.print_exc()
        plt.close('all')


def plot_heatmaps(results, noise_type, results_folder):
    """Generate heatmap visualizations for SOC, prices, battery charge/discharge"""
    try:
        print(f"  Generating heatmaps for {noise_type}...")

        safe_name = noise_type.replace('ornstein-uhlenbeck', 'ou')

        n_hours = len(results)
        n_days = n_hours // 24

        if n_days == 0:
            print(f"  Not enough data for heatmaps")
            return

        results_truncated = results.iloc[:n_days * 24].copy()

        soc_matrix = results_truncated['PPO_SOC'].values.reshape(n_days, 24) * 100
        price_matrix = results_truncated['True_Price'].values.reshape(n_days, 24)

        energy_matrix = results_truncated['PPO_Energy_Actual'].values.reshape(n_days, 24)
        charge_matrix = np.where(energy_matrix > 0, energy_matrix, 0)
        discharge_matrix = np.where(energy_matrix < 0, -energy_matrix, 0)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Price heatmap
        ax = axes[0, 0]
        im = ax.imshow(price_matrix, aspect='auto', cmap='plasma', interpolation='bilinear')
        ax.set_xlabel('Hour of Day', fontweight='bold', fontsize=11)
        ax.set_ylabel('Day of Year', fontweight='bold', fontsize=11)
        ax.set_title('Price [EUR/MWh]', fontweight='bold', fontsize=12)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xticklabels(np.arange(0, 24, 3))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('EUR/MWh', fontweight='bold')

        # Battery Charge heatmap
        ax = axes[0, 1]
        im = ax.imshow(charge_matrix, aspect='auto', cmap='YlOrRd', interpolation='bilinear',
                       vmin=0, vmax=np.max(charge_matrix) if np.max(charge_matrix) > 0 else 1)
        ax.set_xlabel('Hour of Day', fontweight='bold', fontsize=11)
        ax.set_ylabel('Day of Year', fontweight='bold', fontsize=11)
        ax.set_title('Battery Charge [MW]', fontweight='bold', fontsize=12)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xticklabels(np.arange(0, 24, 3))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('MW', fontweight='bold')

        # Battery Discharge heatmap
        ax = axes[1, 0]
        im = ax.imshow(discharge_matrix, aspect='auto', cmap='Blues', interpolation='bilinear',
                       vmin=0, vmax=np.max(discharge_matrix) if np.max(discharge_matrix) > 0 else 1)
        ax.set_xlabel('Hour of Day', fontweight='bold', fontsize=11)
        ax.set_ylabel('Day of Year', fontweight='bold', fontsize=11)
        ax.set_title('Battery Discharge [MW]', fontweight='bold', fontsize=12)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xticklabels(np.arange(0, 24, 3))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('MW', fontweight='bold')

        # SOC heatmap
        ax = axes[1, 1]
        im = ax.imshow(soc_matrix, aspect='auto', cmap='RdYlGn', interpolation='bilinear',
                       vmin=0, vmax=100)
        ax.set_xlabel('Hour of Day', fontweight='bold', fontsize=11)
        ax.set_ylabel('Day of Year', fontweight='bold', fontsize=11)
        ax.set_title('State of Charge [%]', fontweight='bold', fontsize=12)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xticklabels(np.arange(0, 24, 3))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('SOC (%)', fontweight='bold')

        plt.suptitle(f'PPO BESS Operation Heatmaps - {noise_type.upper()}\n'
                     f'Year-long Analysis ({n_days} days x 24 hours)',
                     fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        filename_png, filename_pdf = save_plot_as_pdf(fig, f'heatmaps_{safe_name}', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR generating heatmaps: {e}")
        traceback.print_exc()
        plt.close('all')


def plot_soh_evolution(results, noise_type, results_folder):
    """Plot State of Health evolution over time"""
    try:
        print(f"  Generating SOH evolution plot for {noise_type}...")

        safe_name = noise_type.replace('ornstein-uhlenbeck', 'ou')

        # Calculate SOH from throughput
        ppo_throughput = []
        ppo_soh = []

        cumulative_ppo_throughput = 0.0

        for idx in range(len(results)):
            cumulative_ppo_throughput += abs(results['PPO_Energy_Actual'].iloc[idx])
            ppo_throughput.append(cumulative_ppo_throughput)

            ppo_cycles = (cumulative_ppo_throughput * 1000) / (2 * BATTERY_CAPACITY * 1000)
            ppo_capacity_pct = degradation(ppo_cycles)
            ppo_soh.append(ppo_capacity_pct)

        results['PPO_SOH'] = ppo_soh
        results['PPO_Throughput_MWh'] = ppo_throughput

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # SOH over time
        ax = axes[0]
        ax.plot(results['Hour'], results['PPO_SOH'],
                label='PPO', color='coral', linewidth=2.5, alpha=0.8)

        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Initial SOH')
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, linewidth=1, label='EOL Threshold (80%)')

        ax.set_xlabel('Hour', fontweight='bold', fontsize=12)
        ax.set_ylabel('SOH (%)', fontweight='bold', fontsize=12)
        ax.set_title(f'State of Health Evolution - {noise_type.upper()}',
                     fontweight='bold', fontsize=13)
        ax.legend(fontsize=11, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([75, 101])

        final_ppo_soh = results['PPO_SOH'].iloc[-1]
        stats_text = f'Final SOH: {final_ppo_soh:.2f}%'

        ax.text(0.98, 0.02, stats_text,
                transform=ax.transAxes, verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                fontsize=10, fontweight='bold')

        # Cumulative throughput
        ax = axes[1]
        ax.plot(results['Hour'], results['PPO_Throughput_MWh'],
                label='PPO', color='coral', linewidth=2.5, alpha=0.8)

        ax.set_xlabel('Hour', fontweight='bold', fontsize=12)
        ax.set_ylabel('Cumulative Throughput (MWh)', fontweight='bold', fontsize=12)
        ax.set_title('Cumulative Energy Throughput', fontweight='bold', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        final_ppo_throughput = results['PPO_Throughput_MWh'].iloc[-1]
        stats_text = f'Final Throughput: {final_ppo_throughput:.1f} MWh'

        ax.text(0.98, 0.02, stats_text,
                transform=ax.transAxes, verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8),
                fontsize=10, fontweight='bold')

        # SOH degradation rate
        ax = axes[2]

        window = 24 * 7  # 7-day rolling window
        ppo_soh_series = pd.Series(results['PPO_SOH'])
        ppo_deg_rate = -ppo_soh_series.diff().rolling(window=window).mean() * 24

        ax.plot(results['Hour'], ppo_deg_rate,
                label='PPO', color='coral', linewidth=2, alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xlabel('Hour', fontweight='bold', fontsize=12)
        ax.set_ylabel('Degradation Rate (%/day)', fontweight='bold', fontsize=12)
        ax.set_title('SOH Degradation Rate (7-day rolling average)', fontweight='bold', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        avg_ppo_deg = ppo_deg_rate[window:].mean()
        stats_text = f'Avg Degradation Rate: {avg_ppo_deg:.4f}%/day'

        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8),
                fontsize=10, fontweight='bold')

        plt.tight_layout()

        filename_png, filename_pdf = save_plot_as_pdf(fig, f'soh_evolution_{safe_name}', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR generating SOH evolution: {e}")
        traceback.print_exc()
        plt.close('all')


def train_ppo(prices_df, noise_type, model_path):
    """Initial PPO training"""
    error_descriptions = {
        'uniform': 'Uniform distribution [-15%, +15%]',
        'normal': 'Normal distribution (σ=5%, clipped ±15%)',
        'ornstein-uhlenbeck': 'O-U process (time-correlated, ±15%)',
        'fixed-bias': 'Fixed bias +10% (deterministic)'
    }

    print(f"\nINITIAL TRAINING - {noise_type.upper()}")
    print(f"  Error type: {error_descriptions.get(noise_type, noise_type)}")
    print("=" * 70)

    prices = prices_df['EUR/MWh'].values
    error_gen = ForecastErrorGenerator(noise_type)
    env = BatteryTradingEnvClean(prices, error_gen)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[PPO_HIDDEN_DIM_1, PPO_HIDDEN_DIM_2],
            vf=[PPO_HIDDEN_DIM_1, PPO_HIDDEN_DIM_2]
        )
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=PPO_LEARNING_RATE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=PPO_ENT_COEF,
        vf_coef=PPO_VF_COEF,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device='auto'
    )

    callback = TrainingCallback()

    print(f"\nSETUP:")
    print(f"  Features:    26")
    print(f"  Network:     {PPO_HIDDEN_DIM_1}-{PPO_HIDDEN_DIM_2}")
    print(f"  Timesteps:   {PPO_TOTAL_TIMESTEPS:,}")
    print(f"  Degradation: {DEGRADATION_COST_PER_MWH} EUR/MWh")
    print(f"  Device:      {model.device}")
    print("=" * 70)
    print(f"\nStarting initial training...\n")

    model.learn(
        total_timesteps=PPO_TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=True
    )

    model.save(model_path)
    print(f"\nInitial training complete. Model saved: {model_path}")

    return model


def monthly_retrain_ppo(ppo_model, all_prices, current_hour, noise_type, error_gen, forecast_history, n_test_hours):
    """Monthly retraining with ACTUAL forecast errors"""
    lookback_hours = MONTHLY_RETRAINING_LOOKBACK * 24
    start_hour = max(0, current_hour - lookback_hours)

    retrain_true_prices = all_prices[start_hour:current_hour]

    print(f"     Retraining on {len(retrain_true_prices)} hours ({len(retrain_true_prices) / 24:.0f} days)")
    print(f"     Period: hour {start_hour} to {current_hour} (absolute)")

    n_train_hours = len(all_prices) - n_test_hours

    if current_hour > n_train_hours:
        test_hour_end = current_hour - n_train_hours
        test_hour_start = max(0, start_hour - n_train_hours)

        relevant_forecasts = []
        for f in forecast_history:
            if test_hour_start <= f['hour'] < test_hour_end:
                relevant_forecasts.append(f)

        if len(relevant_forecasts) > 0:
            print(f"     Found {len(relevant_forecasts)} ACTUAL forecast observations")

            temp_env = BatteryTradingEnvWithActualForecasts(
                retrain_true_prices,
                relevant_forecasts,
                offset=test_hour_start
            )
        else:
            print(f"     Using random errors as fallback")
            temp_env = BatteryTradingEnvClean(retrain_true_prices, error_gen)
    else:
        print(f"     Retraining window entirely in training period")
        temp_env = BatteryTradingEnvClean(retrain_true_prices, error_gen)

    original_env = ppo_model.env
    ppo_model.set_env(temp_env)

    print(f"     Training for {MONTHLY_RETRAINING_TIMESTEPS:,} timesteps...")

    ppo_model.learn(
        total_timesteps=MONTHLY_RETRAINING_TIMESTEPS,
        reset_num_timesteps=False,
        progress_bar=False
    )

    ppo_model.set_env(original_env)

    print(f"     Retraining complete")


def test_ppo(ppo_model, prices_df, all_train_test_prices, noise_type, error_gen):
    """Test PPO with MONTHLY RETRAINING"""
    prices = prices_df['EUR/MWh'].values
    dates = pd.to_datetime(prices_df['Data'], dayfirst=True)
    n_hours = len(prices)

    print(f"  Testing on {n_hours:,} hours ({n_hours / 24:.0f} days)")

    if MONTHLY_RETRAINING_ENABLED:
        expected_retrains = n_hours // (MONTHLY_RETRAINING_FREQUENCY * 24)
        print(f"  MONTHLY RETRAINING:")
        print(f"     Frequency:  Every {MONTHLY_RETRAINING_FREQUENCY} days")
        print(f"     Lookback:   {MONTHLY_RETRAINING_LOOKBACK} days")
        print(f"     Timesteps:  {MONTHLY_RETRAINING_TIMESTEPS:,} per retrain")
        print(f"     Expected:   ~{expected_retrains} retrainings")

    action_values = np.linspace(-BATTERY_POWER, BATTERY_POWER, N_DISCRETE_ACTIONS)

    battery_ppo = Battery()
    ppo_energy_actual = []
    ppo_socs = []
    ppo_instant_profits = []
    ppo_profits = []
    ppo_degradation_costs = []

    forecast_prices_all = []
    actual_forecast_history = []

    cumulative_ppo = 0
    cumulative_ppo_degradation = 0

    monthly_retrains = 0
    retrain_times = []
    performance_before_retrain = []

    progress_markers = [0] + [int(n_hours * i / 20) for i in range(1, 21)]
    start_time = datetime.now()

    test_start_offset = len(all_train_test_prices) - len(prices)

    for t in range(n_hours):
        true_price = prices[t]
        current_day = t // 24

        horizon = min(FORECAST_HORIZON, n_hours - t)
        forecast_window = []

        for h in range(horizon):
            if t + h < n_hours:
                fp, _ = error_gen.apply_error(prices[t + h])
                forecast_window.append(fp)

        current_forecast = forecast_window[0] if len(forecast_window) > 0 else true_price
        forecast_prices_all.append(current_forecast)

        actual_forecast_history.append({
            'hour': t,
            'true_price': true_price,
            'forecast_price': current_forecast
        })

        # Monthly retraining
        if MONTHLY_RETRAINING_ENABLED and t > 0 and current_day > 0 and current_day % MONTHLY_RETRAINING_FREQUENCY == 0:
            if current_day not in [rt // 24 for rt in retrain_times]:
                print(f"\n  MONTHLY RETRAINING #{monthly_retrains + 1} at day {current_day} (hour {t})...")

                recent_window = min(MONTHLY_RETRAINING_FREQUENCY * 24, len(ppo_instant_profits))
                perf_before = sum(ppo_instant_profits[-recent_window:])
                performance_before_retrain.append(perf_before)

                absolute_hour = test_start_offset + t

                monthly_retrain_ppo(
                    ppo_model,
                    all_train_test_prices,
                    absolute_hour,
                    noise_type,
                    error_gen,
                    actual_forecast_history,
                    len(prices)
                )

                monthly_retrains += 1
                retrain_times.append(t)

                print(f"  Retrained! Total retrains: {monthly_retrains}")
                print()

        # PPO action
        soc = battery_ppo.soc
        forecast_prices_norm = []
        for h in range(FORECAST_HORIZON):
            if h < len(forecast_window):
                forecast_prices_norm.append(forecast_window[h] / 200.0)
            else:
                forecast_prices_norm.append(0.5)

        current_price_norm = true_price / 200.0

        obs = np.array([
            soc,
            *forecast_prices_norm,
            current_price_norm
        ], dtype=np.float32)

        action_idx, _ = ppo_model.predict(obs, deterministic=True)
        action_ppo = float(action_values[action_idx])

        energy_ppo = battery_ppo.step(action_ppo)

        profit_ppo = -energy_ppo * true_price
        degradation_cost_ppo = abs(energy_ppo) * DEGRADATION_COST_PER_MWH / (2 * 6000)

        cumulative_ppo += profit_ppo
        cumulative_ppo_degradation += degradation_cost_ppo

        ppo_energy_actual.append(energy_ppo)
        ppo_socs.append(battery_ppo.soc)
        ppo_profits.append(cumulative_ppo)
        ppo_instant_profits.append(profit_ppo)
        ppo_degradation_costs.append(cumulative_ppo_degradation)

        # Progress tracking
        if t in progress_markers and t > 0:
            progress_pct = int((t / n_hours * 100))
            elapsed = (datetime.now() - start_time).total_seconds()
            remaining = elapsed * (n_hours - t) / t
            bar_length = 30
            filled = int(bar_length * t / n_hours)
            bar = '#' * filled + '.' * (bar_length - filled)

            info_str = ""
            if MONTHLY_RETRAINING_ENABLED:
                info_str += f" | Retrains:{monthly_retrains}"

            print(f"  [{bar}] {progress_pct:3d}% | "
                  f"PPO: EUR {cumulative_ppo:>8,.0f}"
                  f"{info_str} | ETA: {remaining / 60:.1f}m", end='\r')

    print()

    results = pd.DataFrame({
        'Hour': range(len(ppo_energy_actual)),
        'Date': dates[:len(ppo_energy_actual)],
        'True_Price': prices[:len(ppo_energy_actual)],
        'Forecast_Price': forecast_prices_all[:len(ppo_energy_actual)],
        'PPO_Energy_Actual': ppo_energy_actual,
        'PPO_SOC': ppo_socs,
        'PPO_Instant_Profit': ppo_instant_profits,
        'PPO_Cumulative_Profit': ppo_profits,
        'PPO_Cumulative_Degradation_Cost': ppo_degradation_costs
    })

    retraining_stats = {
        'total_retrains': monthly_retrains,
        'retrain_times': retrain_times,
        'performance_before': performance_before_retrain,
        'enabled': MONTHLY_RETRAINING_ENABLED
    }

    degradation_info = {
        'ppo_total_degradation': cumulative_ppo_degradation
    }

    return results, cumulative_ppo, retraining_stats, degradation_info


def analyze_trading_behavior(results, noise_type):
    """Trading behavior analysis"""
    print(f"\n  TRADING BEHAVIOR - {noise_type.upper()}:")
    print(f"  {'-' * 70}")

    energy_col = 'PPO_Energy_Actual'

    buys = (results[energy_col] > 0.1).sum()
    sells = (results[energy_col] < -0.1).sum()
    inactive = (results[energy_col].abs() < 0.1).sum()

    avg_buy_price = results[results[energy_col] > 0.1]['True_Price'].mean() if buys > 0 else 0
    avg_sell_price = results[results[energy_col] < -0.1]['True_Price'].mean() if sells > 0 else 0

    soc_min_obs = results['PPO_SOC'].min()
    soc_max_obs = results['PPO_SOC'].max()

    print(f"\n  PPO:")
    print(f"    Buys:  {buys:>5} hours (avg: EUR {avg_buy_price:>6.2f})")
    print(f"    Sells: {sells:>5} hours (avg: EUR {avg_sell_price:>6.2f})")
    print(f"    Idle:  {inactive:>5} hours ({inactive / len(results) * 100:.1f}%)")

    if buys > 0 and sells > 0:
        spread = avg_sell_price - avg_buy_price
        print(f"    Spread: EUR {spread:>6.2f}", end="")
        if spread > 0:
            print(f" (positive)")
        else:
            print(f" (negative)")

    print(f"    SOC: [{soc_min_obs * 100:.1f}%, {soc_max_obs * 100:.1f}%]")

    print(f"  {'-' * 70}")


def plot_monthly_retraining_performance(results, retrain_stats, noise_type, results_folder):
    """Plot retraining performance"""
    safe_name = noise_type.replace('ornstein-uhlenbeck', 'ou')

    try:
        if not retrain_stats['enabled'] or retrain_stats['total_retrains'] == 0:
            print(f"  Skipping retraining plot for {noise_type}")
            return

        print(f"  Generating monthly retraining plot for {noise_type}...")

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # Cumulative profit with retraining markers
        ax = axes[0]
        ax.plot(results['Hour'], results['PPO_Cumulative_Profit'],
                label='PPO (Monthly Retraining)', color='coral', linewidth=2.5, alpha=0.8)

        for retrain_time in retrain_stats['retrain_times']:
            ax.axvline(x=retrain_time, color='purple', linestyle='--', alpha=0.6, linewidth=2)

        ax.set_xlabel('Hour', fontweight='bold', fontsize=12)
        ax.set_ylabel('Cumulative Profit (EUR)', fontweight='bold', fontsize=12)
        ax.set_title(f'{noise_type.upper()}: Monthly Retraining Impact\n'
                     f'Purple lines = retraining ({retrain_stats["total_retrains"]} total)',
                     fontweight='bold', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        final_ppo = results['PPO_Cumulative_Profit'].iloc[-1]

        ax.text(0.02, 0.98,
                f'PPO: EUR {final_ppo:,.0f}\n'
                f'Retrains: {retrain_stats["total_retrains"]}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                fontsize=10, fontweight='bold')

        # Rolling average profit
        ax = axes[1]
        window = 24 * 7

        ppo_rolling = pd.Series(results['PPO_Instant_Profit']).rolling(window=window).mean()

        ax.plot(results['Hour'], ppo_rolling, label='PPO (7-day avg)',
                color='coral', linewidth=2, alpha=0.7)

        for retrain_time in retrain_stats['retrain_times']:
            ax.axvline(x=retrain_time, color='purple', linestyle='--', alpha=0.6, linewidth=2)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xlabel('Hour', fontweight='bold', fontsize=12)
        ax.set_ylabel('Profit/Hour (EUR, 7-day avg)', fontweight='bold', fontsize=12)
        ax.set_title('Rolling Average Profit', fontweight='bold', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filename_png, filename_pdf = save_plot_as_pdf(fig, f'monthly_retraining_{safe_name}', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        plt.close('all')


def plot_daily_detailed(results, noise_type, results_folder, n_days=5):
    """Detailed daily plot"""
    safe_name = noise_type.replace('ornstein-uhlenbeck', 'ou')

    try:
        print(f"  Generating daily detailed plot for {noise_type}...")

        results['Month'] = results['Date'].dt.month
        results['Day'] = results['Date'].dt.day

        unique_days = results.groupby(['Month', 'Day']).size().reset_index()[['Month', 'Day']]
        days_to_plot = min(n_days, len(unique_days))

        if days_to_plot == 0:
            return

        fig, axes = plt.subplots(days_to_plot, 2, figsize=(20, 4 * days_to_plot))
        if days_to_plot == 1:
            axes = axes.reshape(1, -1)

        for idx in range(days_to_plot):
            month = unique_days.iloc[idx]['Month']
            day = unique_days.iloc[idx]['Day']
            day_data = results[(results['Month'] == month) & (results['Day'] == day)].iloc[:24]

            if len(day_data) == 0:
                continue

            # Energy and price plot
            ax_energy = axes[idx, 0]
            hours = range(len(day_data))

            ax_price = ax_energy.twinx()
            ax_price.plot(hours, day_data['True_Price'], 'k-', linewidth=2.5,
                          label='True Price', alpha=0.9, zorder=5)
            ax_price.plot(hours, day_data['Forecast_Price'], 'k--', linewidth=1.5,
                          label='Forecast', alpha=0.6, zorder=4)

            ppo_charge_color = '#FF7F50'
            ppo_discharge_color = '#4169E1'

            for i, h in enumerate(hours):
                energy_ppo = day_data['PPO_Energy_Actual'].iloc[i]
                color_ppo = ppo_charge_color if energy_ppo > 0.01 else ppo_discharge_color if energy_ppo < -0.01 else 'lightgray'
                ax_energy.bar(h, energy_ppo, width=0.7,
                              color=color_ppo, alpha=0.8, edgecolor='black', linewidth=0.8)

            ax_energy.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax_energy.set_xlabel('Hour', fontweight='bold', fontsize=11)
            ax_energy.set_ylabel('Energy (MW)', fontweight='bold', fontsize=10)
            ax_price.set_ylabel('Price (EUR/MWh)', fontweight='bold', fontsize=11)

            date_str = day_data['Date'].iloc[0].strftime('%d %B %Y')
            ax_energy.set_title(f'{date_str} - {noise_type.upper()}',
                                fontweight='bold', fontsize=12)
            ax_energy.grid(True, alpha=0.3)
            ax_energy.set_xlim(-0.5, 23.5)
            ax_energy.set_xticks(range(0, 24, 2))
            ax_energy.set_ylim(-2.5, 2.5)

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=ppo_charge_color, alpha=0.8, edgecolor='black', label='PPO Charge'),
                Patch(facecolor=ppo_discharge_color, alpha=0.8, edgecolor='black', label='PPO Discharge'),
                plt.Line2D([0], [0], color='k', linewidth=2.5, label='True Price'),
                plt.Line2D([0], [0], color='k', linewidth=1.5, linestyle='--', label='Forecast')
            ]
            ax_energy.legend(handles=legend_elements, loc='upper left', fontsize=8, ncol=2)

            # SOC plot
            ax_soc = axes[idx, 1]

            ax_soc.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Max')
            ax_soc.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Min')

            for i, h in enumerate(hours):
                ax_soc.bar(h, day_data['PPO_SOC'].iloc[i] * 100, width=0.7,
                           color='#FF7F50', alpha=0.7, edgecolor='black', linewidth=0.8,
                           label='PPO' if i == 0 else '')

            ax_soc.set_xlabel('Hour', fontweight='bold', fontsize=11)
            ax_soc.set_ylabel('SOC (%)', fontweight='bold', fontsize=11)
            ax_soc.set_title(f'{date_str} - SOC', fontweight='bold', fontsize=12)
            ax_soc.grid(True, alpha=0.3)
            ax_soc.set_xlim(-0.5, 23.5)
            ax_soc.set_xticks(range(0, 24, 2))
            ax_soc.set_ylim(0, 100)
            ax_soc.legend(loc='upper right', fontsize=9)

        plt.suptitle(f'Daily Analysis - PPO - {noise_type.upper()}',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        filename_png, filename_pdf = save_plot_as_pdf(fig, f'daily_detailed_{safe_name}', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        plt.close('all')


def plot_monthly_comparison(results, noise_type, results_folder):
    """Monthly comparison plot"""
    safe_name = noise_type.replace('ornstein-uhlenbeck', 'ou')

    try:
        print(f"  Generating monthly comparison for {noise_type}...")

        results['Month'] = results['Date'].dt.month
        results['Day'] = results['Date'].dt.day

        months = results['Month'].unique()
        n_months = len(months)

        if n_months == 0:
            return

        fig, axes = plt.subplots(4, 3, figsize=(20, 16))
        axes = axes.flatten()

        for idx, month in enumerate(sorted(months)):
            if idx >= 12:
                break

            month_data = results[results['Month'] == month]
            mid_day = month_data['Day'].median()
            day_data = month_data[month_data['Day'] == int(mid_day)].iloc[:24]

            if len(day_data) == 0:
                day_data = month_data.iloc[:24]

            if len(day_data) == 0:
                continue

            ax = axes[idx]
            hours = range(len(day_data))
            ax2 = ax.twinx()

            ax.plot(hours, day_data['True_Price'], 'k-', linewidth=2, label='Price', alpha=0.8)
            ax.plot(hours, day_data['Forecast_Price'], 'k--', linewidth=1.5, label='Forecast', alpha=0.6)

            ax2.bar(hours, day_data['PPO_Energy_Actual'], width=0.6,
                    color='coral', alpha=0.6, label='PPO', edgecolor='black')

            ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

            ax.set_xlabel('Hour', fontweight='bold')
            ax.set_ylabel('Price (EUR/MWh)', fontweight='bold')
            ax2.set_ylabel('Energy (MW)', fontweight='bold')

            month_name = day_data['Date'].iloc[0].strftime('%B %Y')
            ax.set_title(f'{month_name}', fontweight='bold')

            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, 23.5)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

        for idx in range(n_months, 12):
            fig.delaxes(axes[idx])

        plt.suptitle(f'Monthly Comparison - PPO - {noise_type.upper()}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        filename_png, filename_pdf = save_plot_as_pdf(fig, f'monthly_comparison_{safe_name}', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        plt.close('all')


def plot_all_errors_comparison(all_results, summary_df, results_folder):
    """Comparison across all error types"""
    try:
        print(f"  Generating all errors comparison...")

        n_noise = len(NOISE_TYPES_TO_TEST)
        fig = plt.figure(figsize=(7 * min(n_noise, 3), 12))
        gs = fig.add_gridspec(3, min(n_noise, 3), hspace=0.3, wspace=0.3)

        noise_labels = {
            'uniform': 'UNIFORM',
            'normal': 'NORMAL',
            'ornstein-uhlenbeck': 'O-U',
            'fixed-bias': 'FIXED-BIAS'
        }

        for idx, noise_type in enumerate(NOISE_TYPES_TO_TEST[:3]):
            if noise_type not in all_results:
                continue

            results = all_results[noise_type]

            # Cumulative profits
            ax = fig.add_subplot(gs[0, idx])
            ax.plot(results['Hour'], results['PPO_Cumulative_Profit'],
                    label='PPO', color='coral', linewidth=2.5, alpha=0.8)
            ax.set_ylabel('Cumulative Profit (EUR)', fontweight='bold', fontsize=11)
            ax.set_title(noise_labels.get(noise_type, noise_type.upper()), fontweight='bold', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            final_ppo = results['PPO_Cumulative_Profit'].iloc[-1]
            ax.text(0.02, 0.98, f'Final: EUR {final_ppo:,.0f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    fontsize=9, fontweight='bold')

            # SOC
            ax = fig.add_subplot(gs[1, idx])
            sample_hours = min(168, len(results))
            ax.plot(results['Hour'][:sample_hours],
                    [s * 100 for s in results['PPO_SOC'][:sample_hours]],
                    label='PPO', color='coral', linewidth=1.5, alpha=0.7)
            ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylabel('SOC (%)', fontweight='bold', fontsize=11)
            ax.set_title('SOC (Week 1)', fontweight='bold', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 100])

            # Energy distribution
            ax = fig.add_subplot(gs[2, idx])
            ax.hist(results['PPO_Energy_Actual'], bins=50, color='coral',
                    alpha=0.7, edgecolor='black', label='PPO', density=True)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
            ax.set_xlabel('Energy (MW)', fontweight='bold', fontsize=11)
            ax.set_ylabel('Density', fontweight='bold', fontsize=11)
            ax.set_title('Energy Distribution', fontweight='bold', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('PPO Performance Across Error Distributions + Monthly Retraining',
                     fontsize=15, fontweight='bold', y=0.995)

        filename_png, filename_pdf = save_plot_as_pdf(fig, 'comparison_all_errors', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        plt.close('all')


def plot_summary_bar(summary_df, results_folder):
    """Summary bar chart"""
    try:
        print(f"  Generating summary bar chart...")

        noise_map = {
            'uniform': 'Uniform',
            'normal': 'Normal',
            'ornstein-uhlenbeck': 'O-U',
            'fixed-bias': 'Fixed-Bias'
        }
        noise_labels = [noise_map.get(n, n) for n in summary_df['Noise_Type']]

        x = np.arange(len(summary_df))

        fig, ax = plt.subplots(figsize=(12, 8))

        ppo_profits = summary_df['PPO_Profit'].values

        bars = ax.bar(x, ppo_profits, color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Annual Profit (EUR)', fontweight='bold', fontsize=13)
        ax.set_title('PPO Profits by Error Distribution', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(noise_labels, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:,.0f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')

        plt.tight_layout()

        filename_png, filename_pdf = save_plot_as_pdf(fig, 'summary_all_errors', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        plt.close('all')


def plot_soh_summary(all_results, summary_df, results_folder):
    """SOH summary across all error types"""
    try:
        print(f"  Generating SOH summary across error types...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        noise_labels = {
            'uniform': 'UNIFORM',
            'normal': 'NORMAL',
            'ornstein-uhlenbeck': 'O-U',
            'fixed-bias': 'FIXED-BIAS'
        }

        # Final SOH bar chart
        final_soh_data = []

        for noise_type in NOISE_TYPES_TO_TEST:
            if noise_type not in all_results:
                continue

            results = all_results[noise_type]

            if 'PPO_SOH' not in results.columns:
                ppo_throughput_cumulative = results['PPO_Energy_Actual'].abs().cumsum()
                ppo_cycles = (ppo_throughput_cumulative * 1000) / (2 * BATTERY_CAPACITY * 1000)
                results['PPO_SOH'] = ppo_cycles.apply(degradation)

            final_soh_data.append({
                'noise_type': noise_type,
                'ppo_final_soh': results['PPO_SOH'].iloc[-1]
            })

        ax = axes[0]
        x = np.arange(len(final_soh_data))

        ppo_sohs = [d['ppo_final_soh'] for d in final_soh_data]
        labels = [noise_labels.get(d['noise_type'], d['noise_type']) for d in final_soh_data]

        bars = ax.bar(x, ppo_sohs, color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)

        ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, linewidth=2, label='EOL (80%)')
        ax.set_ylabel('Final SOH (%)', fontweight='bold', fontsize=12)
        ax.set_title('Final State of Health by Error Type', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([75, 101])

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        # SOH evolution for all error types
        ax = axes[1]
        for noise_type in NOISE_TYPES_TO_TEST:
            if noise_type in all_results:
                results = all_results[noise_type]
                sample_indices = range(0, len(results), 24)
                ax.plot(results['Hour'].iloc[sample_indices],
                        results['PPO_SOH'].iloc[sample_indices],
                        label=noise_labels.get(noise_type, noise_type),
                        linewidth=2, alpha=0.8)

        ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.set_xlabel('Hour', fontweight='bold', fontsize=11)
        ax.set_ylabel('SOH (%)', fontweight='bold', fontsize=11)
        ax.set_title('PPO: SOH Evolution by Error Type', fontweight='bold', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([75, 101])

        plt.suptitle('State of Health Analysis - PPO Only',
                     fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        filename_png, filename_pdf = save_plot_as_pdf(fig, 'soh_summary_all_errors', results_folder)

        plt.close()
        print(f"  Saved (PNG): {filename_png}")
        print(f"  Saved (PDF): {filename_pdf}")

    except Exception as e:
        print(f"  ERROR generating SOH summary: {e}")
        traceback.print_exc()
        plt.close('all')


def main():
    print("=" * 80)
    print(" " * 5 + "PPO ONLY - BESS OPTIMIZATION")
    print(" " * 5 + "Multiple Error Distributions + Monthly Retraining + PDF Export")
    print("=" * 80)

    print(f"\nSETUP:")
    print(f"   Features:         26")
    print(f"   Network:          {PPO_HIDDEN_DIM_1}-{PPO_HIDDEN_DIM_2}")
    print(f"   Actions:          {N_DISCRETE_ACTIONS}")
    print(f"   Timesteps:        {PPO_TOTAL_TIMESTEPS:,}")
    print(f"   Forecast error:   ±{FORECAST_ERROR_MAX * 100:.0f}%")
    print(f"   Degradation cost: {DEGRADATION_COST_PER_MWH} EUR/MWh")

    print(f"\n   Error types:      {len(NOISE_TYPES_TO_TEST)}")
    for nt in NOISE_TYPES_TO_TEST:
        print(f"      - {nt}")

    print(f"\n   Monthly Retraining: {'ENABLED' if MONTHLY_RETRAINING_ENABLED else 'DISABLED'}")
    if MONTHLY_RETRAINING_ENABLED:
        print(f"      Frequency:     Every {MONTHLY_RETRAINING_FREQUENCY} days")
        print(f"      Lookback:      {MONTHLY_RETRAINING_LOOKBACK} days")
        print(f"      Timesteps:     {MONTHLY_RETRAINING_TIMESTEPS:,}")

    print(f"\n   Export Formats:")
    print(f"      PNG:           300 DPI")
    print(f"      PDF:           500 DPI")

    data_folder = 'data'
    train_path = os.path.join(data_folder, TRAIN_FILE)
    test_path = os.path.join(data_folder, TEST_FILE)

    print(f"\nLoading data...")
    try:
        df_train = pd.read_excel(train_path)
        if 'EUR/MWh' not in df_train.columns and '€/MWh' in df_train.columns:
            df_train.rename(columns={'€/MWh': 'EUR/MWh'}, inplace=True)
        if df_train['EUR/MWh'].dtype == 'object':
            df_train['EUR/MWh'] = df_train['EUR/MWh'].astype(str).str.replace(',', '.').astype(float)
        print(f"  Training: {len(df_train)} rows ({len(df_train) / 24:.0f} days)")

        df_test = pd.read_excel(test_path)
        if 'EUR/MWh' not in df_test.columns and '€/MWh' in df_test.columns:
            df_test.rename(columns={'€/MWh': 'EUR/MWh'}, inplace=True)
        if df_test['EUR/MWh'].dtype == 'object':
            df_test['EUR/MWh'] = df_test['EUR/MWh'].astype(str).str.replace(',', '.').astype(float)

        if TEST_FIRST_MONTH_ONLY:
            df_test = df_test.iloc[:24 * 31]
            print(f"  Test: {len(df_test)} rows ({len(df_test) / 24:.0f} days)")
        else:
            print(f"  Test: {len(df_test)} rows ({len(df_test) / 24:.0f} days)")

        all_train_test_prices = np.concatenate([df_train['EUR/MWh'].values, df_test['EUR/MWh'].values])
        print(f"  Combined: {len(all_train_test_prices)} hours")

    except Exception as e:
        print(f"  Error: {e}")
        return

    all_results = {}
    summary_data = []

    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Generate error types comparison
    print("\n" + "=" * 80)
    print("GENERATING ERROR TYPES COMPARISON")
    print("=" * 80)
    plot_error_types_comparison(results_folder)

    print("\n" + "=" * 80)
    print("TESTING ALL ERROR DISTRIBUTIONS")
    print("=" * 80)

    for noise_idx, noise_type in enumerate(NOISE_TYPES_TO_TEST):
        print("\n" + "+" + "-" * 78 + "+")
        print(f"|{' ' * 10}ERROR TYPE {noise_idx + 1}/{len(NOISE_TYPES_TO_TEST)}: {noise_type.upper():^40}|")
        print("+" + "-" * 78 + "+")

        error_descriptions = {
            'uniform': 'Uniformly distributed random errors in [-15%, +15%]',
            'normal': 'Normally distributed errors (Gaussian) σ=5%, clipped ±15%',
            'ornstein-uhlenbeck': 'Time-correlated errors using Ornstein-Uhlenbeck process',
            'fixed-bias': 'Deterministic fixed bias +10% (always overestimate)'
        }

        if noise_type in error_descriptions:
            print(f"\n  Error: {error_descriptions[noise_type]}")

        try:
            model_path = os.path.join(results_folder, f'ppo_only_{noise_type}')

            if os.path.exists(model_path + '.zip') and not FORCE_RETRAIN:
                print(f"\n  Found existing model")
                error_gen_dummy = ForecastErrorGenerator(noise_type)
                env_dummy = BatteryTradingEnvClean(df_train['EUR/MWh'].values, error_gen_dummy)
                ppo_model = PPO.load(model_path, env=env_dummy)
                print("  Model loaded")
            else:
                print(f"\n  Training NEW model...")
                ppo_model = train_ppo(df_train, noise_type, model_path)

            print(f"\n  Testing {noise_type.upper()}...")
            print("-" * 78)

            error_gen_test = ForecastErrorGenerator(noise_type)
            error_gen_test.reset()

            start_time = datetime.now()
            results_df, profit_ppo, retrain_stats, degradation_info = test_ppo(
                ppo_model, df_test, all_train_test_prices, noise_type, error_gen_test
            )
            end_time = datetime.now()

            all_results[noise_type] = results_df

            results_path = os.path.join(results_folder, f'results_{noise_type}.csv')
            results_df.to_csv(results_path, index=False)
            print(f"\n  Results saved: {results_path}")

            analyze_trading_behavior(results_df, noise_type)

            # Generate plots
            print(f"\n  Generating plots...")
            plot_daily_detailed(results_df, noise_type, results_folder, n_days=5)
            plot_monthly_comparison(results_df, noise_type, results_folder)
            plot_heatmaps(results_df, noise_type, results_folder)
            plot_soh_evolution(results_df, noise_type, results_folder)

            if MONTHLY_RETRAINING_ENABLED:
                plot_monthly_retraining_performance(results_df, retrain_stats, noise_type, results_folder)

            print(f"\n{'-' * 78}")
            print(f"FINAL RESULTS - {noise_type.upper()}")
            print(f"{'-' * 78}")
            print(f"\nOPERATIONAL PROFITS (Real Revenue):")
            print(f"  PPO:        EUR {profit_ppo:>15,.2f}")

            print(f"\nDEGRADATION COSTS (Tracking Only):")
            print(f"  PPO:        EUR {degradation_info['ppo_total_degradation']:>15,.2f}")

            print(f"\nNET PROFIT (Operational - Degradation):")
            net_ppo = profit_ppo - degradation_info['ppo_total_degradation']
            print(f"  PPO:        EUR {net_ppo:>15,.2f}")

            if MONTHLY_RETRAINING_ENABLED and retrain_stats['total_retrains'] > 0:
                print(f"\n  Monthly retrains: {retrain_stats['total_retrains']}")

            print(f"\n  Time: {(end_time - start_time).total_seconds():.1f}s")
            print(f"{'-' * 78}")

            summary_data.append({
                'Noise_Type': noise_type,
                'PPO_Profit': profit_ppo,
                'PPO_Degradation': degradation_info['ppo_total_degradation'],
                'PPO_Net_Profit': net_ppo,
                'Monthly_Retrains': retrain_stats['total_retrains'] if MONTHLY_RETRAINING_ENABLED else 0
            })

        except KeyboardInterrupt:
            print("\n\nInterrupted")
            break
        except Exception as e:
            print(f"\n\nError: {e}")
            traceback.print_exc()
            continue

    # Final summary
    if len(summary_data) > 0:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(results_folder, 'summary_ppo_only.csv')
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 80)
        print("FINAL SUMMARY - PPO ONLY")
        print("=" * 80)

        print(
            f"\n{'Error Type':<25} {'PPO Profit (EUR)':<20} {'Degradation (EUR)':<20} {'Net Profit (EUR)':<20} {'Retrains':<10}")
        print("-" * 95)

        for _, row in summary_df.iterrows():
            retrains_str = f"{int(row['Monthly_Retrains'])}" if MONTHLY_RETRAINING_ENABLED else "N/A"
            print(f"{row['Noise_Type']:<25} "
                  f"EUR {row['PPO_Profit']:>15,.0f} "
                  f"EUR {row['PPO_Degradation']:>15,.0f} "
                  f"EUR {row['PPO_Net_Profit']:>15,.0f} "
                  f"{retrains_str:<10}")

        print("-" * 95)

        avg_profit = summary_df['PPO_Profit'].mean()
        avg_degradation = summary_df['PPO_Degradation'].mean()
        avg_net = summary_df['PPO_Net_Profit'].mean()

        print(f"\nAVERAGE:")
        print(f"   PPO Profit:       EUR {avg_profit:,.2f}")
        print(f"   Degradation:      EUR {avg_degradation:,.2f}")
        print(f"   Net Profit:       EUR {avg_net:,.2f}")

        if len(summary_df) > 1:
            best_error = summary_df.loc[summary_df['PPO_Profit'].idxmax()]
            worst_error = summary_df.loc[summary_df['PPO_Profit'].idxmin()]

            print(f"\nPERFORMANCE:")
            print(f"   Best:  {best_error['Noise_Type']} (EUR {best_error['PPO_Profit']:,.0f})")
            print(f"   Worst: {worst_error['Noise_Type']} (EUR {worst_error['PPO_Profit']:,.0f})")
            print(f"   Std:   EUR {summary_df['PPO_Profit'].std():,.0f}")

        print("\n" + "=" * 80)

        # Generate comparison plots
        if len(all_results) > 0:
            print(f"\nGenerating comparison plots...")
            plot_all_errors_comparison(all_results, summary_df, results_folder)
            plot_summary_bar(summary_df, results_folder)
            plot_soh_summary(all_results, summary_df, results_folder)
            print(f"  All plots generated!")

    print("\n" + "=" * 80)
    print("COMPLETE - PPO ONLY")
    print("=" * 80)

    print(f"\nFiles in: {results_folder}/")
    print(f"\nAll plots saved in:")
    print(f"   - PNG (300 DPI)")
    print(f"   - PDF (500 DPI)")

    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"\nDevice: {device_info}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)



if __name__ == "__main__":
    main()