#!/usr/bin/env python3
"""
Main entry point for the PPO vs MILP BESS comparison pipeline.

Pipeline stages:
    1. Setup            -> CLI parsing, seeding, output directory layout
    2. Market data      -> Generate synthetic PUN prices and ARERA flexibility services
                           for pretraining + annual horizon
    3. PPO pretraining  -> Train a PPO agent on the pretraining window (skippable)
    4. Comparison       -> Run daily MILP optimization and PPO simulation in lockstep
                           over the annual horizon, with monthly PPO retraining
    5. Reporting        -> Aggregate metrics, save CSV/JSON, produce summary plots

Usage:
    # Full pipeline (training + 1-year comparison, 6-12 hours on consumer CPU):
    python main.py --mode full

    # Skip training, reuse an existing model:
    python main.py --mode compare --model-path ppo_models/main_ppo.zip

    # Quick smoke test (7 days, 50k training steps):
    python main.py --mode fast

    # Only run PPO pretraining:
    python main.py --mode train

Author: Lorenzo Giannuzzo
Reviewed/fixed: 2026-05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib in non-interactive mode for headless execution
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Local imports (fixed versions in this folder)
from milp_optimizer import BatteryParameters, MILPOptimizer
from flexibility_market import FlexibilityMarket, FlexibilityService, ServiceType
from extended_ppo_environment import ExtendedBatteryTradingEnv
from drl_flexibility_analysis import ForecastErrorGenerator
from plotting import generate_all_plots

# Optional PPO dependency: handled gracefully
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False

# Optional tensorboard dependency (used only for training logs)
try:
    import tensorboard  # noqa: F401

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


# ============================================================================
# CLI configuration
# ============================================================================


@dataclass
class PipelineConfig:
    """Pipeline-wide configuration."""

    mode: str = "full"  # full | train | compare | fast
    days: int = 365
    pretrain_months: int = 2
    retrain_frequency_months: int = 1

    # Forecast error distributions:
    #   - train_error_distribution: the single distribution used to generate
    #     forecast errors during PPO pretraining and monthly retraining
    #   - eval_error_distributions: the list of distributions over which the
    #     trained agent and the MILP are benchmarked. The agent is trained
    #     once on train_error_distribution and evaluated on each entry of
    #     eval_error_distributions independently (train-once, eval-on-many).
    train_error_distribution: str = "normal"
    eval_error_distributions: List[str] = field(
        default_factory=lambda: ["uniform", "normal", "ornstein-uhlenbeck", "fixed-bias"]
    )

    seed: int = 42

    # PPO training hyperparameters (Fix 3: increased training budget)
    pretrain_timesteps: int = 1_000_000   # was 500k; price sensitivity needs more samples
    retrain_timesteps: int = 200_000      # was 100k; same reason
    learning_rate: float = 3e-4
    ent_coef: float = 0.05         # Fix 1: increased from 0.01 to escape mode collapse

    # Battery parameters
    battery_capacity_mwh: float = 4.0
    battery_max_power_mw: float = 2.0
    battery_efficiency: float = 0.95
    cycle_life: int = 6000
    degradation_cost_per_mwh: float = 25000.0

    # I/O
    output_root: str = "results"
    model_path: str = "ppo_models/main_ppo.zip"
    run_name: str = field(default_factory=lambda: datetime.now().strftime("run_%Y%m%d_%H%M%S"))

    # Behavior toggles
    skip_train: bool = False
    enable_reward_shaping: bool = True   # Fix 2: ON during training, OFF in eval
    flex_price_error_pct: float = 0.05   # Opt B: forecast error on FCR/aFRR/mFRR tariffs (5% default)
    verbose: bool = True


def parse_args() -> PipelineConfig:
    """Parse command-line arguments into a PipelineConfig."""
    p = argparse.ArgumentParser(
        description="PPO vs MILP BESS comparison pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["full", "train", "compare", "fast"], default="full")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--pretrain-months", type=int, default=2)
    p.add_argument("--retrain-frequency-months", type=int, default=1)
    p.add_argument(
        "--train-dist",
        choices=["uniform", "normal", "ornstein-uhlenbeck", "fixed-bias"],
        default="normal",
        dest="train_error_distribution",
        help="Forecast-error distribution used during PPO pretraining and monthly retraining.",
    )
    p.add_argument(
        "--eval-dists",
        nargs="+",
        choices=["uniform", "normal", "ornstein-uhlenbeck", "fixed-bias"],
        default=["uniform", "normal", "ornstein-uhlenbeck", "fixed-bias"],
        dest="eval_error_distributions",
        help="One or more forecast-error distributions over which the trained agent "
             "and the MILP are benchmarked. Pass several space-separated names.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrain-timesteps", type=int, default=1_000_000)
    p.add_argument("--retrain-timesteps", type=int, default=200_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.05,
                   help="PPO entropy coefficient (Fix 1: default 0.05 to avoid mode collapse)")
    p.add_argument("--output-root", default="results")
    p.add_argument("--model-path", default="ppo_models/main_ppo.zip")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--no-reward-shaping", action="store_true",
                   help="Disable Fix-2 price-sensitivity shaping during training "
                        "(ablation flag; default ON)")
    p.add_argument("--flex-price-error", type=float, default=0.05,
                   help="Half-width of the uniform forecast error on FCR/aFRR/mFRR "
                        "tariffs (default 0.05 = +/-5%%). Set to 0 to recover the "
                        "deterministic-flexibility-tariff regime (ablation).")
    p.add_argument("--quiet", action="store_true")

    args = p.parse_args()
    cfg = PipelineConfig(
        mode=args.mode,
        days=args.days,
        pretrain_months=args.pretrain_months,
        retrain_frequency_months=args.retrain_frequency_months,
        train_error_distribution=args.train_error_distribution,
        eval_error_distributions=args.eval_error_distributions,
        seed=args.seed,
        pretrain_timesteps=args.pretrain_timesteps,
        retrain_timesteps=args.retrain_timesteps,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        output_root=args.output_root,
        model_path=args.model_path,
        skip_train=args.skip_train,
        enable_reward_shaping=not args.no_reward_shaping,
        flex_price_error_pct=args.flex_price_error,
        verbose=not args.quiet,
    )

    # Mode-specific overrides
    if cfg.mode == "fast":
        cfg.days = 7
        cfg.pretrain_months = 0           # no pretraining in fast mode
        cfg.pretrain_timesteps = 50_000
        cfg.retrain_timesteps = 0         # no retraining
        # Fast mode evaluates only one distribution to stay quick
        cfg.eval_error_distributions = [cfg.train_error_distribution]
    if cfg.mode == "compare":
        cfg.skip_train = True

    return cfg


# ============================================================================
# Output directory layout
# ============================================================================


@dataclass
class OutputPaths:
    """All output paths used by the pipeline, organized into named subdirectories."""

    root: Path
    data: Path
    models: Path
    comparison: Path
    logs: Path
    figures: Path

    @classmethod
    def create(cls, output_root: str, run_name: str) -> "OutputPaths":
        root = Path(output_root) / run_name
        paths = cls(
            root=root,
            data=root / "data",
            models=root / "models",
            comparison=root / "comparison",
            logs=root / "logs",
            figures=root / "figures",
        )
        for p in [paths.root, paths.data, paths.models, paths.comparison, paths.logs, paths.figures]:
            p.mkdir(parents=True, exist_ok=True)
        return paths


# ============================================================================
# Stage 1: Market data generation
# ============================================================================


def generate_synthetic_prices(
    n_hours: int,
    seed: int = 42,
    base_level: float = 80.0,
    daily_amplitude: float = 30.0,
    seasonal_amplitude: float = 25.0,
    noise_std: float = 8.0,
    weekend_factor: float = 0.9,
) -> np.ndarray:
    """Generate a deterministic synthetic PUN-like price series.

    Components:
        - Daily sinusoidal pattern (peak around 18:00-20:00, trough around 04:00)
        - Annual sinusoidal pattern (winter peak, summer trough)
        - Weekend discount factor
        - Gaussian noise with fixed seed

    The series is in EUR/MWh and floored at 10 EUR/MWh.
    """
    rng = np.random.default_rng(seed)
    prices = np.zeros(n_hours, dtype=np.float64)
    for t in range(n_hours):
        hour = t % 24
        day = t // 24
        day_of_year = day % 365
        day_of_week = day % 7

        # Daily pattern: peak in evening, trough at night
        daily = daily_amplitude * np.sin(2.0 * np.pi * (hour - 6) / 24.0)
        # Annual pattern: higher in winter
        seasonal = seasonal_amplitude * np.cos(2.0 * np.pi * day_of_year / 365.0)
        # Weekend discount
        wkn = weekend_factor if day_of_week >= 5 else 1.0
        # Noise
        eps = rng.normal(0.0, noise_std)

        prices[t] = max(10.0, (base_level + daily + seasonal) * wkn + eps)
    return prices


def generate_flexibility_services(
    n_hours: int,
    flex_market: FlexibilityMarket,
) -> List[List[FlexibilityService]]:
    """Generate per-hour flexibility service opportunities using the market model."""
    services: List[List[FlexibilityService]] = []
    for t in range(n_hours):
        hour = t % 24
        day_of_week = (t // 24) % 7
        services.append(flex_market.get_flexibility_opportunities(hour, day_of_week))
    return services


def build_market_data(cfg: PipelineConfig, paths: OutputPaths) -> Dict[str, Any]:
    """Generate the full synthetic dataset and persist it to disk."""
    pretrain_hours = cfg.pretrain_months * 30 * 24
    annual_hours = cfg.days * 24
    total_hours = pretrain_hours + annual_hours

    if cfg.verbose:
        print(f"  Generating {total_hours} hours of synthetic prices ({pretrain_hours} pretrain + {annual_hours} annual)")

    prices = generate_synthetic_prices(total_hours, seed=cfg.seed)
    flex_market = FlexibilityMarket(random_seed=cfg.seed,
                                    flex_price_error_pct=cfg.flex_price_error_pct)
    services = generate_flexibility_services(total_hours, flex_market)

    # Persist prices to CSV for inspection / reproducibility
    df = pd.DataFrame(
        {
            "hour": np.arange(total_hours),
            "phase": ["pretrain"] * pretrain_hours + ["annual"] * annual_hours,
            "price_eur_mwh": prices,
        }
    )
    df.to_csv(paths.data / "synthetic_prices.csv", index=False)

    return {
        "pretrain_prices": prices[:pretrain_hours].tolist(),
        "pretrain_services": services[:pretrain_hours],
        "annual_prices": prices[pretrain_hours:].tolist(),
        "annual_services": services[pretrain_hours:],
        "flex_market": flex_market,
    }


# ============================================================================
# Stage 2: PPO training
# ============================================================================


def make_env(
    prices: List[float],
    error_distribution: str,
    seed: int,
    enable_reward_shaping: bool = True,
    flex_price_error_pct: float = 0.05,
) -> ExtendedBatteryTradingEnv:
    """Construct an ExtendedBatteryTradingEnv with a forecast error generator.

    Args:
        enable_reward_shaping: if True, the env adds the Fix-2 intrinsic
            bonus during step() to teach price sensitivity. Set to False at
            evaluation time so that the realized reward equals the economic
            net profit (the metric compared to the MILP).
        flex_price_error_pct: half-width of the uniform forecast error on
            FCR/aFRR/mFRR tariffs. Default 5%. See FlexibilityMarket docstring.
    """
    error_gen = ForecastErrorGenerator(error_distribution)
    return ExtendedBatteryTradingEnv(
        prices=prices,
        error_generator=error_gen,
        flexibility_enabled=True,
        conflict_strategy="equal_priority",  # Fix B: avoid zeroing arbitrage when flex saturates
        random_seed=seed,
        enable_reward_shaping=enable_reward_shaping,
        flex_price_error_pct=flex_price_error_pct,
    )


def train_ppo(
    cfg: PipelineConfig,
    paths: OutputPaths,
    pretrain_prices: List[float],
) -> Optional[str]:
    """Pretrain a PPO agent on the pretraining window. Returns the saved model path."""
    if not PPO_AVAILABLE:
        print("WARNING: stable-baselines3 is not installed. Skipping PPO training.")
        return None
    if not pretrain_prices:
        print("WARNING: no pretraining data provided. Skipping PPO training.")
        return None

    print(f"  Pretraining PPO for {cfg.pretrain_timesteps:,} timesteps "
          f"(error dist: {cfg.train_error_distribution})...")
    env = make_env(pretrain_prices, cfg.train_error_distribution, cfg.seed,
                   enable_reward_shaping=cfg.enable_reward_shaping,
                   flex_price_error_pct=cfg.flex_price_error_pct)
    vec_env = DummyVecEnv([lambda: env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=cfg.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=cfg.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1 if cfg.verbose else 0,
        tensorboard_log=str(paths.logs / "tensorboard") if TENSORBOARD_AVAILABLE else None,
        seed=cfg.seed,
    )

    t0 = time.time()
    model.learn(total_timesteps=cfg.pretrain_timesteps)
    train_time = time.time() - t0

    # Save both to the run directory and to the user-specified model_path
    run_model_path = paths.models / "main_ppo_pretrained.zip"
    model.save(str(run_model_path))
    Path(cfg.model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(cfg.model_path)

    print(f"  Pretraining done in {train_time/60:.1f} minutes. Model saved to {run_model_path}")
    return str(run_model_path)


def retrain_ppo(
    model: "PPO",
    cfg: PipelineConfig,
    paths: OutputPaths,
    recent_prices: List[float],
    month_idx: int,
) -> None:
    """Continue PPO training on the most recent window (monthly transfer learning)."""
    if cfg.retrain_timesteps <= 0 or not recent_prices:
        return

    print(f"  Monthly retraining for month {month_idx} ({cfg.retrain_timesteps:,} timesteps)...")
    env = make_env(recent_prices, cfg.train_error_distribution, cfg.seed + month_idx,
                   enable_reward_shaping=cfg.enable_reward_shaping,
                   flex_price_error_pct=cfg.flex_price_error_pct)
    vec_env = DummyVecEnv([lambda: env])
    model.set_env(vec_env)
    model.learn(total_timesteps=cfg.retrain_timesteps, reset_num_timesteps=False)

    model_path = paths.models / f"main_ppo_month_{month_idx:02d}.zip"
    model.save(str(model_path))


# ============================================================================
# Stage 3: Annual comparison
# ============================================================================


def build_battery_params(cfg: PipelineConfig) -> BatteryParameters:
    """Construct BatteryParameters consistent with the PPO env constants."""
    return BatteryParameters(
        capacity_mwh=cfg.battery_capacity_mwh,
        max_power_mw=cfg.battery_max_power_mw,
        efficiency=cfg.battery_efficiency,
        soc_min=0.1,
        soc_max=0.9,
        initial_soc=0.5,
        degradation_cost_per_mwh=cfg.degradation_cost_per_mwh,
        cycle_life=cfg.cycle_life,
    )


def compute_milp_service_breakdown(
    flex_reservations: Dict[str, List[float]],
    daily_services: List[List[FlexibilityService]],
) -> Dict[str, float]:
    """Compute MILP capacity revenue per service from reservations and prices.

    The MILP OptimizationResult only reports aggregate flexibility_revenue, but
    for plots we need the per-service breakdown. We multiply the hourly reserved
    capacity by the corresponding service's capacity_price (and approximate the
    expected energy revenue with the activation probability factor as in the
    objective function).
    """
    out = {"FCR": 0.0, "aFRR": 0.0, "mFRR": 0.0}
    n_hours = len(daily_services)
    for t in range(n_hours):
        services_t = {s.service_type.value: s for s in daily_services[t]}
        for key in ("FCR", "aFRR", "mFRR"):
            r = flex_reservations.get(key, [0.0] * n_hours)[t]
            if r <= 0 or key not in services_t:
                continue
            svc = services_t[key]
            # Capacity revenue + expected energy revenue (0.5 activation factor)
            out[key] += r * svc.capacity_price
            out[key] += 0.5 * r * svc.energy_price * svc.activation_probability
    return out


def run_milp_day(
    battery_params: BatteryParameters,
    flex_market: FlexibilityMarket,
    daily_prices: List[float],
    daily_services: List[List[FlexibilityService]],
    forecast_errors: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Run a 24-hour MILP optimization.

    If `forecast_errors` is provided, the prices fed to the MILP are
    `daily_prices * (1 + forecast_errors)`, simulating an imperfect-foresight
    MILP that sees the same kind of noisy forecast that the PPO observes.
    The realized profit is then computed against the TRUE daily_prices,
    consistently with how the PPO environment computes its realized reward.
    Without `forecast_errors`, the MILP runs in perfect-foresight mode.
    """
    if forecast_errors is not None and len(forecast_errors) == len(daily_prices):
        forecast_prices = [p * (1.0 + e) for p, e in zip(daily_prices, forecast_errors)]
    else:
        forecast_prices = list(daily_prices)

    optimizer = MILPOptimizer(battery_params, flex_market)
    t0 = time.time()
    result = optimizer.optimize(
        time_horizon=len(forecast_prices),
        energy_prices=forecast_prices,
        flexibility_services=daily_services,
    )
    elapsed = time.time() - t0

    # If a forecast error was applied, the MILP optimized against noisy prices
    # but the realized arbitrage profit/throughput should be evaluated against
    # the true prices. We recompute the realized arbitrage from the planned
    # power_trajectory and the true daily_prices.
    if forecast_errors is not None:
        realized_arbitrage = sum(
            p * tp for p, tp in zip(result.power_trajectory, daily_prices)
        )
        # power_trajectory in OptimizationResult is signed: P_discharge - P_charge.
        # Revenue from selling minus cost of buying = power * price.
        # We then need to subtract degradation cost (already computed correctly
        # since degradation depends on |throughput|, not on price).
        realized_net_profit = (realized_arbitrage
                               + result.flexibility_revenue
                               - result.degradation_cost)
        arbitrage_out = realized_arbitrage
        net_profit_out = realized_net_profit
    else:
        arbitrage_out = result.arbitrage_profit
        net_profit_out = result.net_profit

    return {
        "algorithm": "MILP",
        "net_profit": net_profit_out,
        "arbitrage_profit": arbitrage_out,
        "flexibility_revenue": result.flexibility_revenue,
        "degradation_cost": result.degradation_cost,
        "execution_time": elapsed,
        "battery_utilization": result.battery_utilization,
        "solver_status": result.solver_status,
        "soc_trajectory": result.soc_trajectory,
        "power_trajectory": result.power_trajectory,
        "flexibility_reservations": result.flexibility_reservations,
    }


def run_ppo_day(
    ppo_model: Optional["PPO"],
    daily_prices: List[float],
    daily_services: List[List[FlexibilityService]],
    cfg: PipelineConfig,
    eval_distribution: str,
) -> Dict[str, Any]:
    """Run a 24-hour PPO simulation using the trained agent if available.

    The environment uses `eval_distribution` for forecast-error generation,
    which may differ from the distribution used during training. This is the
    setup that probes the agent's out-of-distribution robustness.

    Reward shaping is DISABLED in evaluation so that the reported reward
    equals the realized economic profit (consistent with MILP).
    """
    env = make_env(daily_prices, eval_distribution, cfg.seed,
                   enable_reward_shaping=False,
                   flex_price_error_pct=cfg.flex_price_error_pct)
    obs, _ = env.reset()
    t0 = time.time()

    total_arb = total_flex = total_deg = 0.0
    fcr_rev = afrr_rev = mfrr_rev = 0.0
    soc_traj = [env.battery.soc]
    power_traj: List[float] = []
    flex_res = {"FCR": [], "aFRR": [], "mFRR": []}
    # Fix A tracking: separate arbitrage / activation throughput
    activation_throughput_total = 0.0
    arbitrage_throughput_total = 0.0
    # Diagnostic tracking: action distribution
    actions_log: List[List[int]] = []

    for _ in range(len(daily_prices)):
        if ppo_model is not None:
            action, _ = ppo_model.predict(obs, deterministic=True)
            # Stable-baselines3 returns the vector directly for MultiDiscrete spaces
            if np.ndim(action) == 0:
                action = int(action)
        else:
            # Greedy fallback (MultiDiscrete vector): charge cheap, discharge expensive,
            # plus 50% aFRR by default
            cur_price = obs[25] * 200.0 if len(obs) > 25 else 0.0
            arb_idx = 15 if cur_price < 60.0 else 5
            action = np.array([arb_idx, 0, 1, 0], dtype=int)

        obs, _reward, done, truncated, info = env.step(action)
        total_arb += info.get("arbitrage_profit", 0.0)
        total_flex += info.get("flexibility_revenue", 0.0)
        total_deg += info.get("degradation_cost", 0.0)
        arbitrage_throughput_total += info.get("arbitrage_throughput_mwh", 0.0)
        activation_throughput_total += info.get("activation_throughput_mwh", 0.0)
        actions_log.append(info.get("action_raw", [10, 0, 0, 0]))
        breakdown = info.get("flexibility_breakdown", {})
        fcr_rev += breakdown.get("fcr_revenue", 0.0)
        afrr_rev += breakdown.get("afrr_revenue", 0.0)
        mfrr_rev += breakdown.get("mfrr_revenue", 0.0)
        soc_traj.append(info.get("soc", env.battery.soc))
        power_traj.append(info.get("arbitrage_energy", 0.0))
        for k in flex_res:
            flex_res[k].append(info.get(f"reserved_{k.lower()}", 0.0))
        if done or truncated:
            break

    elapsed = time.time() - t0
    throughput = sum(abs(p) for p in power_traj)
    max_th = cfg.battery_max_power_mw * len(power_traj)
    utilization = throughput / max_th if max_th > 0 else 0.0

    return {
        "algorithm": "PPO" if ppo_model is not None else "PPO-fallback-greedy",
        "net_profit": total_arb + total_flex - total_deg,
        "arbitrage_profit": total_arb,
        "flexibility_revenue": total_flex,
        "degradation_cost": total_deg,
        "fcr_revenue": fcr_rev,
        "afrr_revenue": afrr_rev,
        "mfrr_revenue": mfrr_rev,
        "execution_time": elapsed,
        "battery_utilization": utilization,
        "soc_trajectory": soc_traj,
        "power_trajectory": power_traj,
        "flexibility_reservations": flex_res,
        # New tracking (Fix A + diagnostics)
        "arbitrage_throughput_mwh": arbitrage_throughput_total,
        "activation_throughput_mwh": activation_throughput_total,
        "total_throughput_mwh": arbitrage_throughput_total + activation_throughput_total,
        "actions_log": actions_log,
    }


def run_annual_comparison(
    cfg: PipelineConfig,
    paths: OutputPaths,
    market: Dict[str, Any],
    ppo_model: Optional["PPO"],
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Daily MILP vs PPO comparison over the annual horizon, looped across
    every entry of `cfg.eval_error_distributions`.

    Methodological setup (train-once, eval-on-many):
        - The PPO agent has been pretrained on `cfg.train_error_distribution`
          and is reloaded fresh at the start of each evaluation distribution
          (so that monthly retraining done in a previous distribution does not
          leak into the next one).
        - Monthly retraining is performed only during the FIRST evaluation
          distribution, and the resulting per-month checkpoints are saved.
          For the remaining distributions, the same checkpoints are reloaded
          at the same month boundaries, guaranteeing that all four
          distributions see the SAME model at any given day.
        - All distributions use independent forecast-error seeds derived
          from `cfg.seed` and the distribution index, so error realizations
          are reproducible across runs.

    Returns:
        df: long-format DataFrame with one row per (day, distribution)
        rep_day: representative-day snapshot, taken from the train distribution
                 if it is in eval_error_distributions, otherwise from the first
                 entry of eval_error_distributions.
    """
    battery_params = build_battery_params(cfg)
    flex_market = market["flex_market"]
    annual_prices: List[float] = market["annual_prices"]
    annual_services: List[List[FlexibilityService]] = market["annual_services"]

    eval_dists = list(cfg.eval_error_distributions)
    if not eval_dists:
        raise ValueError("eval_error_distributions must contain at least one entry")

    # The "primary" distribution is the one we collect the representative-day
    # snapshot from. We prefer the train distribution if present, otherwise
    # the first eval distribution.
    primary_dist = (cfg.train_error_distribution
                    if cfg.train_error_distribution in eval_dists
                    else eval_dists[0])

    # Path of the fresh pretrained model on disk (for reload between dists)
    pretrained_path = paths.models / "main_ppo_pretrained.zip"
    if ppo_model is not None and not pretrained_path.exists():
        # Save current model state so we can reload it between distributions
        ppo_model.save(str(pretrained_path))

    all_rows: List[Dict[str, Any]] = []
    primary_snapshots: List[Dict[str, Any]] = []

    print(f"  Evaluating across {len(eval_dists)} forecast-error "
          f"distributions: {eval_dists}")

    for dist_idx, dist in enumerate(eval_dists):
        print(f"\n  [Distribution {dist_idx+1}/{len(eval_dists)}] {dist}")

        # Pre-generate the forecast-error sequence for the whole horizon.
        # We use a per-distribution seed offset so that different distributions
        # produce different but reproducible error realizations, and we reset
        # the OU process state at the beginning.
        rng_state_seed = cfg.seed + 1000 * (dist_idx + 1)
        np.random.seed(rng_state_seed)
        err_gen = ForecastErrorGenerator(dist)
        err_gen.reset()
        all_errors = [err_gen.generate_error() for _ in range(cfg.days * 24)]

        # Reload the pretrained model fresh at the start of every distribution
        # so that retraining done in a previous distribution does not leak.
        if ppo_model is not None and pretrained_path.exists():
            try:
                from stable_baselines3 import PPO
                current_model = PPO.load(str(pretrained_path))
            except Exception as e:
                print(f"    WARNING: could not reload pretrained model ({e}); "
                      f"using in-memory model.")
                current_model = ppo_model
        else:
            current_model = ppo_model

        # Decide whether this pass is allowed to do monthly retraining.
        # We only retrain during the FIRST distribution and save checkpoints;
        # subsequent distributions reload the saved checkpoints.
        is_first_pass = (dist_idx == 0)
        do_retrain = is_first_pass and cfg.retrain_timesteps > 0 and current_model is not None

        milp_cum = ppo_cum = 0.0
        milp_th = ppo_th = 0.0

        for day in range(cfg.days):
            # Monthly model update: retrain on the first pass, reload on others
            if (current_model is not None
                    and day > 0
                    and day % (30 * cfg.retrain_frequency_months) == 0):
                month_idx = day // 30
                ckpt = paths.models / f"main_ppo_month_{month_idx:02d}.zip"
                if do_retrain:
                    window_start = max(0, (day - 60) * 24)
                    window_end = day * 24
                    retrain_ppo(current_model, cfg, paths,
                                annual_prices[window_start:window_end], month_idx)
                    # retrain_ppo already saves the checkpoint to that path
                elif ckpt.exists():
                    try:
                        from stable_baselines3 import PPO
                        current_model = PPO.load(str(ckpt))
                    except Exception as e:
                        print(f"      WARNING: could not reload monthly "
                              f"checkpoint {ckpt} ({e})")

            d0, d1 = day * 24, (day + 1) * 24
            daily_prices = annual_prices[d0:d1]
            daily_services = annual_services[d0:d1]
            daily_errors = all_errors[d0:d1]

            milp_res = run_milp_day(battery_params, flex_market,
                                    daily_prices, daily_services,
                                    forecast_errors=daily_errors)
            ppo_res = run_ppo_day(current_model, daily_prices,
                                  daily_services, cfg, eval_distribution=dist)

            milp_breakdown = compute_milp_service_breakdown(
                milp_res["flexibility_reservations"], daily_services
            )
            milp_th += sum(abs(p) for p in milp_res["power_trajectory"])
            ppo_th += sum(abs(p) for p in ppo_res["power_trajectory"])
            milp_cum += milp_res["net_profit"]
            ppo_cum += ppo_res["net_profit"]

            all_rows.append(
                {
                    "error_distribution": dist,
                    "day": day + 1,
                    "avg_price": float(np.mean(daily_prices)),
                    "price_volatility": float(np.std(daily_prices)),
                    "milp_net_profit": milp_res["net_profit"],
                    "ppo_net_profit": ppo_res["net_profit"],
                    "milp_arbitrage": milp_res["arbitrage_profit"],
                    "ppo_arbitrage": ppo_res["arbitrage_profit"],
                    "milp_flexibility": milp_res["flexibility_revenue"],
                    "ppo_flexibility": ppo_res["flexibility_revenue"],
                    "milp_degradation": milp_res["degradation_cost"],
                    "ppo_degradation": ppo_res["degradation_cost"],
                    "milp_fcr_revenue": milp_breakdown["FCR"],
                    "milp_afrr_revenue": milp_breakdown["aFRR"],
                    "milp_mfrr_revenue": milp_breakdown["mFRR"],
                    "ppo_fcr_revenue": ppo_res["fcr_revenue"],
                    "ppo_afrr_revenue": ppo_res["afrr_revenue"],
                    "ppo_mfrr_revenue": ppo_res["mfrr_revenue"],
                    "milp_exec_time": milp_res["execution_time"],
                    "ppo_exec_time": ppo_res["execution_time"],
                    "milp_utilization": milp_res["battery_utilization"],
                    "ppo_utilization": ppo_res["battery_utilization"],
                    "milp_cumulative": milp_cum,
                    "ppo_cumulative": ppo_cum,
                    "milp_throughput_cum": milp_th,
                    "ppo_throughput_cum": ppo_th,
                    # New tracking from Fix A
                    "ppo_arbitrage_throughput": ppo_res.get("arbitrage_throughput_mwh", 0.0),
                    "ppo_activation_throughput": ppo_res.get("activation_throughput_mwh", 0.0),
                    "ppo_total_throughput": ppo_res.get("total_throughput_mwh", 0.0),
                }
            )

            # Snapshot of daily trajectories on the primary distribution only
            if dist == primary_dist:
                primary_snapshots.append(
                    {
                        "day": day + 1,
                        "volatility": float(np.std(daily_prices)),
                        "prices": list(daily_prices),
                        "milp_soc": list(milp_res["soc_trajectory"]),
                        "milp_power": list(milp_res["power_trajectory"]),
                        "ppo_soc": list(ppo_res["soc_trajectory"]),
                        "ppo_power": list(ppo_res["power_trajectory"]),
                        "ppo_actions": ppo_res.get("actions_log", []),
                        "milp_flex_reservations": milp_res.get("flexibility_reservations", {}),
                        "ppo_flex_reservations": ppo_res.get("flexibility_reservations", {}),
                    }
                )

            if cfg.verbose and ((day + 1) % 30 == 0 or day == 0):
                print(
                    f"      Day {day+1:3d}/{cfg.days}: "
                    f"MILP={milp_cum:9.0f} EUR  PPO={ppo_cum:9.0f} EUR  "
                    f"gap={milp_cum-ppo_cum:+8.0f} EUR"
                )

    df = pd.DataFrame(all_rows)
    df.to_csv(paths.comparison / "daily_results.csv", index=False)

    if primary_snapshots:
        vols = np.array([s["volatility"] for s in primary_snapshots])
        med_idx = int(np.argmin(np.abs(vols - np.median(vols))))
        rep_day = primary_snapshots[med_idx]
        # Don't serialize the heavy action / reservation arrays inside the rep_day JSON
        rep_day_light = {k: v for k, v in rep_day.items()
                         if k not in ("ppo_actions", "milp_flex_reservations",
                                      "ppo_flex_reservations")}
        with open(paths.comparison / "representative_day.json", "w",
                  encoding="utf-8") as f:
            json.dump(rep_day_light, f, indent=2, default=str)

        # Save full diagnostics (action distribution + hourly reservations across
        # the whole primary distribution) for the diagnostic plots 11-14
        diagnostics = {
            "primary_distribution": primary_dist,
            "ppo_actions_all_days": [
                act for snap in primary_snapshots for act in snap.get("ppo_actions", [])
            ],
            "milp_flex_reservations_all_days": [
                snap.get("milp_flex_reservations", {}) for snap in primary_snapshots
            ],
            "ppo_flex_reservations_all_days": [
                snap.get("ppo_flex_reservations", {}) for snap in primary_snapshots
            ],
            "milp_soc_all_days": [snap.get("milp_soc", []) for snap in primary_snapshots],
            "ppo_soc_all_days":  [snap.get("ppo_soc", [])  for snap in primary_snapshots],
            "milp_power_all_days": [snap.get("milp_power", []) for snap in primary_snapshots],
            "ppo_power_all_days":  [snap.get("ppo_power", [])  for snap in primary_snapshots],
        }
        with open(paths.comparison / "diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, default=str)
    else:
        rep_day = None

    return df, rep_day


# ============================================================================
# Stage 4: Reporting
# ============================================================================


def make_summary(df: pd.DataFrame, cfg: PipelineConfig) -> Dict[str, Any]:
    """Aggregate daily results into a JSON-serializable summary, split by
    forecast-error distribution.

    The summary contains:
        - per_distribution: dict keyed by distribution name with final
          cumulative profit, average daily profit, breakdown, etc.
        - overall: pooled statistics across all distributions
        - robustness: coefficient of variation (CV) of annual profit across
          distributions for each algorithm. Lower CV = more robust.
    """
    dists = sorted(df["error_distribution"].unique())

    def _block_for(sub: pd.DataFrame) -> Dict[str, Any]:
        return {
            "horizon_days": int(sub["day"].max()),
            "final_cumulative_profit_eur": {
                "milp": float(sub["milp_cumulative"].iloc[-1]),
                "ppo":  float(sub["ppo_cumulative"].iloc[-1]),
                "milp_minus_ppo": float(sub["milp_cumulative"].iloc[-1] -
                                        sub["ppo_cumulative"].iloc[-1]),
            },
            "average_daily_profit_eur": {
                "milp": float(sub["milp_net_profit"].mean()),
                "ppo":  float(sub["ppo_net_profit"].mean()),
            },
            "profit_breakdown_total_eur": {
                "milp": {
                    "arbitrage":   float(sub["milp_arbitrage"].sum()),
                    "flexibility": float(sub["milp_flexibility"].sum()),
                    "fcr":         float(sub["milp_fcr_revenue"].sum()),
                    "afrr":        float(sub["milp_afrr_revenue"].sum()),
                    "mfrr":        float(sub["milp_mfrr_revenue"].sum()),
                    "degradation_cost": float(sub["milp_degradation"].sum()),
                },
                "ppo": {
                    "arbitrage":   float(sub["ppo_arbitrage"].sum()),
                    "flexibility": float(sub["ppo_flexibility"].sum()),
                    "fcr":         float(sub["ppo_fcr_revenue"].sum()),
                    "afrr":        float(sub["ppo_afrr_revenue"].sum()),
                    "mfrr":        float(sub["ppo_mfrr_revenue"].sum()),
                    "degradation_cost": float(sub["ppo_degradation"].sum()),
                },
            },
            "execution_time_seconds": {
                "milp_total":        float(sub["milp_exec_time"].sum()),
                "milp_mean_per_day": float(sub["milp_exec_time"].mean()),
                "ppo_total":         float(sub["ppo_exec_time"].sum()),
                "ppo_mean_per_day":  float(sub["ppo_exec_time"].mean()),
                "milp_over_ppo_ratio": float(
                    sub["milp_exec_time"].sum() / max(sub["ppo_exec_time"].sum(), 1e-9)
                ),
            },
            "battery_utilization": {
                "milp_mean": float(sub["milp_utilization"].mean()),
                "ppo_mean":  float(sub["ppo_utilization"].mean()),
            },
            "throughput_total_mwh": {
                "milp": float(sub["milp_throughput_cum"].iloc[-1]),
                "ppo":  float(sub["ppo_throughput_cum"].iloc[-1]),
            },
        }

    per_dist: Dict[str, Any] = {}
    for d in dists:
        sub = df[df["error_distribution"] == d].sort_values("day").reset_index(drop=True)
        per_dist[d] = _block_for(sub)

    # Robustness: coefficient of variation of annual profit across distributions
    milp_annual = np.array([per_dist[d]["final_cumulative_profit_eur"]["milp"] for d in dists])
    ppo_annual  = np.array([per_dist[d]["final_cumulative_profit_eur"]["ppo"]  for d in dists])

    def _cv(x: np.ndarray) -> float:
        mu = float(np.mean(x))
        return float(np.std(x) / abs(mu)) if abs(mu) > 1e-9 else float("inf")

    robustness = {
        "milp": {
            "annual_profit_mean":  float(np.mean(milp_annual)),
            "annual_profit_std":   float(np.std(milp_annual)),
            "annual_profit_range": float(np.ptp(milp_annual)),
            "coefficient_of_variation": _cv(milp_annual),
        },
        "ppo": {
            "annual_profit_mean":  float(np.mean(ppo_annual)),
            "annual_profit_std":   float(np.std(ppo_annual)),
            "annual_profit_range": float(np.ptp(ppo_annual)),
            "coefficient_of_variation": _cv(ppo_annual),
        },
    }

    return {
        "config": asdict(cfg),
        "evaluated_distributions": dists,
        "train_distribution": cfg.train_error_distribution,
        "per_distribution": per_dist,
        "robustness_cross_distribution": robustness,
    }


def write_report(summary: Dict[str, Any], paths: OutputPaths) -> None:
    """Write a plain-text summary report alongside the JSON."""
    with open(paths.comparison / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    cfg = summary["config"]
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("PPO vs MILP BESS Comparison - Cross-Distribution Summary Report")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Train distribution:           {summary['train_distribution']}")
    lines.append(f"Evaluated distributions:      {summary['evaluated_distributions']}")
    lines.append(f"Battery:                      {cfg['battery_capacity_mwh']} MWh / "
                 f"{cfg['battery_max_power_mw']} MW, "
                 f"cycle_life={cfg['cycle_life']}")
    lines.append("")

    # Per-distribution block
    for dist, block in summary["per_distribution"].items():
        lines.append("-" * 78)
        lines.append(f"Distribution: {dist}   (horizon = {block['horizon_days']} days)")
        lines.append("-" * 78)
        fcp = block["final_cumulative_profit_eur"]
        et  = block["execution_time_seconds"]
        pb  = block["profit_breakdown_total_eur"]
        th  = block["throughput_total_mwh"]
        lines.append(f"  Final cumulative profit:")
        lines.append(f"    MILP:                 {fcp['milp']:>14.2f} EUR")
        lines.append(f"    PPO:                  {fcp['ppo']:>14.2f} EUR")
        lines.append(f"    MILP - PPO gap:       {fcp['milp_minus_ppo']:>+14.2f} EUR")
        lines.append(f"  Execution time:")
        lines.append(f"    MILP total:           {et['milp_total']:>14.2f} s "
                     f"({et['milp_mean_per_day']:.3f} s/day)")
        lines.append(f"    PPO  total:           {et['ppo_total']:>14.2f} s "
                     f"({et['ppo_mean_per_day']:.3f} s/day)")
        lines.append(f"    MILP/PPO ratio:       {et['milp_over_ppo_ratio']:>14.2f}x")
        lines.append(f"  Throughput (MWh):       MILP={th['milp']:.2f}   PPO={th['ppo']:.2f}")
        lines.append(f"  MILP revenue breakdown (EUR):")
        lines.append(f"    Arbitrage:            {pb['milp']['arbitrage']:>+14.2f}")
        lines.append(f"    FCR:                  {pb['milp']['fcr']:>+14.2f}")
        lines.append(f"    aFRR:                 {pb['milp']['afrr']:>+14.2f}")
        lines.append(f"    mFRR:                 {pb['milp']['mfrr']:>+14.2f}")
        lines.append(f"    Degradation:          {-pb['milp']['degradation_cost']:>+14.2f}")
        lines.append(f"  PPO  revenue breakdown (EUR):")
        lines.append(f"    Arbitrage:            {pb['ppo']['arbitrage']:>+14.2f}")
        lines.append(f"    FCR:                  {pb['ppo']['fcr']:>+14.2f}")
        lines.append(f"    aFRR:                 {pb['ppo']['afrr']:>+14.2f}")
        lines.append(f"    mFRR:                 {pb['ppo']['mfrr']:>+14.2f}")
        lines.append(f"    Degradation:          {-pb['ppo']['degradation_cost']:>+14.2f}")
        lines.append("")

    # Robustness section
    lines.append("=" * 78)
    lines.append("Robustness across forecast-error distributions")
    lines.append("=" * 78)
    lines.append("")
    rb = summary["robustness_cross_distribution"]
    lines.append("  Annual profit across distributions (mean +/- std):")
    lines.append(f"    MILP: {rb['milp']['annual_profit_mean']:>10.2f} "
                 f"+/- {rb['milp']['annual_profit_std']:.2f} EUR "
                 f"(range = {rb['milp']['annual_profit_range']:.2f}, "
                 f"CV = {rb['milp']['coefficient_of_variation']:.4f})")
    lines.append(f"    PPO:  {rb['ppo']['annual_profit_mean']:>10.2f} "
                 f"+/- {rb['ppo']['annual_profit_std']:.2f} EUR "
                 f"(range = {rb['ppo']['annual_profit_range']:.2f}, "
                 f"CV = {rb['ppo']['coefficient_of_variation']:.4f})")
    lines.append("")
    lines.append("  Interpretation: lower CV = more robust to forecast-error structure.")
    lines.append("  Compare the two CVs to assess which algorithm degrades less under")
    lines.append("  forecast-error distributions it was not trained on.")
    lines.append("")
    lines.append("=" * 78)

    report_text = "\n".join(lines)
    with open(paths.comparison / "report.txt", "w") as f:
        f.write(report_text)
    print("\n" + report_text)


# ============================================================================
# Orchestration
# ============================================================================


def main() -> int:
    cfg = parse_args()
    np.random.seed(cfg.seed)
    paths = OutputPaths.create(cfg.output_root, cfg.run_name)

    print("=" * 78)
    print(f"BESS Comparison Pipeline ({cfg.mode} mode)")
    print(f"Run name:        {cfg.run_name}")
    print(f"Output root:     {paths.root}")
    print(f"Days:            {cfg.days}")
    print(f"Pretrain months: {cfg.pretrain_months}")
    print(f"Train dist:      {cfg.train_error_distribution}")
    print(f"Eval dists:      {cfg.eval_error_distributions}")
    print(f"Seed:            {cfg.seed}")
    print("=" * 78)

    # Stage 2: market data
    print("\n[1/4] Generating market data...")
    market = build_market_data(cfg, paths)

    # Stage 3: PPO training
    ppo_model: Optional["PPO"] = None
    if cfg.mode != "compare" and not cfg.skip_train and cfg.pretrain_months > 0:
        print("\n[2/4] Pretraining PPO...")
        train_ppo(cfg, paths, market["pretrain_prices"])

    if cfg.mode == "train":
        print("\nTraining-only mode finished. Skipping comparison.")
        return 0

    # Load model for comparison
    if PPO_AVAILABLE:
        model_path = cfg.model_path if cfg.skip_train else str(paths.models / "main_ppo_pretrained.zip")
        if Path(model_path).exists():
            print(f"\n  Loading PPO model from {model_path}")
            ppo_model = PPO.load(model_path)
        else:
            print(f"\n  WARNING: PPO model not found at {model_path}. Using greedy fallback.")
    else:
        print("\n  WARNING: stable-baselines3 not installed. Using greedy fallback.")

    # Stage 4: comparison
    print("\n[3/4] Running annual comparison...")
    t0 = time.time()
    df, rep_day = run_annual_comparison(cfg, paths, market, ppo_model)
    elapsed = time.time() - t0
    print(f"  Comparison completed in {elapsed/60:.1f} minutes")

    # Stage 5: reporting
    print("\n[4/4] Generating reports and plots...")
    summary = make_summary(df, cfg)
    write_report(summary, paths)
    primary = (cfg.train_error_distribution
               if cfg.train_error_distribution in cfg.eval_error_distributions
               else cfg.eval_error_distributions[0])

    # Load diagnostics that the comparison loop has saved
    diagnostics = None
    diag_path = paths.comparison / "diagnostics.json"
    if diag_path.exists():
        try:
            with open(diag_path, "r", encoding="utf-8") as f:
                diagnostics = json.load(f)
        except Exception as e:
            print(f"  WARNING: could not load diagnostics.json ({e})")

    # Build action_space_info from a temporary env (cheap)
    action_space_info = None
    try:
        tmp_env = make_env([50.0] * 24, cfg.train_error_distribution, cfg.seed,
                           enable_reward_shaping=False,
                           flex_price_error_pct=cfg.flex_price_error_pct)
        action_space_info = {
            "arbitrage_values": list(tmp_env.arbitrage_values),
            "fcr_percentages": list(tmp_env.fcr_percentages),
            "afrr_percentages": list(tmp_env.afrr_percentages),
            "mfrr_percentages": list(tmp_env.mfrr_percentages),
        }
    except Exception as e:
        print(f"  WARNING: could not build action_space_info ({e})")

    generate_all_plots(df, paths.figures,
                       representative_day=rep_day,
                       primary_distribution=primary,
                       train_distribution=cfg.train_error_distribution,
                       diagnostics=diagnostics,
                       action_space_info=action_space_info,
                       verbose=cfg.verbose)

    print(f"\nAll outputs saved under: {paths.root.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception:
        print("\nFATAL ERROR:")
        traceback.print_exc()
        sys.exit(1)