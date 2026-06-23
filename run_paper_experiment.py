""""
Run the multi-agent BSP paper experiment end-to-end.

Usage
-----
Edit the CONFIG block below to choose the scale you want, then:

    python run_paper_experiment.py

Three scales are supported via the SCALE constant:

    "smoke"       N=3 BESS, 7 train days + 3 test days     (~30 seconds)
                  Quick sanity check that the pipeline runs on your machine.

    "medium"      N=10 BESS, 30 train days + 10 test days  (~2-5 minutes)
                  Good for iterating on hyperparameters during development.

    "production"  N=50 heterogeneous fleet, 273 + 92 days  (~20-40 minutes)
                  Paper headline numbers.

Three outputs are written to ./results/<run_id>/:

    metrics.json     full FullPipelineResult fields as JSON, ready for tables
    bc_policy.pt     trained BC network state_dict, ready for deployment
    daily_profits.csv  per-day MILP and BC profits over the test window

When you have the real PUN 2024 CSV, set USE_REAL_PUN=True and point
REAL_PUN_CSV_PATH to your file (with columns 'datetime' and 'pun_eur_mwh').
"""

from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch

warnings.simplefilter("ignore")

# ============================================================================
# CONFIG - edit this block
# ============================================================================

SCALE = "smoke"   # one of: "smoke", "medium", "production"

USE_REAL_PUN = True
REAL_PUN_XLSX_PATH = [
    "./data/20230101_20231231_PUN.xlsx",
    "./data/20240101_20241231_PUN.xlsx",
]

USE_REAL_MSD = True
REAL_MSD_XLSX_PATHS = [
    "./data/EsitiMSD_PrezziExAnte_20230101_20230401_Nord.xlsx",
    "./data/EsitiMSD_PrezziExAnte_20230401_20240331_Nord.xlsx",
    "./data/EsitiMSD_PrezziExAnte_20240401_20241231_Nord.xlsx",
]

USE_NONLINEAR_DEGRADATION = False  # set True to use Schmalstieg+Naumann
                                     # physics-based MILP rate (Step 4)

# ---- PPO comparison (Step 6 algorithm) ----
# Set RUN_PPO_VANILLA=True to train PPO from scratch and add it as a fourth
# column in the test-window comparison. Set RUN_PPO_BC_WARMSTART=True to
# additionally train a PPO initialised from the BC weights (the washout-test).
# PPO_ITERATIONS controls training length; 50-200 is typical for smoke,
# 500-2000 for production paper runs.
RUN_PPO_VANILLA = True
RUN_PPO_BC_WARMSTART = True
PPO_ITERATIONS = 200   # smoke default; raise for production runs
BC_LOSS_SIGMA = 1.0

# Directional services mode (June 2026 extension):
# When True, aFRR and mFRR are split into upward (BSP discharges) and downward
# (BSP charges) directions with mutual exclusion per service per hour. Bids
# can only be in ONE direction per service per hour. FCR stays symmetric.
# The MILP enforces SOC sustain asymmetrically (upward needs energy, downward
# needs headroom), and SOC dynamics include service activations.
# When True, requires REAL_MSD_XLSX_PATHS (downward prices come from Prezzo
# Medio di Vendita column, upward from Prezzo Medio di Acquisto).
# Note: action space switches from 5 to 7 axes; BC and PPO retrain accordingly.
DIRECTIONAL_SERVICES = True

OUTPUT_DIR = Path("./results")

# ============================================================================


def get_scale_config():
    """Return (fleet_builder, pipeline_kwargs) for the chosen scale."""
    from milp_optimizer import BatteryParameters
    from milp_optimizer_multi import MultiBatteryParameters

    if SCALE == "smoke":
        def fleet_builder():
            bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0,
                                    degradation_cost_per_mwh=25000.0,
                                    cycle_life=6000)
            return MultiBatteryParameters([bp] * 3)
        pipeline_kwargs = dict(
            train_start=datetime(2024, 4, 1),
            train_days=7,
            test_days=3,
            bc_n_epochs=30,
            bc_batch_size=32,
            bc_lr=3e-3,
            enable_commitments=True,
            commitment_lead_time=24,
            penalty_k=1.5,
            master_seed=42,
        )
    elif SCALE == "medium":
        def fleet_builder():
            bp = BatteryParameters(capacity_mwh=4.0, max_power_mw=2.0,
                                    degradation_cost_per_mwh=25000.0,
                                    cycle_life=6000)
            return MultiBatteryParameters([bp] * 10)
        pipeline_kwargs = dict(
            train_start=datetime(2024, 4, 1),
            train_days=30,
            test_days=10,
            bc_n_epochs=50,
            bc_batch_size=64,
            bc_lr=1e-3,
            enable_commitments=True,
            commitment_lead_time=24,
            penalty_k=1.5,
            master_seed=42,
        )
    elif SCALE == "production":
        def fleet_builder():
            return MultiBatteryParameters.heterogeneous_fleet(seed=0)
        pipeline_kwargs = dict(
            train_start=datetime(2023, 1, 1),
            train_days=365,
            test_days=366,   # 2024 is a leap year
            bc_n_epochs=50,
            bc_batch_size=128,
            bc_lr=1e-3,
            enable_commitments=True,
            commitment_lead_time=24,
            penalty_k=1.5,
            master_seed=42,
        )
    else:
        raise ValueError(f"unknown SCALE: {SCALE}")
    return fleet_builder, pipeline_kwargs


def main():
    # Lazy imports so the file can be parsed even before deps are installed
    from pipeline_step8 import run_full_pipeline

    fleet_builder, pipeline_kwargs = get_scale_config()
    fleet = fleet_builder()

    run_id = f"{SCALE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = OUTPUT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive test window start (used by extra_charts for calendar axes)
    test_start = (pipeline_kwargs["train_start"]
                  + timedelta(days=pipeline_kwargs["train_days"]))

    print("=" * 70)
    print(f"Multi-agent BSP paper experiment")
    print("=" * 70)
    print(f"Run ID:       {run_id}")
    print(f"Scale:        {SCALE}")
    print(f"Fleet size:   N = {fleet.n_batteries} BESS")
    print(f"Train days:   {pipeline_kwargs['train_days']}")
    print(f"Test days:    {pipeline_kwargs['test_days']}")
    print(f"Non-lin deg:  {USE_NONLINEAR_DEGRADATION}")
    print(f"Directional:  {DIRECTIONAL_SERVICES}  "
          f"(aFRR/mFRR up+dn mutex; FCR symmetric)")
    print(f"PPO vanilla:  {RUN_PPO_VANILLA}  (iters={PPO_ITERATIONS})")
    print(f"PPO BC-warm:  {RUN_PPO_BC_WARMSTART}")
    print(f"Output dir:   {out_dir}")
    print()

    if USE_REAL_PUN:
        # REAL_PUN_XLSX_PATH may be a single string or a list of strings.
        # Validate each path; the loader (load_pun_from_gme_xlsx_multi) accepts
        # both forms and concatenates/de-duplicates internally.
        if isinstance(REAL_PUN_XLSX_PATH, (str, bytes)):
            pun_paths = [REAL_PUN_XLSX_PATH]
        else:
            pun_paths = list(REAL_PUN_XLSX_PATH)
        for p in pun_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"USE_REAL_PUN=True but PUN xlsx not found: {p}"
                )
        if len(pun_paths) == 1:
            print(f"PUN source:   REAL GME xlsx from {pun_paths[0]}")
        else:
            print(f"PUN source:   REAL GME from {len(pun_paths)} files:")
            for p in pun_paths:
                print(f"                {p}")
    else:
        pun_paths = None
        print(f"PUN source:   SYNTHETIC generator")

    msd_paths = None
    if USE_REAL_MSD:
        if not REAL_MSD_XLSX_PATHS:
            raise ValueError("USE_REAL_MSD=True but REAL_MSD_XLSX_PATHS is empty")
        for p in REAL_MSD_XLSX_PATHS:
            if not os.path.exists(p):
                raise FileNotFoundError(f"MSD file not found: {p}")
        msd_paths = list(REAL_MSD_XLSX_PATHS)
        print(f"MSD source:   REAL Terna ex-ante from {len(msd_paths)} file(s)")
    else:
        print(f"MSD source:   SYNTHETIC (calibrated defaults from reference workbook)")
    print()

    # `real_pun_xlsx_path` accepts either a string or a list — the loader
    # (load_pun_from_gme_xlsx_multi inside make_market_window_from_real_pun)
    # normalises internally.
    real_pun_arg = (pun_paths[0] if pun_paths and len(pun_paths) == 1
                    else pun_paths)

    t0 = time.time()
    result = run_full_pipeline(
        fleet=fleet,
        use_nonlinear_degradation=USE_NONLINEAR_DEGRADATION,
        real_pun_xlsx_path=real_pun_arg,
        real_msd_xlsx_paths=msd_paths,
        run_ppo_vanilla=RUN_PPO_VANILLA,
        run_ppo_bc_warmstart=RUN_PPO_BC_WARMSTART,
        ppo_iterations=PPO_ITERATIONS,
        directional_services=DIRECTIONAL_SERVICES,
        **pipeline_kwargs,
    )
    elapsed = time.time() - t0

    # ---- Print headline table ----
    print()
    print("=" * 70)
    print("HEADLINE RESULTS")
    print("=" * 70)
    print(f"Total elapsed:                {elapsed:.1f} seconds")
    print(f"  - demo generation:          {result.demo_generation_seconds:.1f}s")
    print(f"  - BC training:              {result.bc_training_seconds:.1f}s")
    print(f"  - evaluation:               {result.evaluation_seconds:.1f}s")
    print()
    print(f"BC dataset size:              {result.bc_dataset_size:,} samples")
    print(f"BC final training loss:       {result.bc_final_train_loss:.4f}")
    print(f"BC validation accuracy:       "
          f"{[round(a, 3) for a in result.bc_final_val_accuracy_per_axis]}")
    print()
    print(f"Train MILP total profit:      "
          f"{result.train_total_milp_profit_eur:>14,.2f} EUR")
    print(f"  (over {result.train_milp_optimal_day_count}/{result.train_days} days)")
    print()
    print("TEST WINDOW (env-realised, apples-to-apples)")
    if result.test_milp_objective_eur is not None:
        print(f"  MILP objective (continuous):"
              f"{result.test_milp_objective_eur:>14,.2f} EUR  <- upper bound")
    print(f"  MILP realised (discrete):   "
          f"{result.test_total_milp_profit_eur:>14,.2f} EUR")
    print(f"  BC trained policy:          "
          f"{result.test_total_bc_profit_eur:>14,.2f} EUR")
    if result.test_ppo_vanilla_profit_eur is not None:
        print(f"  PPO vanilla:                "
              f"{result.test_ppo_vanilla_profit_eur:>14,.2f} EUR")
    if result.test_ppo_bc_warmstart_profit_eur is not None:
        print(f"  PPO BC-warmstart:           "
              f"{result.test_ppo_bc_warmstart_profit_eur:>14,.2f} EUR")
    print(f"  Random baseline:            "
          f"{result.test_total_random_profit_eur:>14,.2f} EUR")
    print()
    print(f"  BC gap to MILP:             "
          f"{result.bc_gap_to_milp_percent:>+13.2f} %")
    print(f"  BC uplift over random:      "
          f"{result.bc_uplift_over_random_factor:>13.2f} x")
    print("=" * 70)

    # ---- Save metrics JSON ----
    metrics = {
        "run_id": run_id,
        "scale": SCALE,
        "use_nonlinear_degradation": USE_NONLINEAR_DEGRADATION,
        "n_batteries": result.n_batteries,
        "train_days": result.train_days,
        "test_days": result.test_days,
        "timing": {
            "total_seconds": elapsed,
            "demo_generation_seconds": result.demo_generation_seconds,
            "bc_training_seconds": result.bc_training_seconds,
            "evaluation_seconds": result.evaluation_seconds,
        },
        "train": {
            "total_milp_profit_eur": result.train_total_milp_profit_eur,
            "optimal_day_count": result.train_milp_optimal_day_count,
        },
        "bc": {
            "dataset_size": result.bc_dataset_size,
            "final_train_loss": result.bc_final_train_loss,
            "final_val_accuracy_per_axis": result.bc_final_val_accuracy_per_axis,
        },
        "test": {
            "milp_objective_continuous_eur": result.test_milp_objective_eur,
            "milp_oracle_total_eur": result.test_total_milp_profit_eur,
            "bc_policy_total_eur": result.test_total_bc_profit_eur,
            "ppo_vanilla_total_eur": result.test_ppo_vanilla_profit_eur,
            "ppo_bc_warmstart_total_eur": result.test_ppo_bc_warmstart_profit_eur,
            "random_baseline_total_eur": result.test_total_random_profit_eur,
            "milp_daily_profits": [float(p) for p in result.test_milp_daily_profits],
            "bc_daily_profits": [float(p) for p in result.test_bc_daily_profits],
            "ppo_vanilla_daily_profits": (
                [float(p) for p in result.test_ppo_vanilla_daily_profits]
                if result.test_ppo_vanilla_daily_profits is not None else None
            ),
            "ppo_bc_warmstart_daily_profits": (
                [float(p) for p in result.test_ppo_bc_warmstart_daily_profits]
                if result.test_ppo_bc_warmstart_daily_profits is not None else None
            ),
            "decomposition": result.decomposition,
        },
        "headline": {
            "bc_gap_to_milp_percent": result.bc_gap_to_milp_percent,
            "bc_uplift_over_random_factor": result.bc_uplift_over_random_factor,
        },
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {out_dir / 'metrics.json'}")

    # ---- Save per-day CSV ----
    with open(out_dir / "daily_profits.csv", "w") as f:
        header = ["day", "milp_oracle_eur", "bc_policy_eur"]
        if result.test_ppo_vanilla_daily_profits is not None:
            header.append("ppo_vanilla_eur")
        if result.test_ppo_bc_warmstart_daily_profits is not None:
            header.append("ppo_bc_warmstart_eur")
        f.write(",".join(header) + "\n")
        n_days = len(result.test_milp_daily_profits)
        for d in range(n_days):
            row = [
                str(d + 1),
                f"{result.test_milp_daily_profits[d]:.4f}",
                f"{result.test_bc_daily_profits[d]:.4f}",
            ]
            if result.test_ppo_vanilla_daily_profits is not None:
                row.append(f"{result.test_ppo_vanilla_daily_profits[d]:.4f}")
            if result.test_ppo_bc_warmstart_daily_profits is not None:
                row.append(f"{result.test_ppo_bc_warmstart_daily_profits[d]:.4f}")
            f.write(",".join(row) + "\n")
    print(f"Saved: {out_dir / 'daily_profits.csv'}")

    # ---- Per-policy behavioural analysis (terminal report + charts) ----
    if result.trajectories is not None:
        try:
            from analysis_step8 import (print_market_behaviour_report,
                                          make_paper_charts)
            print()
            print_market_behaviour_report(result.trajectories, fleet,
                                            output_dir=out_dir)
            make_paper_charts(result.trajectories, out_dir, result=result,
                                ppo_training_metrics=result.ppo_training_metrics)
        except Exception as exc:
            print(f"[warn] analysis step failed: {exc}")

    # ---- Extra paper figures (monthly stacked revenue, activation 2D hist,
    #      annual SOC/NET carpets, profit-vs-degradation scatter) ----
    if result.trajectories is not None:
        try:
            from extra_charts import make_extra_charts
            p_max_fleet = float(sum(b.max_power_mw for b in fleet.batteries))
            make_extra_charts(
                result.trajectories,
                out_dir,
                start_date=test_start,
                p_max_fleet=p_max_fleet,
            )
        except Exception as exc:
            print(f"[warn] extra_charts step failed: {exc}")

    # ---- Save BC policy weights ----
    # We need to retrain or get the bc_net from the result. Currently the
    # pipeline doesn't expose the trained net; we redo BC training here for
    # the smoke runs (cheap) and skip for production (expensive; instead
    # extend the pipeline to return the net if you need it).
    print()
    print(f"NOTE: trained BC weights not saved by default.")
    print(f"      To export, extend pipeline_step8.run_full_pipeline to return")
    print(f"      the bc_net object and torch.save it here.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()