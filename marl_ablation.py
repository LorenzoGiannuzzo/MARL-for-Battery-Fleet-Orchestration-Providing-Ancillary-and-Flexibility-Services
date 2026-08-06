# -*- coding: utf-8 -*-
"""
marl_ablation.py — Controlled ablation of the behavioural-cloning warm start,
repeated across seeds, plus the tau_FCR sensitivity sweep.

WHY
---
All three EI.A reviewers rejected the warm-start claim on the same grounds:
the warm-started agent differed from the vanilla agent in FOUR respects at
once (initialisation, a tenfold lower learning rate, no entropy bonus, and a
persistent KL anchoring term), and each variant was trained exactly once.

    R1 pt 1: "controlled ablations in which all PPO settings are identical
              except for the initial weights, followed by separate tests of
              the anchoring term and the altered learning settings. A
              behavioural-clone-only policy should also be reported.
              Results should be repeated across several random seeds and
              presented with means, dispersion, and learning curves."
    R2 pt 6: "three to five seeds plus a one-factor ablation on the
              anchoring term would suffice."
    R3 pt 1: "An ablation study is needed before attributing the performance
              improvement specifically to behavioural cloning."

This module runs exactly that.

DESIGN
------
The grid is CUMULATIVE (one factor added per row), not full-factorial. With
five configurations and three seeds it is fifteen PPO runs; the full 2^3
factorial on top of initialisation would be twenty-four, which buys little
because the factors are not independent in practice (the low learning rate
exists to protect the initialisation). The cumulative ladder answers the
reviewers' question directly: how much of the gain survives when you remove
each protection in turn.

    id  configuration                       init  lr     entropy  anchor
    --  ----------------------------------  ----  -----  -------  ------
    V   vanilla PPO                         no    1.0x   base     no
    I   BC init only                        yes   1.0x   base     no
    IL  BC init + low LR                    yes   0.1x   base     no
    ILE BC init + low LR + no entropy       yes   0.1x   0.0      no
    W   full warm start (paper config)      yes   0.1x   0.0      yes

Two reference rows come free with every pipeline run and are recorded
alongside: the discrete MILP benchmark and the BC-only policy. The BC-only
row is the one R1 explicitly asked for and it is the row most likely to
change the paper's narrative.

USAGE
-----
    # full ablation, 3 seeds  (this is the expensive one)
    python marl_ablation.py --mode warmstart --seeds 0 1 2

    # tau_FCR sensitivity (Reviewer 2 pt 2); MILP + BC only, no PPO
    python marl_ablation.py --mode tau_fcr

    # smoke test: tiny fleet, few days, 2 iterations, verifies plumbing
    python marl_ablation.py --mode warmstart --smoke

Outputs, written to --outdir (default ablation_results/):
    ablation_raw.csv        one row per (config, seed)
    ablation_summary.csv    mean / std / n per config
    ablation_table.md       the table to paste into the paper
    ablation_curves.json    per-iteration learning curves, per (config, seed)

RESUMABILITY
------------
Each (config, seed) cell is checkpointed to outdir as soon as it finishes.
Re-running skips completed cells, so a crash fifteen hours into a sweep does
not cost the whole sweep. Delete the .json cell files to force a re-run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from market_constants import (SustainDurations, DEFAULT_SUSTAIN,
                              TAU_FCR_SWEEP)


# ===========================================================================
# Configuration grid
# ===========================================================================

@dataclass(frozen=True)
class AblationConfig:
    """One row of the ablation ladder."""
    key: str
    label: str
    warm: bool                      # factor 1: BC initialisation
    lr_scale: float                 # factor 2: learning-rate scale
    entropy: Optional[float]        # factor 3: None = keep vanilla entropy
    anchor: bool                    # factor 4: KL anchoring to the clone

    def factors(self) -> str:
        return (f"init={'BC' if self.warm else 'random'}, "
                f"lr={self.lr_scale:g}x, "
                f"entropy={'base' if self.entropy is None else self.entropy:g}"
                if self.entropy is not None else
                f"init={'BC' if self.warm else 'random'}, "
                f"lr={self.lr_scale:g}x, entropy=base") + \
               f", anchor={'on' if self.anchor else 'off'}"


WARMSTART_LADDER: Tuple[AblationConfig, ...] = (
    AblationConfig("V",   "PPO vanilla",
                   warm=False, lr_scale=1.0, entropy=None, anchor=False),
    AblationConfig("I",   "PPO + BC init",
                   warm=True,  lr_scale=1.0, entropy=None, anchor=False),
    AblationConfig("IL",  "PPO + BC init + low LR",
                   warm=True,  lr_scale=0.1, entropy=None, anchor=False),
    AblationConfig("ILE", "PPO + BC init + low LR + no entropy",
                   warm=True,  lr_scale=0.1, entropy=0.0,  anchor=False),
    AblationConfig("W",   "PPO warm start (full)",
                   warm=True,  lr_scale=0.1, entropy=0.0,  anchor=True),
)


# ===========================================================================
# Result records
# ===========================================================================

@dataclass
class CellResult:
    """Outcome of one (config, seed) pair."""
    config_key: str
    config_label: str
    seed: int
    sustain_label: str
    tau_fcr: float

    # headline profits on the held-out window, EUR
    milp_discrete_eur: float = float("nan")
    milp_objective_eur: float = float("nan")
    bc_eur: float = float("nan")
    ppo_eur: float = float("nan")
    # Same weights, sampled instead of argmax. Reported alongside because
    # the two diverge sharply until the policy concentrates its mass.
    ppo_eur_stochastic: float = float("nan")
    random_eur: float = float("nan")

    # derived
    ppo_share_of_milp: float = float("nan")
    bc_share_of_milp: float = float("nan")

    wall_seconds: float = float("nan")
    ok: bool = False
    error: str = ""
    learning_curve: List[float] = field(default_factory=list)

    # Set when the trained policy emits a constant (all-zero) action at
    # evaluation. A collapsed policy scores exactly 0 EUR with every revenue
    # component at 0, which previously logged as a normal "OK" result. It is
    # not a result, it is a training failure, and it has to be loud: a grid
    # of five collapsed configurations measures nothing at all.
    collapsed: bool = False

    def finalise(self) -> "CellResult":
        m = self.milp_discrete_eur
        if m and not math.isnan(m) and abs(m) > 1e-9:
            self.ppo_share_of_milp = self.ppo_eur / m
            self.bc_share_of_milp = self.bc_eur / m
        return self


# ===========================================================================
# Runner
# ===========================================================================

def _cell_path(outdir: str, cfg_key: str, seed: int, tag: str) -> str:
    return os.path.join(outdir, f"cell_{tag}_{cfg_key}_seed{seed}.json")


def run_cell(
    cfg: AblationConfig,
    seed: int,
    *,
    fleet,
    train_start,
    train_days: int,
    test_days: int,
    sustain: SustainDurations,
    ppo_iterations: int,
    real_pun_xlsx_path: Optional[str],
    real_msd_xlsx_paths: Optional[List[str]],
    expected_reward_training: bool,
    milp_mode: str,
    extra_pipeline_kwargs: Dict[str, Any],
) -> CellResult:
    """Run one (configuration, seed) pair through the full pipeline."""
    from marl_pipeline import run_full_pipeline

    res = CellResult(
        config_key=cfg.key, config_label=cfg.label, seed=seed,
        sustain_label=sustain.label, tau_fcr=sustain.fcr,
    )
    t0 = time.time()
    try:
        out = run_full_pipeline(
            fleet=fleet,
            train_start=train_start,
            train_days=train_days,
            test_days=test_days,
            master_seed=seed,
            sustain=sustain,
            # Exactly one PPO variant per cell. The vanilla slot is used for
            # the un-warm-started row and the warm slot for every row that
            # starts from BC weights, so the three decoupled factors below
            # are the only thing that changes across rows.
            run_ppo_vanilla=(not cfg.warm),
            run_ppo_bc_warmstart=cfg.warm,
            warmstart_lr_scale=cfg.lr_scale,
            warmstart_entropy_coeff=cfg.entropy,
            use_kl_anchor=cfg.anchor,
            ppo_iterations=ppo_iterations,
            directional_services=True,
            real_pun_xlsx_path=real_pun_xlsx_path,
            real_msd_xlsx_paths=real_msd_xlsx_paths,
            expected_reward_training=expected_reward_training,
            milp_mode=milp_mode,
            **extra_pipeline_kwargs,
        )
        res.milp_discrete_eur = float(out.test_total_milp_profit_eur)
        res.milp_objective_eur = float(out.test_milp_objective_eur or float("nan"))
        res.bc_eur = float(out.test_total_bc_profit_eur)
        res.random_eur = float(out.test_total_random_profit_eur)
        res.ppo_eur = float(
            out.test_ppo_vanilla_profit_eur if not cfg.warm
            else out.test_ppo_bc_warmstart_profit_eur)
        _st = (out.test_ppo_vanilla_profit_stochastic_eur if not cfg.warm
               else out.test_ppo_bc_warmstart_profit_stochastic_eur)
        res.ppo_eur_stochastic = (float("nan") if _st is None
                                  else float(_st))
        curves = out.ppo_training_metrics or {}
        res.learning_curve = list(
            curves.get("ppo_vanilla" if not cfg.warm else "ppo_bc", []))

        # Collapse detection. Every revenue component at exactly zero is the
        # signature of a policy that takes the no-op action on every axis at
        # every hour: no arbitrage, no reserve, and therefore no degradation
        # either. Floating-point profit is never exactly 0.0 by chance once a
        # single MWh has moved.
        decomp = (out.decomposition or {}).get(
            "ppo_vanilla" if not cfg.warm else "ppo_bc", {})
        parts = [float(decomp.get(k, 0.0) or 0.0)
                 for k in ("arbitrage", "flexibility", "degradation")]
        # A genuine collapse means the policy earns nothing under EITHER
        # protocol. Zero under argmax alone is not a training failure: it
        # means the mode is the no-op while the distribution still earns,
        # which is what a high-entropy policy looks like early in
        # training.
        _stoch_dead = (math.isnan(res.ppo_eur_stochastic)
                       or abs(res.ppo_eur_stochastic) < 1.0)
        res.collapsed = (abs(res.ppo_eur) < 1e-9
                         and all(abs(p) < 1e-9 for p in parts)
                         and _stoch_dead)
        res.ok = True
    except Exception as exc:  # keep the sweep alive; record and move on
        res.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    res.wall_seconds = time.time() - t0
    return res.finalise()


def run_ablation(
    *,
    configs: Tuple[AblationConfig, ...],
    seeds: List[int],
    fleet,
    train_start,
    train_days: int,
    test_days: int,
    sustain: SustainDurations,
    ppo_iterations: int,
    outdir: str,
    tag: str = "warmstart",
    real_pun_xlsx_path: Optional[str] = None,
    real_msd_xlsx_paths: Optional[List[str]] = None,
    expected_reward_training: bool = False,
    milp_mode: str = "discrete",
    extra_pipeline_kwargs: Optional[Dict[str, Any]] = None,
) -> List[CellResult]:
    os.makedirs(outdir, exist_ok=True)
    extra_pipeline_kwargs = extra_pipeline_kwargs or {}
    results: List[CellResult] = []

    total = len(configs) * len(seeds)
    done = 0
    print("=" * 74)
    print(f"ABLATION [{tag}]  {len(configs)} configs x {len(seeds)} seeds "
          f"= {total} runs")
    print(f"  sustain: {sustain.describe()}")
    print(f"  PPO iterations per run: {ppo_iterations}")
    print(f"  training reward: "
          f"{'EXPECTED (MILP-aligned)' if expected_reward_training else 'SAMPLED (Bernoulli draws)'}")
    print(f"  outdir: {outdir}")
    print("=" * 74)

    for cfg in configs:
        for seed in seeds:
            done += 1
            path = _cell_path(outdir, cfg.key, seed, tag)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    cached = CellResult(**json.load(f))
                results.append(cached)
                print(f"[{done}/{total}] {cfg.key:<4} seed={seed}  CACHED "
                      f"(ppo {cached.ppo_eur:,.0f} EUR)")
                continue

            print(f"\n[{done}/{total}] {cfg.key:<4} seed={seed}  "
                  f"{cfg.label}")
            print(f"          factors: {cfg.factors()}")
            r = run_cell(
                cfg, seed,
                fleet=fleet, train_start=train_start,
                train_days=train_days, test_days=test_days,
                sustain=sustain, ppo_iterations=ppo_iterations,
                real_pun_xlsx_path=real_pun_xlsx_path,
                real_msd_xlsx_paths=real_msd_xlsx_paths,
                expected_reward_training=expected_reward_training,
                milp_mode=milp_mode,
                extra_pipeline_kwargs=extra_pipeline_kwargs,
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(r), f, indent=2)
            results.append(r)
            if not r.ok:
                status = f"FAILED ({r.error})"
            elif r.collapsed:
                status = "COLLAPSED"
            else:
                status = "OK"
            _sto = ("" if math.isnan(r.ppo_eur_stochastic)
                    else f" | sampled={r.ppo_eur_stochastic:,.0f}")
            print(f"          -> {status}  argmax={r.ppo_eur:,.0f} EUR"
                  f"{_sto}  ({r.wall_seconds/60:.1f} min)")
            if r.collapsed:
                print("             the policy takes the no-op action on "
                      "every axis; this is a training failure, not a result.")
                print(f"             the clone it started from scored "
                      f"{r.bc_eur:,.0f} EUR on the same window.")

    return results


# ===========================================================================
# Reporting
# ===========================================================================

def _agg(values: List[float]) -> Tuple[float, float, int]:
    v = [x for x in values if x is not None and not math.isnan(x)]
    if not v:
        return float("nan"), float("nan"), 0
    if len(v) == 1:
        return float(v[0]), float("nan"), 1
    return float(np.mean(v)), float(np.std(v, ddof=1)), len(v)


def write_reports(results: List[CellResult], outdir: str,
                  tag: str = "warmstart") -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    paths = {}

    # ---- raw ----
    raw_path = os.path.join(outdir, f"{tag}_raw.csv")
    cols = ["config_key", "config_label", "seed", "sustain_label", "tau_fcr",
            "milp_discrete_eur", "milp_objective_eur", "bc_eur", "ppo_eur",
            "ppo_eur_stochastic", "random_eur", "ppo_share_of_milp",
            "bc_share_of_milp",
            "wall_seconds", "ok", "collapsed", "error"]
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            d = asdict(r)
            f.write(",".join(
                f'"{d[c]}"' if isinstance(d[c], str) else f"{d[c]}"
                for c in cols) + "\n")
    paths["raw"] = raw_path

    # ---- summary ----
    by_cfg: Dict[str, List[CellResult]] = {}
    for r in results:
        by_cfg.setdefault(r.config_key, []).append(r)

    order = [c.key for c in WARMSTART_LADDER]
    keys = sorted(by_cfg, key=lambda k: order.index(k) if k in order else 99)

    sum_path = os.path.join(outdir, f"{tag}_summary.csv")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("config_key,config_label,n_seeds,ppo_mean_eur,ppo_std_eur,"
                "ppo_share_mean,ppo_share_std,bc_mean_eur,milp_mean_eur,"
                "random_mean_eur\n")
        for k in keys:
            rs = [r for r in by_cfg[k] if r.ok]
            pm, ps, n = _agg([r.ppo_eur for r in rs])
            sm, ss, _ = _agg([r.ppo_share_of_milp for r in rs])
            bm, _, _ = _agg([r.bc_eur for r in rs])
            mm, _, _ = _agg([r.milp_discrete_eur for r in rs])
            rm, _, _ = _agg([r.random_eur for r in rs])
            label = rs[0].config_label if rs else by_cfg[k][0].config_label
            f.write(f"{k},{label},{n},{pm},{ps},{sm},{ss},{bm},{mm},{rm}\n")
    paths["summary"] = sum_path

    # ---- markdown table for the paper ----
    md_path = os.path.join(outdir, f"{tag}_table.md")
    ok_all = [r for r in results if r.ok]
    milp_m, milp_s, _ = _agg([r.milp_discrete_eur for r in ok_all])
    bc_m, bc_s, _ = _agg([r.bc_eur for r in ok_all])
    rnd_m, rnd_s, _ = _agg([r.random_eur for r in ok_all])

    def fmt(mean, std, n=None):
        if math.isnan(mean):
            return "n/a"
        if std is None or math.isnan(std):
            return f"{mean/1e6:.2f}"
        return f"{mean/1e6:.2f} ± {std/1e6:.2f}"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Warm-start ablation ({tag})\n\n")
        f.write(f"Generated {datetime.now():%Y-%m-%d %H:%M}. "
                f"Profits in M EUR over the held-out window, "
                f"mean ± sample standard deviation across seeds.\n\n")
        f.write("| Config | Init | LR | Entropy | KL anchor | "
                "Profit argmax [M EUR] | Profit sampled [M EUR] | "
                "Share of MILP | n |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        f.write(f"| Discrete MILP benchmark | – | – | – | – | "
                f"{fmt(milp_m, milp_s)} | – | 1.000 | {len(ok_all)} |\n")
        f.write(f"| Behavioural clone (no RL) | – | – | – | – | "
                f"{fmt(bc_m, bc_s)} | "
                f"– | {bc_m/milp_m:.3f} | {len(ok_all)} |\n"
                if not math.isnan(milp_m) and abs(milp_m) > 1e-9 else
                f"| Behavioural clone (no RL) | – | – | – | – | "
                f"{fmt(bc_m, bc_s)} | – | n/a | {len(ok_all)} |\n")
        for k in keys:
            rs = [r for r in by_cfg[k] if r.ok]
            if not rs:
                continue
            cfg = next((c for c in WARMSTART_LADDER if c.key == k), None)
            pm, ps, n = _agg([r.ppo_eur for r in rs])
            sm, ss, _ = _agg([r.ppo_share_of_milp for r in rs])
            share = ("n/a" if math.isnan(sm)
                     else (f"{sm:.3f}" if math.isnan(ss)
                           else f"{sm:.3f} ± {ss:.3f}"))
            if cfg is None:
                f.write(f"| {rs[0].config_label} | | | | | "
                        f"{fmt(pm, ps)} | – | {share} | {n} |\n")
            else:
                sm2, ss2, _ = _agg([r.ppo_eur_stochastic for r in rs])
                f.write(
                    f"| {cfg.label} | "
                    f"{'BC' if cfg.warm else 'random'} | "
                    f"{cfg.lr_scale:g}x | "
                    f"{'base' if cfg.entropy is None else f'{cfg.entropy:g}'} | "
                    f"{'yes' if cfg.anchor else 'no'} | "
                    f"{fmt(pm, ps)} | {fmt(sm2, ss2)} | {share} | {n} |\n")
        f.write(f"| Random baseline | – | – | – | – | "
                f"{fmt(rnd_m, rnd_s)} | – | – | {len(ok_all)} |\n")

        f.write("\n## Reading the ladder\n\n")
        f.write("Each row adds one factor to the row above. The difference "
                "between `PPO vanilla` and `PPO + BC init` is the effect "
                "attributable to behavioural cloning alone, which is the "
                "quantity the paper previously claimed without isolating "
                "it.\n")
    paths["table"] = md_path

    # ---- learning curves ----
    cur_path = os.path.join(outdir, f"{tag}_curves.json")
    with open(cur_path, "w", encoding="utf-8") as f:
        json.dump({f"{r.config_key}_seed{r.seed}": r.learning_curve
                   for r in results if r.learning_curve}, f)
    paths["curves"] = cur_path

    print("\nreports written:")
    for k, v in paths.items():
        print(f"  {k:<8} {v}")
    return paths


# ===========================================================================
# tau_FCR sensitivity
# ===========================================================================

def run_tau_sweep(
    *,
    fleet,
    train_start,
    train_days: int,
    test_days: int,
    outdir: str,
    taus=TAU_FCR_SWEEP,
    seeds: Optional[List[int]] = None,
    ppo_iterations: int = 0,
    real_pun_xlsx_path: Optional[str] = None,
    real_msd_xlsx_paths: Optional[List[str]] = None,
    milp_mode: str = "discrete",
    extra_pipeline_kwargs: Optional[Dict[str, Any]] = None,
) -> List[CellResult]:
    """tau_FCR sensitivity (Reviewer 2 pt 2, Reviewer 1 pt 2).

    By default this runs MILP + BC + random only (ppo_iterations=0 skips the
    PPO variants), because the question it answers is whether the market mix
    and the concentration on downward aFRR survive a correct FCR sustain
    window. That is a property of the environment and the optimizer, and it
    does not need a trained policy to demonstrate.
    """
    seeds = seeds or [0]
    extra_pipeline_kwargs = extra_pipeline_kwargs or {}
    from marl_pipeline import run_full_pipeline

    os.makedirs(outdir, exist_ok=True)
    out: List[CellResult] = []
    for tau in taus:
        sustain = SustainDurations.with_tau_fcr(tau)
        for seed in seeds:
            key = f"tau{tau:g}"
            path = _cell_path(outdir, key, seed, "tau")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    out.append(CellResult(**json.load(f)))
                print(f"  tau_FCR={tau:g}h seed={seed}  CACHED")
                continue

            print(f"\n=== tau_FCR = {tau:g} h  (seed {seed}) "
                  f"{'[outside SO GL]' if not sustain.is_sogl_compliant() else ''}")
            r = CellResult(config_key=key, config_label=f"tau_FCR={tau:g}h",
                           seed=seed, sustain_label=sustain.label,
                           tau_fcr=float(tau))
            t0 = time.time()
            try:
                res = run_full_pipeline(
                    fleet=fleet, train_start=train_start,
                    train_days=train_days, test_days=test_days,
                    master_seed=seed, sustain=sustain,
                    run_ppo_vanilla=False,
                    run_ppo_bc_warmstart=bool(ppo_iterations),
                    ppo_iterations=ppo_iterations or 1,
                    directional_services=True,
                    real_pun_xlsx_path=real_pun_xlsx_path,
                    real_msd_xlsx_paths=real_msd_xlsx_paths,
                    milp_mode=milp_mode,
                    **extra_pipeline_kwargs,
                )
                r.milp_discrete_eur = float(res.test_total_milp_profit_eur)
                r.bc_eur = float(res.test_total_bc_profit_eur)
                r.random_eur = float(res.test_total_random_profit_eur)
                if ppo_iterations:
                    r.ppo_eur = float(
                        res.test_ppo_bc_warmstart_profit_eur or float("nan"))
                r.ok = True
            except Exception as exc:
                r.error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            r.wall_seconds = time.time() - t0
            r.finalise()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(r), f, indent=2)
            out.append(r)
    return out


# ===========================================================================
# CLI
# ===========================================================================

def _build_fleet(smoke: bool):
    from marl_milp_continuous import MultiBatteryParameters
    from milp_optimizer import BatteryParameters
    if smoke:
        bp = BatteryParameters(capacity_mwh=2.0, max_power_mw=1.0,
                               degradation_cost_per_mwh=25000.0,
                               cycle_life=8000)
        return MultiBatteryParameters([bp] * 3)
    return MultiBatteryParameters.heterogeneous_fleet(seed=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("warmstart", "tau_fcr"),
                    default="warmstart")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="random seeds (3-5 recommended; R2 asks for 3-5)")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="subset of config keys, e.g. V I W")
    ap.add_argument("--ppo-iterations", type=int, default=200)
    ap.add_argument("--train-days", type=int, default=365)
    ap.add_argument("--test-days", type=int, default=365)
    ap.add_argument("--train-start", default="2023-01-01")
    ap.add_argument("--outdir", default="ablation_results")
    ap.add_argument("--pun-xlsx", default=None,
                    help="GME price workbook; omit to use the synthetic series")
    ap.add_argument("--msd-xlsx", nargs="*", default=None)
    ap.add_argument("--milp-mode", choices=("continuous", "discrete"),
                    default="discrete")
    ap.add_argument("--sampled-training", action="store_true",
                    help="train on sampled Bernoulli draws instead of "
                         "expected rewards (Reviewer 2 pt 8, Reviewer 1 pt 4)")
    ap.add_argument("--tau", type=float, default=None,
                    help="tau_FCR for the warm-start ablation "
                         "(default: SO GL 0.25 h)")
    ap.add_argument("--smoke", action="store_true",
                    help="3 units, 10 days, 2 PPO iterations: plumbing only")
    args = ap.parse_args(argv)

    train_start = datetime.strptime(args.train_start, "%Y-%m-%d")
    train_days, test_days = args.train_days, args.test_days
    ppo_iters = args.ppo_iterations
    seeds = args.seeds
    if args.smoke:
        train_days, test_days, ppo_iters, seeds = 10, 5, 2, [0]
        print("[smoke] 3 units, 10 train / 5 test days, 2 PPO iters, 1 seed")

    fleet = _build_fleet(args.smoke)
    sustain = (SustainDurations.with_tau_fcr(args.tau) if args.tau
               else DEFAULT_SUSTAIN)

    if args.mode == "tau_fcr":
        res = run_tau_sweep(
            fleet=fleet, train_start=train_start,
            train_days=train_days, test_days=test_days,
            outdir=args.outdir, seeds=seeds,
            ppo_iterations=0 if not args.smoke else ppo_iters,
            real_pun_xlsx_path=args.pun_xlsx,
            real_msd_xlsx_paths=args.msd_xlsx,
            milp_mode=args.milp_mode,
        )
        write_reports(res, args.outdir, tag="tau")
        return 0

    configs = WARMSTART_LADDER
    if args.configs:
        wanted = set(args.configs)
        configs = tuple(c for c in WARMSTART_LADDER if c.key in wanted)
        if not configs:
            print(f"no config matched {args.configs}; "
                  f"available: {[c.key for c in WARMSTART_LADDER]}")
            return 2

    res = run_ablation(
        configs=configs, seeds=seeds, fleet=fleet,
        train_start=train_start, train_days=train_days, test_days=test_days,
        sustain=sustain, ppo_iterations=ppo_iters, outdir=args.outdir,
        real_pun_xlsx_path=args.pun_xlsx,
        real_msd_xlsx_paths=args.msd_xlsx,
        expected_reward_training=(not args.sampled_training),
        milp_mode=args.milp_mode,
    )
    write_reports(res, args.outdir, tag="warmstart")

    failed = [r for r in res if not r.ok]
    if failed:
        print(f"\n{len(failed)} cell(s) failed:")
        for r in failed:
            print(f"  {r.config_key} seed={r.seed}: {r.error}")
        return 1

    collapsed = [r for r in res if r.collapsed]
    if collapsed:
        print(f"\n{len(collapsed)} of {len(res)} cell(s) COLLAPSED to the "
              f"no-op policy:")
        for r in collapsed:
            print(f"  {r.config_key} seed={r.seed}")
        print("  Do not read the summary table: configurations that all "
              "collapse are indistinguishable, so the ablation measures "
              "nothing. Fix training first.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
