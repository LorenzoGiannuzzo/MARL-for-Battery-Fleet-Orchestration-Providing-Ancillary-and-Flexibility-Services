# -*- coding: utf-8 -*-
"""
seeding.py — Centralised, reproducible seeding for the multi-agent BSP pipeline.

Why this module exists
-----------------------
The pipeline draws randomness from FIVE independent sources, and a run is only
bit-reproducible if ALL of them are pinned:

  1. Python's `random`            (rarely used directly, but libraries touch it)
  2. NumPy global RNG             (np.random.choice / uniform in policies &
                                   marl_trainer fallback sampling)
  3. PyTorch CPU RNG              (PPO weight init, action sampling)
  4. PyTorch CUDA RNG             (if a GPU is ever used)
  5. RLlib / Ray                  (rollout workers, already via .debugging(seed))

`.debugging(seed=...)` in build_default_config covers (5) for the learner, but
it does NOT touch the process-global (1)-(4), which is why "identical" vanilla
runs drifted (1.25M / 5.05M / 2.42M). Call `set_global_seeds(seed)` ONCE at the
very top of run_full_pipeline (before building any env, model, or config) and
the global generators become deterministic for the whole process.

Determinism caveats (document these in the paper's reproducibility note)
------------------------------------------------------------------------
  * Ray rollout workers are SEPARATE PROCESSES. Each worker seeds itself from
    the RLlib config seed, but OS thread scheduling across workers can still
    reorder sample collection. With num_env_runners=1 (your production setting)
    this is effectively deterministic; with >1 workers expect tiny drift.
  * torch.use_deterministic_algorithms(True) forces deterministic CUDA kernels
    but (a) is slower and (b) raises if an op has no deterministic impl. We
    enable it best-effort and fall back with a warning, so CPU runs (yours) are
    fully deterministic and GPU runs are as-deterministic-as-feasible.
  * CUBLAS needs CUBLAS_WORKSPACE_CONFIG set BEFORE torch initialises CUDA for
    deterministic matmuls; we set it here defensively.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seeds(
    seed: int = 0,
    deterministic_torch: bool = True,
    verbose: bool = True,
) -> int:
    """Pin every global RNG the pipeline can reach. Call ONCE, early.

    Parameters
    ----------
    seed : int
        Master seed. Per-component seeds in the pipeline (env_seed, bc_seed,
        eval_seed, data_seed) should be DERIVED from this deterministically so
        one number reproduces the whole run.
    deterministic_torch : bool
        If True, request deterministic torch algorithms (best-effort).
    verbose : bool
        Print a one-line confirmation.

    Returns
    -------
    int : the seed actually applied (echoed for logging into metrics.json).
    """
    # CUBLAS determinism must be set before CUDA context creation.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Make Python hashing reproducible (affects set/dict iteration order in
    # some library code paths). Note: only takes effect for NEW processes, but
    # setting it is harmless and documents intent.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # Older torch without warn_only kwarg
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
            # cuDNN determinism knobs (no-op on CPU)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    except ImportError:
        pass

    if verbose:
        print(f"[seeding] global seeds pinned to {seed} "
              f"(python/numpy/torch{'/cuda' if _cuda_ok() else ''}); "
              f"deterministic_torch={deterministic_torch}")
    return seed


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def derive_seeds(master_seed: int = 0) -> dict:
    """Deterministically derive the pipeline's per-stage seeds from one master.

    Use these instead of hand-picking eval_seed=100 etc., so a single master
    seed fully determines the run and there are no accidental collisions.

    The offsets are arbitrary but fixed; what matters is that they are distinct
    and reproducible. Returns a dict you can splat into run_full_pipeline.
    """
    # Use a SeedSequence to spawn well-separated, high-quality child seeds.
    ss = np.random.SeedSequence(master_seed)
    children = ss.spawn(5)
    vals = [int(c.generate_state(1)[0]) % (2**31 - 1) for c in children]
    return {
        "data_seed": vals[0],
        "bc_seed": vals[1],
        "eval_seed": vals[2],
        "fleet_heterogeneous_seed": vals[3],
        "ppo_seed": vals[4],
        "master_seed": int(master_seed),
    }
