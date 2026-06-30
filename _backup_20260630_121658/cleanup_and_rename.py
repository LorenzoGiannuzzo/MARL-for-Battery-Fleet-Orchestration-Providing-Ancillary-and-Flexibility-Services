#!/usr/bin/env python3
"""
Cleanup + rename utility for the Full-Foresight MARL branch.

What it does, in order:
  1. BACKUP the whole branch into ./_backup_<timestamp>/ (always, unless --no-backup).
  2. DELETE the 16 legacy single-BESS files unreachable from run_paper_experiment.py
     (the old main.py world and its satellites).
  3. RENAME the surviving multi-agent core files to a consistent `marl_` scheme,
     and rewrite every `import` / `from ... import` across the kept files so
     nothing breaks. The single-BESS MILP stays `milp_optimizer.py` (project rule).
  4. NEUTRALISE the dead `generate_expert_demos` continuous-projection demo path
     inside the (renamed) BC module, so BC can ONLY ever learn from the native
     discrete MILP via pipeline_step8.generate_multi_day_expert_demos.

SAFETY:
  * Dry-run by default. Nothing is touched until you pass --apply.
  * Run it FROM INSIDE the branch directory.
  * Re-runnable: rename map is idempotent (skips files already renamed).

Usage:
    python cleanup_and_rename.py                 # dry-run, prints the plan
    python cleanup_and_rename.py --apply         # do it (with backup)
    python cleanup_and_rename.py --apply --no-backup
    python cleanup_and_rename.py --apply --no-rename   # only delete legacy
"""

from __future__ import annotations

import argparse
import datetime
import re
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 1) Files to DELETE: the legacy single-BESS world (main.py + satellites).
#    These are unreachable from run_paper_experiment.py (verified by import-graph
#    tracing). They reference each other but nothing the paper pipeline needs.
# ---------------------------------------------------------------------------
LEGACY_DELETE = [
    "main.py",
    "bc_pretraining.py",              # single-BESS BC (multi version is separate)
    "extended_ppo_environment.py",
    "extended_action_space.py",
    "export_functionality.py",
    "performance_metrics.py",
    "plotting.py",
    "visualization_system.py",
    "statistical_analysis_reporting.py",
    "sensitivity_analysis.py",
    "forecast_error_calibration.py",
    "forecast_error_plots.py",
    "market_data_integration.py",
    "market_data_loader.py",
    "ppo_comparison_patch.py",
    "personalized_run_stard.py",
]

# Optional: legacy non-.py artefacts of the old pipeline. Commented out by
# default; uncomment if you also want these gone.
LEGACY_DELETE_EXTRAS = [
    "main.diff",
    "main_block3.diff",
    "milp_optimizer.diff",
    "drl_flexibility_analysis.diff",
    # "forecast_error_calibration.pdf",
]


# ---------------------------------------------------------------------------
# 2) RENAME map: old_stem -> new_stem.  Only the multi-agent core is renamed.
#    Project rule preserved: milp_optimizer.py (single-BESS) is NOT renamed.
#    flexibility_market / italian_market_* / degradation_model / seeding /
#    commitments stay as-is (shared infrastructure, already well-named).
# ---------------------------------------------------------------------------
RENAME = {
    "milp_optimizer_multi":          "marl_milp_continuous",
    "milp_optimizer_multi_discrete": "marl_milp_discrete",
    "multi_bess_env":                "marl_env",
    "bc_pretraining_multi":          "marl_bc",
    "bc_kl_anchor":                  "marl_bc_kl_anchor",
    "pipeline_step8":                "marl_pipeline",
    "analysis_step8":                "marl_analysis",
    "extra_charts":                  "marl_charts",
    "mappo_trainer":                 "marl_trainer",
    "ppo_comparison":                "marl_ppo_comparison",
    "ppo_convergence":               "marl_ppo_convergence",
    "run_paper_experiment":          "marl_run_experiment",
}

# Files that are kept but NOT renamed (so the import-rewriter leaves their
# module name alone, but still rewrites references inside them).
KEEP_UNRENAMED = [
    "milp_optimizer",          # project rule: single-BESS MILP keeps its name
    "flexibility_market",
    "italian_market_config",
    "italian_market_data",
    "degradation_model",
    "seeding",
    "commitments",
    "drl_flexibility_analysis",  # imported (locally) by milp_optimizer
]


def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def backup(root: Path) -> Path:
    dest = root / f"_backup_{timestamp()}"
    dest.mkdir(exist_ok=False)
    for p in root.iterdir():
        if p.name.startswith("_backup_"):
            continue
        if p.is_file():
            shutil.copy2(p, dest / p.name)
    return dest


def rewrite_imports_in_text(text: str, rename: dict[str, str]) -> str:
    """Rewrite `import X`, `from X import ...`, and bare module refs `X.` for
    every X in the rename map. Word-boundary anchored to avoid partial hits
    (e.g. milp_optimizer must not match milp_optimizer_multi)."""
    # Sort longest-first so milp_optimizer_multi_discrete is handled before
    # milp_optimizer_multi before milp_optimizer.
    for old in sorted(rename, key=len, reverse=True):
        new = rename[old]
        # from <old> import ...
        text = re.sub(rf'(\bfrom\s+){re.escape(old)}(\s+import\b)',
                      rf'\1{new}\2', text)
        # import <old>   (optionally  as alias)
        text = re.sub(rf'(\bimport\s+){re.escape(old)}\b',
                      rf'\1{new}', text)
        # module-qualified use:  <old>.something
        text = re.sub(rf'\b{re.escape(old)}(\.)',
                      rf'{new}\1', text)
        # bare mentions in comments / docstrings (e.g. "saved by pipeline_step8")
        # Only rewrite when followed by a word boundary that is NOT '_' so we
        # never turn milp_optimizer into marl_milp_continuous_multi etc.
        text = re.sub(rf'\b{re.escape(old)}(?![\w])',
                      rf'{new}', text)
    return text


def neutralise_dead_bc_demo(text: str) -> str:
    """Guard the dead continuous-projection demo generator so it can never be
    used by mistake. We do not delete it (keeps git history readable); we make
    it raise immediately, documenting that BC must learn from the native
    discrete MILP via the pipeline instead."""
    marker = "def generate_expert_demos("
    if marker not in text:
        return text
    guard = (
        "def generate_expert_demos(*args, **kwargs):\n"
        "    raise RuntimeError(\n"
        "        \"generate_expert_demos (continuous-MILP projection path) is \"\n"
        "        \"DISABLED. BC must learn from the NATIVE discrete MILP via \"\n"
        "        \"marl_pipeline.generate_multi_day_expert_demos with \"\n"
        "        \"milp_mode='discrete'.\")\n"
        "\n"
        "def _deprecated_generate_expert_demos("
    )
    return text.replace(marker, guard, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually perform changes (default: dry-run)")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the backup copy")
    ap.add_argument("--no-rename", action="store_true",
                    help="only delete legacy files, do not rename")
    ap.add_argument("--delete-extras", action="store_true",
                    help="also delete legacy .diff artefacts")
    ap.add_argument("--root", default=".",
                    help="branch directory (default: current dir)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    print(f"Branch root: {root}")
    dry = not args.apply
    tag = "[DRY-RUN] " if dry else ""

    # --- backup ---
    if args.apply and not args.no_backup:
        b = backup(root)
        print(f"Backup created: {b}")
    elif dry:
        print(f"{tag}would create backup ./_backup_<ts>/")

    # --- delete legacy ---
    to_delete = list(LEGACY_DELETE)
    if args.delete_extras:
        to_delete += LEGACY_DELETE_EXTRAS
    print("\n--- DELETE legacy ---")
    for name in to_delete:
        p = root / name
        if p.exists():
            print(f"{tag}delete {name}")
            if args.apply:
                p.unlink()
        else:
            print(f"  (absent) {name}")

    # --- rename + import rewrite ---
    if not args.no_rename:
        print("\n--- RENAME core + rewrite imports ---")
        # 1) rename the physical files
        for old, new in RENAME.items():
            src = root / f"{old}.py"
            dst = root / f"{new}.py"
            if src.exists():
                print(f"{tag}rename {old}.py -> {new}.py")
                if args.apply:
                    src.rename(dst)
            elif dst.exists():
                print(f"  (already renamed) {new}.py")
            else:
                print(f"  (missing) {old}.py")

        # 2) rewrite imports inside ALL surviving .py files
        survivors = set(RENAME.values()) | set(KEEP_UNRENAMED)
        print(f"\n{tag}rewriting imports in {len(survivors)} kept modules...")
        if args.apply:
            for stem in survivors:
                f = root / f"{stem}.py"
                if not f.exists():
                    continue
                txt = f.read_text(encoding="utf-8")
                new_txt = rewrite_imports_in_text(txt, RENAME)
                if stem == RENAME.get("bc_pretraining_multi", "marl_bc"):
                    new_txt = neutralise_dead_bc_demo(new_txt)
                if new_txt != txt:
                    f.write_text(new_txt, encoding="utf-8")
                    print(f"  rewrote imports in {stem}.py")
    else:
        print("\n--- RENAME skipped (--no-rename) ---")

    print("\nDone." if args.apply else "\nDry-run complete. Re-run with --apply to execute.")
    if args.apply:
        print("Next: `python -m py_compile *.py` then run the smoke scale.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
