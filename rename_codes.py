#!/usr/bin/env python3
"""
Rename the MARL-BESS codebase to professional module names and rewrite every
import / textual reference accordingly. Run this script from INSIDE the folder
that contains the .py files. It renames files on disk and patches all references.

Safe to run once. Make a git commit / backup before running.
"""
import os, re, sys

# old_stem -> new_stem  (NO .py extension)
RENAME = {
    "run_paper_experiment":        "main",
    "pipeline_step8":              "pipeline",
    "analysis_step8":              "analysis",
    "drl_flexibility_analysis":    "legacy_degradation",
    "milp_optimizer_multi":        "milp_fleet",     # MUST come before milp_optimizer
    "milp_optimizer":              "milp_single",
    "multi_bess_env":              "marl_env",
    "mappo_trainer":               "marl_trainer",
    "ppo_comparison":              "marl_ppo",
    "ppo_convergence":             "marl_metrics",
    "bc_pretraining_multi":        "bc",
    "degradation_model":           "degradation",
    "flexibility_market":          "markets",
    "italian_market_data":         "market_data",
    "italian_market_config":       "market_config",
    "extra_charts":                "charts_extra",
    "forecast_error_calibration":  "forecast_calibration",
    "forecast_error_plots":        "forecast_plots",
}

# Order replacements by descending length of the OLD name so that longer names
# (milp_optimizer_multi) are replaced before their prefixes (milp_optimizer).
ordered = sorted(RENAME.items(), key=lambda kv: -len(kv[0]))

def patch_text(text):
    for old, new in ordered:
        # \b word boundary; but '_' is a word char, so milp_optimizer won't match
        # inside milp_optimizer_multi ONLY if we also guard the trailing side.
        # Use negative lookbehind/ahead on word chars to be exact.
        text = re.sub(rf'(?<![\w]){re.escape(old)}(?![\w])', new, text)
    return text

def main():
    pyfiles = [f for f in os.listdir('.') if f.endswith('.py') and f != os.path.basename(__file__)]
    # 1) patch contents of every .py
    for f in pyfiles:
        with open(f, encoding='utf-8') as fh:
            src = fh.read()
        new_src = patch_text(src)
        if new_src != src:
            with open(f, 'w', encoding='utf-8') as fh:
                fh.write(new_src)
            print(f"patched references in: {f}")
    # 2) rename the files themselves
    for old, new in ordered:
        oldf, newf = old + '.py', new + '.py'
        if os.path.exists(oldf):
            os.rename(oldf, newf)
            print(f"renamed: {oldf} -> {newf}")
    print("Done.")

if __name__ == "__main__":
    main()
