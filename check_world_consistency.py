# -*- coding: utf-8 -*-
"""
check_world_consistency.py — does every stage run in the SAME world?

WHY
---
A run has several stages that each build their own environment: the MILP
demonstrations, the clone's evaluation, the random baseline, the PPO
evaluations. Each takes the parameters that define the world it runs in.

Every one of those parameters has a default. That is convenient and it is the
trap: omitting one at a call site does not raise, it silently builds a
DIFFERENT world for that stage, and the comparison the whole paper rests on
quietly stops being a comparison.

This has now happened three times:

  tau_FCR          defined in seven places, three values live at once, so the
                   optimizer planned against 0.25 h while the environment
                   clipped at 4 h;
  connection limit  the MILP planned with a shared connection and the clone
                   was evaluated without one, because `bc_eval` never received
                   the `coupling` argument;
  global state      the clone was trained on 35-wide observations and
                   evaluated on 29-wide ones, which at least crashed instead
                   of quietly scoring the wrong number.

Only the third announced itself. The first two produced plausible numbers.

So the rule enforced here: every call that builds a stage must pass EVERY
world-defining parameter explicitly, defaults notwithstanding. Writing it out
at each site is redundant by design, because the redundancy is what makes an
omission visible.
"""

from __future__ import annotations

import ast
import io
import os
import sys
from typing import List

# Parameters that define the world a stage runs in. A stage that defaults any
# of these is running a different experiment from its siblings.
WORLD_PARAMS = {
    "sustain",              # reserve sustain windows
    "coupling",             # shared connection limit
    "duration_features",    # observation width and content
    "centralized_critic",   # observation width and content
    "relative_features",    # observation width and content
    "directional_services", # which products exist
}

# Functions that build a stage and therefore must receive all of the above.
STAGE_BUILDERS = {
    "evaluate_policy_multi_day",
    "generate_multi_day_expert_demos",
}


def main(root: str = ".") -> int:
    problems: List[str] = []
    checked = 0

    for fname in sorted(x for x in os.listdir(root) if x.endswith(".py")):
        path = os.path.join(root, fname)
        try:
            tree = ast.parse(io.open(path, encoding="utf-8").read())
        except SyntaxError:
            continue

        # Inside the stage builders themselves the parameters are in scope as
        # locals, so a call there is a definition rather than a use.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None)
            if name not in STAGE_BUILDERS:
                continue
            checked += 1
            passed = {k.arg for k in node.keywords if k.arg}
            missing = sorted(WORLD_PARAMS - passed)
            if missing:
                problems.append(
                    f"{fname}:{node.lineno}  {name}(...)  "
                    f"does not pass {', '.join(missing)}  "
                    f"-> this stage would run in a different world")

    print(f"checked {checked} stage-building calls")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  {p}")
        print("\nPass every world parameter explicitly at every site. The "
              "redundancy is the point: a default that silently differs is "
              "how three of this project's worst bugs happened.")
        return 1
    print("every stage is built with the same world parameters")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
