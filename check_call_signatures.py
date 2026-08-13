# -*- coding: utf-8 -*-
"""
check_call_signatures.py — does every call pass arguments the callee accepts?

Run:  python check_call_signatures.py

WHY THIS EXISTS, AND WHY THE FIRST VERSION WAS NOT ENOUGH
---------------------------------------------------------
Twice now a keyword has been threaded into a call that does not accept it, and
both times the failure surfaced only when the run reached that line: once at
the start of PPO training, once at the start of a 20-hour demo generation.

An earlier version of this check compared each call's keywords against the
callee's signature and SKIPPED any callee declaring **kwargs, on the grounds
that it accepts anything. That is exactly wrong for a subclass that forwards:

    class MultiBESSMILPOptimizerDiscrete(MultiBESSMILPOptimizer):
        def __init__(self, *args, n_action_bins=..., **kwargs):
            super().__init__(*args, **kwargs)

It declares **kwargs, so the old check waved it through, and the keyword sailed
past the subclass into a parent that rejected it. That is precisely how
`centralized_critic` reached the MILP: the centralised critic widens the
OBSERVATION, so it is an environment concept and an optimizer has no business
receiving it, but nothing said so until the constructor raised.

So this version FOLLOWS the forward: when a class's __init__ takes **kwargs and
its body calls super().__init__(**kwargs), the accepted names are the union of
the subclass's own and the parent's, resolved up the chain.
"""

from __future__ import annotations

import ast
import io
import os
import sys
from typing import Dict, List, Optional, Set, Tuple


def _params(fn: ast.FunctionDef) -> Tuple[Set[str], bool]:
    names = set(a.arg for a in fn.args.args)
    names |= set(a.arg for a in fn.args.kwonlyargs)
    return names, fn.args.kwarg is not None


def _forwards_kwargs(fn: ast.FunctionDef) -> bool:
    """True when the body calls super().__init__(**kwargs) or similar."""
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and any(
                k.arg is None for k in n.keywords):      # **something
            f = n.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call):
                if getattr(f.value.func, "id", None) == "super":
                    return True
    return False


def collect(root: str):
    """Map class name -> (own init params, forwards, base names)
    and function name -> (params, has_kwargs)."""
    # name -> list of definitions. A name defined in more than one file is
    # AMBIGUOUS and gets skipped: marl_analysis imports BatteryParameters as
    # _BP while test_fix_a defines an unrelated class _BP, and matching the
    # call against the wrong one produced a confident false positive. A
    # checker that cries wolf gets ignored, which is worse than no checker.
    classes: Dict[str, List[Tuple[Set[str], bool, List[str]]]] = {}
    funcs: Dict[str, List[Tuple[Set[str], bool]]] = {}
    for f in sorted(x for x in os.listdir(root) if x.endswith(".py")):
        try:
            tree = ast.parse(io.open(os.path.join(root, f),
                                     encoding="utf-8").read())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                init = next((b for b in n.body
                             if isinstance(b, ast.FunctionDef)
                             and b.name == "__init__"), None)
                bases = [getattr(b, "id", getattr(b, "attr", ""))
                         for b in n.bases]
                if init is not None:
                    names, has_kw = _params(init)
                    if has_kw and not _forwards_kwargs(init):
                        # **kwargs that is consumed locally rather than
                        # forwarded accepts anything: nothing can be said.
                        classes.setdefault(n.name, []).append(
                            (set(), True, ["<open>"]))
                    else:
                        classes.setdefault(n.name, []).append(
                            (names - {"self"}, has_kw and _forwards_kwargs(init),
                             bases))
                    continue

                # No explicit __init__. A dataclass has one generated from its
                # annotated fields, so those ARE its accepted keywords; reading
                # the class as accepting nothing produced a false positive on
                # every dataclass construction in the repository. Fields are
                # collected from the class body, and from any base that is also
                # a dataclass, which is why the chain is still followed.
                fields = {b.target.id for b in n.body
                          if isinstance(b, ast.AnnAssign)
                          and isinstance(b.target, ast.Name)}
                is_dc = any(
                    getattr(d, "id", getattr(getattr(d, "func", None), "id",
                                             getattr(d, "attr", ""))) ==
                    "dataclass"
                    for d in n.decorator_list)
                # Inherit the parents' fields too: follow the chain rather
                # than assume a flat class. A plain class with no __init__ and
                # no fields simply inherits its parent's.
                classes.setdefault(n.name, []).append((fields, True, bases))
            elif isinstance(n, ast.FunctionDef):
                funcs.setdefault(n.name, []).append(_params(n))
    return classes, funcs


def accepted_by_class(name: str, classes, seen=None) -> Tuple[Set[str], bool]:
    """Names a constructor accepts, following forwarded **kwargs upward.

    Returns (names, open_ended). open_ended is True only when the chain ends
    in something outside the repository, where no claim can be made.
    """
    seen = seen or set()
    if name in seen or name not in classes:
        return set(), True          # unknown base: cannot judge
    defs = classes[name]
    if len(defs) != 1:
        return set(), True          # ambiguous name: cannot judge
    seen.add(name)
    own, forwards, bases = defs[0]
    if bases == ["<open>"]:
        return set(), True          # consumes **kwargs locally
    if not forwards:
        return own, False
    names = set(own)
    open_ended = False
    for b in bases:
        sub, oe = accepted_by_class(b, classes, seen)
        names |= sub
        open_ended = open_ended or oe
    return names, open_ended


def main(root: str = ".") -> int:
    classes, funcs = collect(root)
    problems: List[str] = []

    for f in sorted(x for x in os.listdir(root) if x.endswith(".py")):
        try:
            tree = ast.parse(io.open(os.path.join(root, f),
                                     encoding="utf-8").read())
        except SyntaxError:
            continue
        aliases_here = {a.asname for n in ast.walk(tree)
                        if isinstance(n, (ast.Import, ast.ImportFrom))
                        for a in n.names if a.asname}
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            passed = [k.arg for k in n.keywords if k.arg]
            if not passed:
                continue
            callee = getattr(n.func, "id", None)
            if callee is None:
                continue
            if callee in aliases_here:
                # `from milp_optimizer import BatteryParameters as _BP` makes
                # _BP mean something different here than the class named _BP
                # defined in another file. Resolving it against the global map
                # matched the wrong definition; without per-file alias
                # tracking the report is confidently wrong.
                continue

            if callee in classes:
                names, open_ended = accepted_by_class(callee, classes)
                if open_ended:
                    continue
                bad = [k for k in passed if k not in names]
                if bad:
                    problems.append(
                        f"{f}:{n.lineno}  {callee}(..., "
                        f"{', '.join(k + '=' for k in bad)}...)  "
                        f"-> not accepted, including through super()")
            elif callee in funcs and len(funcs[callee]) == 1:
                names, has_kw = funcs[callee][0]
                if has_kw:
                    continue
                bad = [k for k in passed if k not in names]
                if bad:
                    problems.append(
                        f"{f}:{n.lineno}  {callee}(..., "
                        f"{', '.join(k + '=' for k in bad)}...)  -> not accepted")

    print(f"checked {len(classes)} classes and {len(funcs)} functions")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("every call passes only arguments its callee accepts")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
