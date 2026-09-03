# -*- coding: utf-8 -*-
"""Guard: both MSD profile builders must map the columns the same way.

WHY THIS EXISTS
---------------
The module carries TWO builders for the same profiles, `msd_hourly_profiles`
and `msd_conditioned_profiles`, and only the second is reached in production
(the log line reads "MSD profiles (CONDITIONED ...)"). Correcting the
direction mapping in the first one alone left every number bit-identical, and
the duplication was only found because the results did not move at all.

The mapping itself follows MSD terminology, where the offer type names the
direction and reads the opposite way round to intuition:

    "offerte di VENDITA"  = movimentazione A SALIRE, the provider SELLS
                            energy to Terna and is PAID.
    "offerte di ACQUISTO" = movimentazione A SCENDERE, the provider BUYS
                            energy back and PAYS.

Confirmed on 2023 northern-zone data against a 177 EUR/MWh mean PUN:
Vendita averages 241 (a premium for short-notice energy, so upward) and
Acquisto averages 111 (a discount buyback, so downward). Mapped the other way
round, upward reserve paid 111 against a 177 spot price and no optimizer ever
offered it: aFRR_up and mFRR_up were exactly zero in every run this project
produced, including the submitted paper.
"""
import ast, io, sys

SRC = "italian_market_data.py"
EXPECT = {"msd_hourly_profiles": ("p_sell_mean", "p_buy_mean"),
          "msd_conditioned_profiles": ("p_sell", "p_buy")}

def main() -> int:
    src = io.open(SRC, encoding="utf-8").read()
    tree = ast.parse(src)
    lines = src.split("\n")
    problems = []
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name not in EXPECT:
            continue
        up_col, dn_col = EXPECT[fn.name]
        body = "\n".join(lines[fn.lineno - 1:fn.end_lineno])
        for var, col, other in (("up", up_col, dn_col), ("dn", dn_col, up_col)):
            assigns = [l for l in body.split("\n")
                       if l.strip().startswith((f"{var}_price", f"{var}_p[",
                                                f"{var}_rate", f"{var}_r["))]
            if not assigns:
                problems.append(f"{fn.name}: no {var}_* assignment found")
                continue
            joined = " ".join(assigns)
            if col not in joined:
                problems.append(
                    f"{fn.name}: the {var}ward series does not read {col}")
            if other in joined:
                problems.append(
                    f"{fn.name}: the {var}ward series reads {other}, which is "
                    f"the OTHER direction")
    print(f"checked {len(EXPECT)} MSD profile builders")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  {p}")
        print("\nVendita = upward (provider is paid), "
              "Acquisto = downward (provider pays).")
        return 1
    print("both builders map Vendita to upward and Acquisto to downward")
    return 0

if __name__ == "__main__":
    sys.exit(main())
