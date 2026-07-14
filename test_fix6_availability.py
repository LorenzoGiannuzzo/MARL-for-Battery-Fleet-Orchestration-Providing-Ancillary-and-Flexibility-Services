"""
Unit test for FIX 6 (availability-scaled capacity payment) in marl_env.py.
Replicates the EXACT math applied inside step()'s Step-D delivery loop and
asserts the intended behaviour on the boundary cases, without needing the
full env dependency stack. Keep as a regression guard.
"""
import numpy as np

# Battery / market constants (match BatteryParameters defaults)
CAP, EFF = 4.0, 0.95
SOC_MIN, SOC_MAX = 0.1, 0.9
CAP_PRICE, MW = 10.0, 2.0
REF_H = {"FCR": 0.5, "aFRR_up": 1.0, "aFRR_dn": 1.0, "mFRR_up": 2.0, "mFRR_dn": 2.0}


def deliverable(direction, soc):
    if direction == "up":
        return max(0.0, (soc - SOC_MIN) * EFF * CAP)
    if direction == "dn":
        return max(0.0, (SOC_MAX - soc) * CAP / EFF)
    return max(0.0, min((soc - SOC_MIN), (SOC_MAX - soc)) * CAP)  # sym / FCR


def net_capacity(service, direction, soc, expected_reward_mode, coeff=1.0, aw=1.0):
    """Returns (cap_pay_full, net_capacity_after_fix)."""
    cap_pay = CAP_PRICE * MW * aw
    if not (expected_reward_mode and MW > 1e-9):
        return cap_pay, cap_pay                       # eval: untouched
    required = MW * REF_H[service]
    avail = float(np.clip(deliverable(direction, soc) / required, 0.0, 1.0)) if required > 1e-9 else 0.0
    pen = coeff * cap_pay * (1.0 - avail)
    return cap_pay, cap_pay - pen


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


checks = []

# 1. FCR mid-band -> full pay (a well-behaved / MILP-like SoC loses nothing)
full, net = net_capacity("FCR", "sym", 0.50, True)
checks.append(("FCR SoC=0.50 -> full pay", approx(net, full)))

# 2. FCR parked high -> zero capacity (symmetric needs down room)
_, net = net_capacity("FCR", "sym", 0.90, True)
checks.append(("FCR SoC=0.90 -> 0 capacity", approx(net, 0.0)))

# 3. FCR parked at 1.0 (clip region) -> zero capacity
_, net = net_capacity("FCR", "sym", 1.00, True)
checks.append(("FCR SoC=1.00 -> 0 capacity", approx(net, 0.0)))

# 4. aFRR_up at high SoC -> full pay (genuinely deliverable: ready to discharge)
full, net = net_capacity("aFRR_up", "up", 0.90, True)
checks.append(("aFRR_up SoC=0.90 -> full pay", approx(net, full)))

# 5. aFRR_up parked low -> capacity collapses (cannot discharge)
full, net = net_capacity("aFRR_up", "up", 0.15, True)
checks.append(("aFRR_up SoC=0.15 -> < 15% pay", net < 0.15 * full))

# 6. aFRR_dn parked high -> capacity collapses (cannot charge)
full, net = net_capacity("aFRR_dn", "dn", 0.90, True)
checks.append(("aFRR_dn SoC=0.90 -> 0 capacity", approx(net, 0.0)))

# 7. EVAL mode -> byte-identical (fix never touches sampled envs)
for svc, d, soc in [("FCR", "sym", 0.90), ("aFRR_up", "up", 0.15), ("FCR", "sym", 0.50)]:
    full, net = net_capacity(svc, d, soc, expected_reward_mode=False)
    checks.append((f"EVAL {svc} SoC={soc} -> unchanged", approx(net, full)))

# 8. Net capacity always within [0, cap_pay] (cannot destabilise training)
rng = np.random.default_rng(0)
ok_bounds = True
for _ in range(2000):
    svc = rng.choice(list(REF_H))
    d = "sym" if svc == "FCR" else ("up" if "up" in svc else "dn")
    soc = float(rng.uniform(0.0, 1.0))
    full, net = net_capacity(svc, d, soc, True, coeff=float(rng.uniform(0, 1)))
    if not (-1e-9 <= net <= full + 1e-9):
        ok_bounds = False
        break
checks.append(("net capacity in [0, cap_pay] over 2000 draws", ok_bounds))

# 9. coeff=0 disables the fix entirely (escape hatch)
full, net = net_capacity("FCR", "sym", 1.00, True, coeff=0.0)
checks.append(("coeff=0 -> fix disabled (full pay)", approx(net, full)))

print("FIX 6 availability-scaling unit test")
print("=" * 52)
all_ok = True
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_ok = all_ok and ok
print("=" * 52)
print("ALL PASSED" if all_ok else "SOME FAILED")
raise SystemExit(0 if all_ok else 1)
