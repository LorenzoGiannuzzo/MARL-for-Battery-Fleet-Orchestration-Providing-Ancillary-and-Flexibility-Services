"""
Test for the real-market-window wiring in marl_trainer._WindowSamplingMultiBESSEnv.

marl_env / ray / pettingzoo are not importable here, so this replicates the
EXACT reset() logic of the patched subclass on a faithful stub of the base
class, including the detail that MultiBESSEnv.reset() re-seeds `self._rng`
from `_default_seed` on EVERY episode. Keep as a regression guard.
"""
import numpy as np

# ---- Fake MarketWindow: 365 days, each with a distinguishable price level ----
class FakeWindow:
    def __init__(self, n_days=365):
        self.n_days = n_days
    def slice_day(self, d):
        prices = [100.0 * d + h for h in range(24)]        # day d is identifiable
        services = [f"svc_day{d}_h{h}" for h in range(24)]
        return prices, services


# ---- Faithful stub of MultiBESSEnv (the parts that matter) ----
class FakeBaseEnv:
    def __init__(self, seed=None, **kw):
        self._default_seed = seed
        self._fixed_prices = kw.get("prices")
        self._fixed_services = kw.get("services")
        self._rng = np.random.default_rng(seed)
        self.episode_hours = 24
    def reset(self, seed=None, options=None):
        # THE CRITICAL DETAIL: re-seeds self._rng every single episode.
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._default_seed is not None:
            self._rng = np.random.default_rng(self._default_seed)
        self._prices_episode = (list(self._fixed_prices)
                                if self._fixed_prices is not None
                                else ["SYNTHETIC_SINUSOID"] * 24)
        return {}, {}


# ---- The PATCHED subclass logic, copied verbatim from marl_trainer.py ----
class WindowSampling(FakeBaseEnv):
    def __init__(self, market_window, day_seed=None, **kwargs):
        self._market_window = market_window
        self._day_rng = np.random.default_rng(day_seed)
        self._sampled_day = None
        super().__init__(**kwargs)
    def reset(self, seed=None, options=None):
        d = int(self._day_rng.integers(0, int(self._market_window.n_days)))
        prices_day, services_day = self._market_window.slice_day(d)
        self._fixed_prices = list(prices_day)
        self._fixed_services = list(services_day)
        self._sampled_day = d
        return super().reset(seed=seed, options=options)


# ---- The BUGGY variant (shares self._rng) — proves the design choice matters --
class WindowSamplingSharedRng(FakeBaseEnv):
    def __init__(self, market_window, **kwargs):
        self._market_window = market_window
        super().__init__(**kwargs)
    def reset(self, seed=None, options=None):
        d = int(self._rng.integers(0, int(self._market_window.n_days)))
        prices_day, services_day = self._market_window.slice_day(d)
        self._fixed_prices = list(prices_day)
        self._sampled_day = d
        return super().reset(seed=seed, options=options)


checks = []
win = FakeWindow(365)
N_EP = 2000

# 1. Patched env: samples MANY distinct real days across episodes
env = WindowSampling(market_window=win, day_seed=7919, seed=0)
days = []
for _ in range(N_EP):
    env.reset()
    days.append(env._sampled_day)
uniq = len(set(days))
checks.append((f"patched: {uniq}/365 distinct days over {N_EP} episodes (>300)",
               uniq > 300))

# 2. Prices actually come from the sampled REAL day (never the sinusoid)
ok_prices = all(
    env._prices_episode[0] == 100.0 * d
    for d, env._sampled_day in [(days[-1], days[-1])]
)
env.reset()
d = env._sampled_day
ok_prices = (env._prices_episode == [100.0 * d + h for h in range(24)])
checks.append(("patched: episode prices == sampled real day's prices", ok_prices))
checks.append(("patched: synthetic sinusoid never used",
               "SYNTHETIC_SINUSOID" not in env._prices_episode))

# 3. Day coverage is uniform (no seasonal bias).
# NB: over 2000 episodes ~1-2 days legitimately never come up (Poisson: each
# day has e^-5.5 ~ 0.4% chance of zero draws), so min>=1 is the WRONG assertion.
# Test uniformity properly instead: (a) with enough episodes every day is hit,
# (b) the spread matches the Poisson std expected under a uniform sampler.
counts = np.bincount(days, minlength=365)
lam = N_EP / 365
spread_ok = abs(counts.std() - np.sqrt(lam)) < 0.6 * np.sqrt(lam)
checks.append((f"patched: spread matches uniform sampler "
               f"(std={counts.std():.2f}, Poisson expects ~{np.sqrt(lam):.2f})",
               spread_ok))

env_cov = WindowSampling(market_window=win, day_seed=7919, seed=0)
cov_days = []
for _ in range(20000):
    env_cov.reset()
    cov_days.append(env_cov._sampled_day)
cov = np.bincount(cov_days, minlength=365)
checks.append((f"patched: all 365 real days covered over 20k episodes "
               f"(min={cov.min()}, max={cov.max()})", cov.min() >= 1))

# 4. THE BUG the separate RNG avoids: sharing self._rng collapses to ONE day
buggy = WindowSamplingSharedRng(market_window=win, seed=0)
bdays = []
for _ in range(200):
    buggy.reset()
    bdays.append(buggy._sampled_day)
checks.append((f"shared-rng variant collapses to {len(set(bdays))} day(s) "
               f"-> separate _day_rng is REQUIRED", len(set(bdays)) == 1))

# 5. Reproducibility: same day_seed -> same day sequence
e1 = WindowSampling(market_window=win, day_seed=7919, seed=0)
e2 = WindowSampling(market_window=win, day_seed=7919, seed=0)
s1 = [(e1.reset(), e1._sampled_day)[1] for _ in range(50)]
s2 = [(e2.reset(), e2._sampled_day)[1] for _ in range(50)]
checks.append(("reproducible: same day_seed -> identical day sequence", s1 == s2))

# 6. Different env seed -> different day stream (vanilla vs warm decorrelated)
e3 = WindowSampling(market_window=win, day_seed=1 + 7919, seed=1)
s3 = [(e3.reset(), e3._sampled_day)[1] for _ in range(50)]
checks.append(("different seed -> different day stream", s1 != s3))

# 7. Fallback: no window -> base env yields the synthetic sinusoid (the old bug)
base = FakeBaseEnv(seed=0)
base.reset()
checks.append(("no window -> synthetic sinusoid (the bug being fixed)",
               base._prices_episode == ["SYNTHETIC_SINUSOID"] * 24))

print("Real-market-window sampling test")
print("=" * 66)
all_ok = True
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_ok = all_ok and ok
print("=" * 66)
print("ALL PASSED" if all_ok else "SOME FAILED")
raise SystemExit(0 if all_ok else 1)
