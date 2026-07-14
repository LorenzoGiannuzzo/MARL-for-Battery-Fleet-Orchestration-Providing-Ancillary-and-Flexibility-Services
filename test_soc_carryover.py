"""
Test for SOC carry-over across PPO training episodes
(marl_trainer._WindowSamplingMultiBESSEnv).

marl_env / ray are not importable here, so this replicates the patched reset()
logic on a faithful stub of MultiBESSEnv, reproducing the two details that
matter: reset() overwrites _socs with soc_init (line 407) and builds the
observations FROM that reset SOC (line 433). Keep as a regression guard.
"""
import numpy as np

N_AGENTS = 3
SOC_INIT = 0.5


class FakeWindow:
    def __init__(self, n_days=365):
        self.n_days = n_days
    def slice_day(self, d):
        return [100.0 * d + h for h in range(24)], [f"svc{d}"] * 24


class FakeBaseEnv:
    """Stub of MultiBESSEnv: only the reset/step state that matters here."""
    def __init__(self, seed=None, soc_init=SOC_INIT, **kw):
        self._default_seed = seed
        self.soc_init = soc_init
        self.n_agents = N_AGENTS
        self.possible_agents = [f"bess_{i}" for i in range(N_AGENTS)]
        self.agents = list(self.possible_agents)
        self._fixed_prices = kw.get("prices")
        self._fixed_services = kw.get("services")
        self._rng = np.random.default_rng(seed)
        self._socs = np.full(N_AGENTS, soc_init, dtype=np.float64)
        self.fce_reset_count = 0

    def _build_observation(self, a):
        i = self.possible_agents.index(a)
        return np.array([self._socs[i], self._hour], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self._hour = 0
        self._socs = np.full(N_AGENTS, self.soc_init, dtype=np.float64)  # line 407
        self.fce_reset_count += 1                                        # LFP rebuild
        self._prices_episode = list(self._fixed_prices)
        obs = {a: self._build_observation(a) for a in self.agents}       # line 433
        return obs, {a: {} for a in self.agents}

    def step_to_end(self, final_socs):
        """Simulate a 24h episode ending at `final_socs`."""
        self._socs = np.array(final_socs, dtype=np.float64)
        self._hour = 24
        self.agents = []


class WindowSampling(FakeBaseEnv):
    """Patched subclass logic, mirrored from marl_trainer.py."""
    def __init__(self, market_window, day_seed=None,
                 carry_soc_across_episodes=True, **kwargs):
        self._market_window = market_window
        self._day_rng = np.random.default_rng(day_seed)
        self._sampled_day = None
        self._carry_soc = bool(carry_soc_across_episodes)
        self._n_resets = 0
        super().__init__(**kwargs)

    def reset(self, seed=None, options=None):
        carried = None
        if self._carry_soc and self._n_resets > 0:
            carried = np.array(self._socs, dtype=np.float64, copy=True)
        d = int(self._day_rng.integers(0, int(self._market_window.n_days)))
        prices_day, services_day = self._market_window.slice_day(d)
        self._fixed_prices = list(prices_day)
        self._fixed_services = list(services_day)
        self._sampled_day = d
        obs, infos = super().reset(seed=seed, options=options)
        if carried is not None:
            self._socs = carried
            obs = {a: self._build_observation(a) for a in self.agents}
        self._n_resets += 1
        return obs, infos


checks = []
win = FakeWindow()

# 1. Cold start: the FIRST episode legitimately begins at soc_init
env = WindowSampling(market_window=win, day_seed=1, seed=0)
obs0, _ = env.reset()
checks.append(("episode 1 (cold start) begins at soc_init=0.5",
               np.allclose(env._socs, SOC_INIT)))

# 2. Episode 2 begins where episode 1 ENDED (not at 0.5)
end_socs = [0.83, 0.12, 0.61]
env.step_to_end(end_socs)
obs1, _ = env.reset()
checks.append((f"episode 2 begins at episode 1's final SOC {end_socs}",
               np.allclose(env._socs, end_socs)))
checks.append(("episode 2 did NOT snap back to 0.5",
               not np.allclose(env._socs, SOC_INIT)))

# 3. THE SUBTLE ONE: observations must reflect the CARRIED soc, not soc_init.
#    super().reset() builds obs from soc_init; if they are not rebuilt the
#    policy acts on a 0.5 that is not the real state.
obs_socs = [float(obs1[a][0]) for a in env.agents]
checks.append((f"observations rebuilt from carried SOC (obs={[round(s,2) for s in obs_socs]})",
               np.allclose(obs_socs, end_socs)))
checks.append(("observations are NOT the stale soc_init=0.5",
               not np.allclose(obs_socs, [SOC_INIT] * N_AGENTS)))

# 4. Carry-over chains across many episodes
env2 = WindowSampling(market_window=win, day_seed=2, seed=0)
env2.reset()
rng = np.random.default_rng(0)
prev = None
chain_ok = True
for k in range(200):
    finals = rng.uniform(0.0, 1.0, N_AGENTS)
    env2.step_to_end(finals)
    o, _ = env2.reset()
    if not np.allclose(env2._socs, finals):
        chain_ok = False
        break
    if not np.allclose([float(o[a][0]) for a in env2.agents], finals):
        chain_ok = False
        break
checks.append(("carry-over + obs rebuild hold over 200 chained episodes", chain_ok))

# 5. Start-SOC distribution is no longer a point mass at 0.5
env3 = WindowSampling(market_window=win, day_seed=3, seed=0)
env3.reset()
starts = []
for k in range(500):
    env3.step_to_end(rng.uniform(0.05, 0.95, N_AGENTS))
    env3.reset()
    starts.append(env3._socs.mean())
starts = np.array(starts)
checks.append((f"start-SOC distribution has spread (std={starts.std():.3f}, "
               f"not a point mass at 0.5)", starts.std() > 0.05))

# 6. Opt-out restores the old behaviour exactly
env4 = WindowSampling(market_window=win, day_seed=4, seed=0,
                      carry_soc_across_episodes=False)
env4.reset()
env4.step_to_end([0.9, 0.9, 0.9])
o4, _ = env4.reset()
checks.append(("carry_soc=False -> old behaviour (resets to soc_init)",
               np.allclose(env4._socs, SOC_INIT)
               and np.allclose([float(o4[a][0]) for a in env4.agents], SOC_INIT)))

# 7. FCE/LFP states are still rebuilt every episode (deliberately NOT carried)
checks.append((f"FCE/LFP rebuilt every episode ({env4.fce_reset_count} rebuilds "
               f"in {env4._n_resets} resets) -> no runaway ageing",
               env4.fce_reset_count == env4._n_resets))

# 8. Day sampling still works alongside carry-over
env5 = WindowSampling(market_window=win, day_seed=5, seed=0)
ds = []
for _ in range(1000):
    env5.reset()
    ds.append(env5._sampled_day)
    env5.step_to_end(rng.uniform(0, 1, N_AGENTS))
checks.append((f"day sampling intact with carry-over ({len(set(ds))} distinct days)",
               len(set(ds)) > 250))

print("SOC carry-over across training episodes")
print("=" * 70)
all_ok = True
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_ok = all_ok and ok
print("=" * 70)
print("ALL PASSED" if all_ok else "SOME FAILED")
raise SystemExit(0 if all_ok else 1)
