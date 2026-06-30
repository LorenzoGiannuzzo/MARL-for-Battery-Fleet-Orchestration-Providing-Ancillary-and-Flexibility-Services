# Methodology

This document describes the methodological framework implemented in this repository for benchmarking a deep reinforcement learning (DRL) controller against a mixed-integer linear programming (MILP) optimizer on the joint problem of energy arbitrage and ancillary-service provision for a battery energy storage system (BESS) operating in the Italian electricity market. The document is organized around the system being modeled, the two optimization paradigms under comparison, and the experimental protocol that ensures the comparison is statistically meaningful.

## 1. Problem statement

A grid-connected BESS owner faces the joint scheduling problem of deciding, at every hour, how much power to charge or discharge against the wholesale energy market (the Italian PUN, *Prezzo Unico Nazionale*), and how much power capacity to reserve for the three categories of frequency-restoration ancillary services regulated by ARERA, namely the Frequency Containment Reserve (FCR, sub-30-second response), the automatic Frequency Restoration Reserve (aFRR, sub-200-second response), and the manual Frequency Restoration Reserve (mFRR, sub-15-minute response). The capacity reserved for an ancillary service receives a guaranteed *capacity payment* in EUR/MW/h, plus a probabilistic *energy payment* in EUR/MWh contingent on the service being activated by the transmission system operator (TSO, Terna in the Italian case). Reserving capacity for ancillary services and trading on the energy market are mutually constraining because the battery has finite power and state of charge (SOC), and so the BESS owner must co-optimize across the two revenue streams while accounting for battery degradation, which is a function of cumulative energy throughput.

The objective is to maximize the net economic value of the BESS over a yearly horizon under forecast uncertainty on the energy price. A natural baseline is a deterministic MILP given perfect or imperfect forecasts and re-solved at each time step in a receding-horizon manner. A natural alternative is a DRL policy trained to map observations to actions directly, which is more attractive in deployment because it amortizes the optimization cost over training and produces decisions in milliseconds. The contribution of this work is a controlled comparison of the two paradigms under identical market data, identical battery dynamics, and identical forecast-error realizations, designed so that the only difference between the two algorithms is the decision-making mechanism itself.

## 2. System modeling

### 2.1 Battery model

The battery is parameterized by its nominal energy capacity $C_\mathrm{nom}$ (default 4 MWh), maximum charge/discharge power $P_\mathrm{max}$ (default 2 MW, i.e. a C/2 rate), round-trip efficiency $\eta$ (default 0.95), and operating SOC window $[\mathrm{SOC}_\mathrm{min}, \mathrm{SOC}_\mathrm{max}] = [0.1, 0.9]$. The discrete-time SOC dynamics with hourly resolution are

$$
\mathrm{SOC}_{t+1} = \mathrm{SOC}_t + \frac{\eta\, P^\mathrm{ch}_t - P^\mathrm{dis}_t / \eta}{C(\mathrm{cycles}_t)},
$$

where $P^\mathrm{ch}_t \geq 0$ and $P^\mathrm{dis}_t \geq 0$ are the (mutually exclusive) charge and discharge powers, and $C(\mathrm{cycles}_t)$ is the current usable capacity, which evolves as a function of cumulative equivalent full cycles through the polynomial degradation curve $f(\mathrm{cycles})$ implemented in `drl_flexibility_analysis.degradation()`. Each MWh of energy throughput incurs a marginal degradation cost of $c_\mathrm{deg}/(2 N_\mathrm{cycle})$, where $c_\mathrm{deg}$ is the replacement-capacity cost (default 25000 EUR/MWh) and $N_\mathrm{cycle}$ is the cycle life to end-of-warranty (default 6000 full cycles). The factor 2 in the denominator accounts for the fact that one full cycle moves $2 C_\mathrm{nom}$ MWh of energy through the battery. With the default parameters, this works out to roughly 2.08 EUR/MWh of throughput, consistent with the levelized cost of storage (LCOS) for current lithium-iron-phosphate systems. Both the MILP and the DRL agent use this identical formulation, which is one of the points where the original implementation diverged and was corrected during the 2026 review.

### 2.2 Energy market model

The wholesale energy market is represented by an hourly price series $\{\lambda^\mathrm{PUN}_t\}$. In the current implementation the price series is generated synthetically as the superposition of a daily sinusoidal component (peak in early evening, trough overnight), an annual sinusoidal component (winter peak, summer trough), a weekend discount factor, and Gaussian noise. The synthetic generator is deterministic given the random seed and is intended as a placeholder for historical PUN data ingested via the `market_data_integration.py` module. Arbitrage profit over a single hour, given a net battery power injection $E_t = P^\mathrm{dis}_t - P^\mathrm{ch}_t$ (positive when discharging to the grid), is

$$
\pi^\mathrm{arb}_t = E_t \cdot \lambda^\mathrm{PUN}_t.
$$

### 2.3 Flexibility-service market model

Each ancillary service $s \in \{\mathrm{FCR}, \mathrm{aFRR}, \mathrm{mFRR}\}$ is parameterized by a capacity price $\lambda^c_s$ (EUR/MW/h), an energy price $\lambda^e_s$ (EUR/MWh), a historical activation probability $p^a_s$, a minimum bid capacity (1 MW per ARERA), a maximum service duration, and a TSO response-time requirement (30 s for FCR, 200 s for aFRR, 900 s for mFRR). The base tariffs are multiplied by hourly and seasonal factors that capture peak/off-peak and winter/summer dynamics observed in historical Terna data. The expected revenue from reserving $R_s$ MW of capacity for service $s$ during one hour is

$$
\mathbb{E}[\pi^\mathrm{flex}_{s,t}] = R_s \cdot \lambda^c_{s,t} + 0.5\,R_s\,\lambda^e_{s,t}\,p^a_{s,t},
$$

where the 0.5 factor reflects the assumption that, when activated, the service consumes on average 50% of the reserved capacity. The MILP optimizes against this expectation; the PPO environment uses the same expectation in its reward signal but also draws Monte Carlo samples of the actual activation outcome through `FlexibilityMarket.simulate_service_activation()`, so that the agent observes the stochastic realization rather than the expectation, which is more realistic but also noisier.

A symmetry constraint applies to FCR: providing FCR means committing to deliver power in both directions on demand, so the SOC must remain bounded away from both $\mathrm{SOC}_\mathrm{min}$ and $\mathrm{SOC}_\mathrm{max}$ by at least the energy required to deliver the reserved capacity over the expected activation window. This is encoded as a pair of linear constraints in the MILP and as an availability check in the DRL environment.

### 2.4 Forecast-error model

Decision-making at time $t$ is based not on the realized future prices but on a forecast that carries error. Four forecast-error distributions are implemented in `ForecastErrorGenerator`:

The *uniform* distribution draws errors as $\varepsilon_t \sim U(-0.15, +0.15)$, the *normal* distribution draws $\varepsilon_t \sim \mathcal{N}(0, 0.05)$ clipped to $[-0.15, +0.15]$, the *Ornstein-Uhlenbeck* process generates temporally correlated errors $d\varepsilon_t = -\theta \varepsilon_t \, dt + \sigma\, dW_t$ with $\theta=0.15$ and $\sigma=0.2$, and the *fixed-bias* distribution returns a constant +10% error for systematic-bias stress tests. The observed forecast price at time $t+h$ is $\hat{\lambda}_{t+h} = \lambda_{t+h}(1 + \varepsilon_{t+h})$, and the same error sequence is shared between MILP and PPO when both are run on the same episode so that the comparison is paired rather than independent.

## 3. MILP formulation

The MILP is formulated as a maximization of net economic surplus over a finite horizon $T$ (24 hours by default for the daily-loop variant, 168 hours for the weekly variant, or arbitrary horizons for the rolling-horizon variant). The decision variables are continuous power flows $P^\mathrm{ch}_t, P^\mathrm{dis}_t \in [0, P_\mathrm{max}]$, continuous SOC variables $\mathrm{SOC}_t \in [\mathrm{SOC}_\mathrm{min}, \mathrm{SOC}_\mathrm{max}]$, continuous reservation variables $R^s_t$ and activation variables $A^s_t$ for each service $s$, and binary indicators $B^\mathrm{ch}_t, B^\mathrm{dis}_t$ enforcing mutual exclusion of charge and discharge.

The objective is

$$
\max \sum_{t=1}^{T} \Big[ (P^\mathrm{dis}_t - P^\mathrm{ch}_t) \lambda^\mathrm{PUN}_t + \sum_s (R^s_t \lambda^c_s + A^s_t \lambda^e_s p^a_s) - c_\mathrm{deg}^\mathrm{marg} (P^\mathrm{ch}_t + P^\mathrm{dis}_t + \sum_s A^s_t) \Big],
$$

with the SOC evolution equation, power-balance constraints $P^\mathrm{ch}_t + \sum_s R^s_t \leq P_\mathrm{max}$ and analogously for discharge, mutual-exclusion constraints $B^\mathrm{ch}_t + B^\mathrm{dis}_t \leq 1$ linked to the continuous power variables via big-M, activation upper bounds $A^s_t \leq R^s_t$, and the FCR-symmetry SOC reserves $\mathrm{SOC}_t \geq \mathrm{SOC}_\mathrm{min} + R^\mathrm{FCR}_t \cdot 0.5 / C_\mathrm{nom}$ and analogously for the upper bound. The problem is solved with PuLP wrapping the CBC solver. ARERA minimum-bid constraints (capacity floor of 1 MW per service) are not yet enforced in the current implementation; closing this gap requires additional big-M constraints linking $R^s_t$ to a service-participation binary, which is left as an open task.

The MILP is offered in two flavors. The single-shot variant in `MILPOptimizer.optimize()` solves the entire horizon at once and is used as the "perfect intra-horizon information" benchmark. The rolling-horizon variant in `MILPOptimizer.optimize_rolling_horizon()` re-solves at each step over a moving window with optional re-optimization when forecast errors exceed a threshold, which is the operationally realistic counterpart that competes with the PPO under matched information conditions.

## 4. DRL formulation

The DRL approach frames the daily scheduling problem as a finite-horizon Markov decision process and trains a stochastic policy via Proximal Policy Optimization (PPO) with the implementation provided by `stable-baselines3`.

### 4.1 State space

The state at time $t$ is a 35-dimensional vector comprising the current SOC, the 24-hour forecast of energy prices (each price normalized by 200 EUR/MWh, which spans the historical PUN range), the current energy price, and a 9-dimensional flexibility block that encodes, for each of the three services, an availability flag and the normalized capacity and energy prices. The "enhanced" variant in `EnhancedBatteryTradingEnv` extends the state to 40 dimensions by adding revenue-density indicators (expected EUR/MW/h for each service), an ordinal ranking of the three services by revenue density, peak-hour and weekend indicators, current reservations, available residual power, and pairwise FCR/aFRR and FCR/mFRR ratios. The richer state is meant to give the agent a more direct signal about which service is more profitable in the current hour, at the cost of needing slightly more samples to fit the value function.

### 4.2 Action space

The action space is a discrete set of 31 actions in the extended environment. The first 21 actions correspond to arbitrage power levels uniformly spaced in $[-P_\mathrm{max}, +P_\mathrm{max}]$, encoding pure charge/discharge decisions. Actions 21 to 24 reserve 0%, 25%, 50%, or 75% of the battery's power for FCR; actions 25 to 27 reserve 0%, 50%, or 100% for aFRR; actions 28 to 30 do the same for mFRR. The "enhanced" environment uses a smaller 15-action space biased toward FCR-heavy strategies, designed to accelerate convergence on the empirically dominant policy at the cost of expressive power. Because the action space is structured as a single discrete categorical with one component active at a time, conflicts between arbitrage and flexibility cannot arise at the action-encoding level. They can however arise dynamically when the cumulative power demand across services and arbitrage exceeds the battery's nominal power, and are resolved by a configurable conflict-resolution strategy (revenue-priority, arbitrage-priority, or equal-priority scaling) applied after action decoding.

### 4.3 Reward function

The instantaneous reward is the realized economic surplus over the hour,

$$
r_t = -E_t \lambda^\mathrm{PUN}_t + \sum_s \pi^\mathrm{flex}_{s,t} - c_\mathrm{deg}^\mathrm{marg} |E_t| - \mathbb{1}\{\mathrm{SOC}\notin[\mathrm{SOC}_\mathrm{min},\mathrm{SOC}_\mathrm{max}]\}\cdot \kappa_\mathrm{soc} - \kappa_\mathrm{conf} \cdot \Delta_\mathrm{conflict},
$$

where the first term is the arbitrage profit with the corrected sign convention (charging at price $\lambda$ is a cost, discharging is a revenue), the second term sums realized capacity and (Monte Carlo) energy payments across services, the third term is the marginal degradation cost shared with the MILP, and the last two are penalties for violating SOC bounds (with $\kappa_\mathrm{soc} = 1000$ EUR per unit of SOC violation) and for being forced to scale down the requested action because of internal power conflicts. The "enhanced" environment adds five auxiliary reward-shaping terms (FCR-preference bonus, efficiency bonus for high utilization, consistency bonus for sustained profitable strategies, diversification penalty for picking suboptimal services, and SOC management bonus for staying in the $[0.4, 0.6]$ range that maximizes future flexibility), which are useful during pre-training but should be ablated when evaluating final performance against the MILP, because they bias the policy away from the economic objective the MILP optimizes.

### 4.4 Training protocol

The agent is trained with curriculum learning across four stages of increasing complexity. Stage 1 uses 200000 timesteps of simple sinusoidal price patterns with low volatility. Stage 2 uses 300000 timesteps of moderate-complexity prices that match the synthetic generator's full pattern. Stage 3 uses 600000 timesteps split across the three stochastic forecast-error distributions (normal, uniform, Ornstein-Uhlenbeck) at 200000 timesteps each, with the agent's environment switching between them while preserving the policy parameters. Stage 4 fine-tunes for 500000 timesteps on a full-year realistic price trajectory with a reduced learning rate. The total training budget is approximately 1.6 million environment steps, which corresponds to roughly 3300 episodes of 480 hours each. The PPO hyperparameters used are the stable-baselines3 defaults except for the entropy coefficient, which is set to 0.05 during pre-training to encourage broad exploration of the action space, and the policy network, which uses a 256/256/128 multilayer perceptron with $\tanh$ activations.

In addition to the curriculum pre-training, a *monthly transfer learning* protocol is applied during the annual simulation. At the beginning of each month, the agent's environment is reset to the most recent two months of price data and the agent is trained for an additional 100000 timesteps. This is intended to model an operational deployment in which the agent is periodically retrained on recent market data to track non-stationarity, and is included in the comparison protocol because it is the standard practice that a DRL controller would be subjected to in production.

## 5. Comparison methodology

The two algorithms are compared under a paired-sample experimental design. For each forecast-error distribution and each episode index, the same realized price sequence and the same realized forecast errors are supplied to both algorithms. The MILP solves the optimization problem using the noisy forecasts as if they were the truth (which is standard practice when MILP is used in receding horizon), and the PPO observes the noisy forecasts as part of its state. Both produce a sequence of actions, both pay the realized degradation cost on their actual energy throughput, and both collect the same stochastic activation outcomes for any flexibility service they reserve.

The primary outcome variable is the cumulative net profit over the horizon. Secondary outcomes include the per-hour execution time (which is by construction many orders of magnitude smaller for PPO inference than for MILP solving), the battery utilization measured as the ratio of cumulative throughput to maximum possible throughput, the end-of-horizon state of health, the revenue breakdown across arbitrage, FCR, aFRR, and mFRR, and the SOC and reservation trajectories. Statistical significance of the profit difference is assessed with Welch's t-test (which does not assume equal variances) and the Mann-Whitney U test (which does not assume normality). Robustness is quantified by computing the coefficient of variation of mean profit across the four error distributions, which captures how much each algorithm's performance depends on the assumed forecast-error model.

## 6. Annual simulation protocol

The annual simulation in `improved_annual_comparison.py` (and orchestrated end-to-end by `main.py`) implements a realistic deployment scenario over 365 days. After the curriculum pre-training, the simulation iterates day by day. On each day, the MILP is solved over the 24-hour horizon using that day's forecasts; in parallel, the PPO agent rolls out a 24-step episode on the same data. Both algorithms update their respective battery state-of-health proxies based on the day's cumulative throughput, using a calendar-plus-cycle aging model that accumulates $10^{-4}$ per day of calendar aging plus $5\cdot10^{-5}$ per MWh of cycle aging, floored at SOH = 0.8 (end of warranty). Every 30 days, the PPO agent is retrained for one cycle of monthly transfer learning, while the MILP receives the new (degraded) battery parameters via the `RealisticBatteryModel`.

The output of the annual simulation is a per-day dataframe containing the realized profit and revenue breakdown for each algorithm, the cumulative profit trajectory, the SOH evolution, the battery utilization, and the per-service reservation utilization. From this dataframe, the reporting layer produces summary statistics, profit-distribution histograms, cumulative-profit time series, and per-service revenue stack plots, all of which are saved under a single named subdirectory of `results/` for each run.

## 7. Implementation notes and caveats

Three implementation issues were identified during the 2026 review and have been corrected in the version of the code accompanying this document. First, the sign convention for arbitrage profit in `extended_ppo_environment.py` and `enhanced_ppo_environment.py` was inverted relative to the convention used by `Battery.step()`, with the result that the agent was being trained to charge at high prices and discharge at low prices. Second, the marginal degradation cost in `milp_optimizer.py` was a factor of 12000 too large because the amortization factor $1/(2 N_\mathrm{cycle})$ was missing, causing the MILP to be heavily disincentivized from using the battery. Third, the `ComparisonEngine.run_ppo_episode` method always used a hand-coded greedy heuristic as a placeholder, so the "PPO" results were never produced by an actual PPO; the engine now accepts a `ppo_model_path` argument and loads a trained `stable_baselines3.PPO` agent. After these fixes, the comparison is methodologically sound. Remaining open tasks include enforcing ARERA minimum-bid constraints in the MILP via big-M formulation, replacing the synthetic price generator with historical PUN/MSD data via `market_data_integration.py`, and consolidating the duplicated environment classes into a single configurable implementation.
