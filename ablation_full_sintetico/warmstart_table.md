# Warm-start ablation (warmstart)

Generated 2026-08-16 16:25. Profits in M EUR over the held-out window, mean ± sample standard deviation across seeds.

| Config | Init | LR | Entropy | KL anchor | Profit argmax [M EUR] | Profit sampled [M EUR] | Share of MILP | n |
|---|---|---|---|---|---|---|---|---|
| Discrete MILP benchmark | – | – | – | – | 7.28 | – | 1.000 | 1 |
| Behavioural clone (no RL) | – | – | – | – | 6.31 | – | 0.867 | 1 |
| PPO warm start (full) | BC | 0.1x | 0 | yes | 6.03 | 5.64 | 0.828 | 1 |
| Random baseline | – | – | – | – | 0.55 | – | – | 1 |

## Reading the ladder

Each row adds one factor to the row above. The difference between `PPO vanilla` and `PPO + BC init` is the effect attributable to behavioural cloning alone, which is the quantity the paper previously claimed without isolating it.
