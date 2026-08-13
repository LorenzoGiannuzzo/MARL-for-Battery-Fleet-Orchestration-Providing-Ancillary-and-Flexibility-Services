# Warm-start ablation (warmstart)

Generated 2026-08-09 13:13. Profits in M EUR over the held-out window, mean ± sample standard deviation across seeds.

| Config | Init | LR | Entropy | KL anchor | Profit argmax [M EUR] | Profit sampled [M EUR] | Share of MILP | n |
|---|---|---|---|---|---|---|---|---|
| Discrete MILP benchmark | – | – | – | – | n/a | – | 1.000 | 0 |
| Behavioural clone (no RL) | – | – | – | – | n/a | – | n/a | 0 |
| Random baseline | – | – | – | – | n/a | – | – | 0 |

## Reading the ladder

Each row adds one factor to the row above. The difference between `PPO vanilla` and `PPO + BC init` is the effect attributable to behavioural cloning alone, which is the quantity the paper previously claimed without isolating it.
