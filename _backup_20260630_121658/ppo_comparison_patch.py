# ============================================================================
# PATCH per pipeline_step8.run_full_pipeline
# Incolla questo blocco DOPO la valutazione del random baseline
# (dopo `random_eval = evaluate_policy_multi_day(...)`) e PRIMA del
# `return FullPipelineResult(...)`.
# ============================================================================

# --- import in cima al file ---
from ppo_comparison import train_ppo_policy, make_rllib_policy_fn

# --- nuovi parametri della firma di run_full_pipeline ---
#     ppo_iterations: int = 200,
#     run_ppo_vanilla: bool = True,
#     run_ppo_bc_warmstart: bool = True,

# --- corpo, dopo random_eval ---
ppo_results = {}

if run_ppo_vanilla:
    print(f"[pipeline] PPO vanilla: {ppo_iterations} iters on train window...")
    algo_v = train_ppo_policy(
        fleet, n_iterations=ppo_iterations, bc_net=None,
        use_nonlinear_degradation=use_nonlinear_degradation, seed=eval_seed,
    )
    print(f"[pipeline] PPO vanilla on test window...")
    ppo_v_eval = evaluate_policy_multi_day(
        fleet, test_window,
        policy_fn=make_rllib_policy_fn(algo_v),
        use_nonlinear_degradation=use_nonlinear_degradation,
        initial_soc=test_initial_soc,
        initial_fce_cumulative=test_initial_fce,
        env_seed=eval_seed + 1000,
    )
    ppo_results["ppo_vanilla"] = ppo_v_eval
    algo_v.stop()

if run_ppo_bc_warmstart:
    print(f"[pipeline] PPO BC-warmstart: {ppo_iterations} iters on train window...")
    algo_w = train_ppo_policy(
        fleet, n_iterations=ppo_iterations, bc_net=bc_net,
        use_nonlinear_degradation=use_nonlinear_degradation, seed=eval_seed,
    )
    print(f"[pipeline] PPO BC-warmstart on test window...")
    ppo_w_eval = evaluate_policy_multi_day(
        fleet, test_window,
        policy_fn=make_rllib_policy_fn(algo_w),
        use_nonlinear_degradation=use_nonlinear_degradation,
        initial_soc=test_initial_soc,
        initial_fce_cumulative=test_initial_fce,
        env_seed=eval_seed + 1000,
    )
    ppo_results["ppo_bc"] = ppo_w_eval
    algo_w.stop()

# --- decomposizione per servizio (la metrica che conta per il gap aFRR) ---
# evaluate_policy_multi_day restituisce già total_arbitrage / total_flex_revenue
# / total_degradation: stampale per ogni politica per vedere DOVE sta il gap.
print("\n" + "=" * 70)
print("CONFRONTO A QUATTRO — env-realised, apples-to-apples")
print("=" * 70)
print(f"  MILP objective (continuo):  {test_demo.total_milp_profit:>14,.2f} EUR  <- vero upper bound")
print(f"  MILP realised (discreto):   {test_demo.total_env_realized_profit:>14,.2f} EUR")
print(f"  BC policy:                  {bc_eval.total_profit_eur:>14,.2f} EUR")
for tag, ev in ppo_results.items():
    print(f"  {tag:<26s}{ev.total_profit_eur:>14,.2f} EUR")
print(f"  Random baseline:            {random_eval.total_profit_eur:>14,.2f} EUR")
print("\n  Decomposizione per servizio (arbitraggio / flex / degradazione):")
def _row(tag, ev):
    print(f"    {tag:<24s} arb {ev.total_arbitrage:>12,.0f} | "
          f"flex {ev.total_flex_revenue:>12,.0f} | deg {ev.total_degradation:>12,.0f}")
_row("BC", bc_eval)
for tag, ev in ppo_results.items():
    _row(tag, ev)
_row("random", random_eval)
print("=" * 70)

# --- aggiungi questi campi al FullPipelineResult / metrics.json ---
#   test_ppo_vanilla_profit_eur = ppo_results.get("ppo_vanilla").total_profit_eur if "ppo_vanilla" in ppo_results else None
#   test_ppo_bc_profit_eur      = ppo_results.get("ppo_bc").total_profit_eur if "ppo_bc" in ppo_results else None
#   test_milp_objective_eur     = test_demo.total_milp_profit   # <-- ESPORTA ANCHE QUESTO
# e nel CSV per-giorno aggiungi le colonne ppo_*_eur usando ev.daily_profits.
