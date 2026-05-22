# Battery Energy Storage System (BESS) — PPO vs MILP Comparison

Sistema di confronto tra algoritmi PPO (Proximal Policy Optimization, deep reinforcement learning) e MILP (Mixed Integer Linear Programming) per l'ottimizzazione di sistemi di accumulo elettrochimico operanti sul mercato italiano dell'energia (PUN) e dei servizi di flessibilità (FCR, aFRR, mFRR) secondo le regole ARERA.

---

## Recent fixes (review 2026-05)

Una review del codice ha individuato tre bug che rendevano il confronto MILP vs PPO strutturalmente sbilanciato. I file in questa cartella contengono i fix applicati. Le righe modificate sono marcate con un commento `BUG FIX:` per facilitare la ricostruzione delle differenze rispetto al codice originale.

1. **Sign convention del profitto di arbitraggio.** In `extended_ppo_environment.py` e `enhanced_ppo_environment.py` la formula `arbitrage_profit = arbitrage_energy * true_price` aveva il segno invertito rispetto alla convenzione di `Battery.step()` (dove `energy > 0` significa caricare, quindi è un costo). La formula corretta, allineata a `BatteryTradingEnvClean` in `drl_flexibility_analysis.py`, è `arbitrage_profit = -arbitrage_energy * true_price`. Con il bug, il PPO veniva addestrato a caricare a prezzi alti e scaricare a prezzi bassi, ovvero a perdere soldi a ogni ciclo.

2. **Costo di degradazione asimmetrico tra MILP e PPO.** Il PPO calcolava `cost = MWh * 25000 / (2 * 6000) ≈ 2.083 EUR/MWh` di throughput (LCOS marginale ammortizzato su 6000 cicli equivalenti), mentre il MILP usava direttamente `cost = MWh * 25000 = 25000 EUR/MWh`, sovrastimando la degradazione di un fattore 12000. Risultato: il MILP era fortemente disincentivato a usare la batteria, mentre il PPO la usava liberamente. È stato aggiunto il parametro `cycle_life: int = 6000` a `BatteryParameters` e la formula del MILP è ora `cost = MWh * degradation_cost_per_mwh / (2 * cycle_life)`, identica a quella del PPO env.

3. **Il "PPO" del ComparisonEngine non era PPO.** `comparison_engine.run_ppo_episode` chiamava sempre `_select_greedy_action`, una euristica price-threshold hand-coded, mai sostituita da un modello PPO addestrato. In più gli indici di azione erano invertiti rispetto al loro commento, quindi anche la euristica caricava ad alto prezzo e scaricava a basso prezzo. Il `ComparisonEngine` ora accetta un argomento opzionale `ppo_model_path` e, se fornito, usa il modello PPO addestrato (via `stable_baselines3.PPO.load`) per produrre le azioni durante gli episodi. Il greedy fallback rimane disponibile come baseline esplicitamente etichettata "non PPO" e con gli indici di azione corretti.

### Problemi noti non ancora risolti

- `fair_annual_comparison.py` è vuoto (0 bytes). Era probabilmente il file in cui i fix sopra dovevano essere consolidati in un'unica pipeline annuale "fair".
- I market constraints di capacità minima ARERA (`milp_optimizer.add_market_constraints`, righe ~280-292) sono ancora dei `pass` vuoti. Per essere conformi a ARERA il MILP dovrebbe imporre `R_fcr ≥ min_capacity` quando `R_fcr > 0`, il che richiede una formulazione big-M con variabili binarie aggiuntive.
- I dati di mercato sono completamente sintetici (sinusoide oraria + rumore). `market_data_integration.py` definisce le strutture `PUNData` e `MSDTariff` ma le pipeline di confronto non le usano: integrare dati storici reali dal GME è un task aperto.
- `extended_action_space.py` non è mai importato (la stessa logica è duplicata inline negli env). È codice morto.
- Coesistono due environment quasi-equivalenti (`ExtendedBatteryTradingEnv` 31 azioni / 35 features e `EnhancedBatteryTradingEnv` 15 azioni / 40 features con reward shaping). Vanno consolidati o motivati separatamente nei paper futuri.

---

## Struttura del progetto

### File principali (componenti core)

- `milp_optimizer.py`: ottimizzatore MILP single-shot e rolling horizon basato su PuLP + CBC. Gestisce arbitraggio energetico e prenotazioni FCR/aFRR/mFRR simultanee.
- `extended_ppo_environment.py`: ambiente Gymnasium con observation space a 35 feature (SOC + 24 forecast prices + current price + 9 flexibility features) e action space a 31 azioni discrete (21 arbitraggio + 4 FCR + 3 aFRR + 3 mFRR), con risoluzione conflitti configurabile (revenue/arbitrage/equal priority).
- `enhanced_ppo_environment.py`: variante con reward shaping orientato a FCR, action space ridotto a 15 azioni e observation space arricchito con feature di revenue density.
- `flexibility_market.py`: modello del mercato italiano dei servizi di flessibilità (FCR, aFRR, mFRR) con tariffe e probabilità di attivazione ARERA, moltiplicatori orari e stagionali.
- `drl_flexibility_analysis.py`: contiene `Battery` class, `ForecastErrorGenerator` (uniform, normal, Ornstein-Uhlenbeck, fixed-bias), il modello polinomiale di degradazione `degradation(cycle_num)`, e l'env originale `BatteryTradingEnvClean` a 26 feature.

### File di confronto e analisi

- `comparison_engine.py`: framework di confronto MILP vs PPO con shared datasets per garantire equità. Ora supporta caricamento di modelli PPO addestrati via parametro `ppo_model_path`.
- `improved_annual_comparison.py`: pipeline annuale giorno-per-giorno con pre-training PPO di 2 mesi e ritraining mensile. Usa `ExtendedBatteryTradingEnv`.
- `enhanced_annual_comparison.py`: stessa logica della precedente ma con `EnhancedBatteryTradingEnv` e curriculum learning.
- `ppo_vs_milp_comparison.py`: confronto semplificato con una euristica `SimplePPOSimulator` come baseline.

### File di training

- `enhanced_ppo_pretraining.py`: pipeline di curriculum learning a 4 stadi (simple → moderate → complex → fine-tuning) più creazione di modelli mensili specializzati.

### Configurazione e utilities

- `italian_market_config.py`: configurazioni centralizzate del mercato italiano (timing, vincoli tecnici ARERA, tariffe per servizio).
- `extended_action_space.py`: classe stand-alone per encoding/decoding delle 31 azioni (non usata attualmente nelle pipeline).
- `market_data_integration.py`: strutture dati per PUN, MSD, BSP costs (non integrate nelle pipeline).
- `performance_metrics.py`, `visualization_system.py`, `statistical_analysis_reporting.py`, `sensitivity_analysis.py`, `export_functionality.py`: utilities di reporting e analisi.

### Entry points

- `run_comparison.py`: esegue un confronto base via `ComparisonEngine`.
- `improved_annual_comparison.py` (eseguibile diretto): esegue il confronto annuale completo.

### Directories

- `ppo_models/`: modelli PPO addestrati (`.zip` di stable-baselines3)
- `annual_results/`, `results/`, `flexibility_results/`: output simulazioni
- `data/`: dati di input (vuota o sintetica)

---

## Utilizzo

### Confronto annuale completo

```bash
python improved_annual_comparison.py
```

Esegue 2 mesi di pre-training PPO, poi simula 365 giorni con ritraining mensile, confrontando con MILP daily. Tempo stimato su CPU consumer: 6-12 ore per la sola componente PPO.

### Pipeline orchestrata con il nuovo `main.py`

```bash
# Full mode: 4 distribuzioni di forecast error in un singolo run
python main.py --mode full --train-dist normal --eval-dists uniform normal ornstein-uhlenbeck fixed-bias

# Smoke test rapido
python main.py --mode fast

# Solo confronto, riusa un modello PPO già addestrato
python main.py --mode compare --model-path ppo_models/main_ppo.zip
```

Il flag `--train-dist` seleziona la singola distribuzione usata per il pretraining e il monthly retraining del PPO; il flag `--eval-dists` accetta una o piu distribuzioni di valutazione su cui sia il PPO sia il MILP vengono benchmarkati con identica sequenza di forecast errors. Il setup train-once/eval-on-many misura la robustezza out-of-distribution del PPO. Sia MILP sia PPO ricevono prezzi forecast affetti dallo stesso tipo di errore in ogni distribuzione di evaluation.

### Confronto via ComparisonEngine con modello PPO addestrato

```python
from comparison_engine import ComparisonEngine, ComparisonConfig

config = ComparisonConfig(
    time_horizon=168,                            # 1 week
    error_distributions=['uniform', 'normal'],
    n_episodes_per_distribution=10,
)

# Passa il path del modello addestrato: ora il confronto usa PPO vero,
# non il greedy fallback.
engine = ComparisonEngine(
    config,
    ppo_model_path='ppo_models/enhanced_ppo_final.zip',
)
summary = engine.run_comparison(base_prices)
```

Se `ppo_model_path` non viene fornito, il sistema usa un greedy price-threshold come baseline esplicito (non PPO, ma corretto come segno).

### Pre-training PPO standalone

```bash
python enhanced_ppo_pretraining.py
```

Produce `ppo_models/enhanced_ppo_final.zip` più modelli mensili specializzati `ppo_models/enhanced_ppo_month_{1..12}.zip`.

---

## Parametri batteria di default

| Parameter                  | Value      | Notes                                  |
|----------------------------|------------|----------------------------------------|
| capacity_mwh               | 4.0        | Energy capacity                        |
| max_power_mw               | 2.0        | C/2 nominal power                      |
| efficiency                 | 0.95       | Round-trip                             |
| soc_min / soc_max          | 0.1 / 0.9  | Operating window                       |
| degradation_cost_per_mwh   | 25000      | EUR/MWh of replacement capacity        |
| cycle_life                 | 6000       | Equivalent full cycles to end-of-life  |

Costo di degradazione marginale effettivo: `25000 / (2 * 6000) ≈ 2.083 EUR/MWh` di throughput.

---

## Convenzione di segno (importante)

In `Battery.step(action)`:
- `action > 0` significa caricare; ritorna `+actual_power_from_grid` (energia immagazzinata, positiva)
- `action < 0` significa scaricare; ritorna `-actual_power_to_grid` (energia ceduta, negativa)

Quindi il profitto di arbitraggio è sempre `profit = -energy * price`:
- `energy > 0` (charge) → profit < 0 (cost)
- `energy < 0` (discharge) → profit > 0 (revenue)

Questa convenzione è coerente in `drl_flexibility_analysis.py`, nei due nuovi env (dopo il fix 1) e in `ppo_vs_milp_comparison.py`.

---

## Dipendenze principali

- `stable-baselines3` (PPO)
- `gymnasium`
- `pulp` (MILP via solver CBC)
- `numpy`, `pandas`, `scipy`
- `matplotlib` (visualization)
