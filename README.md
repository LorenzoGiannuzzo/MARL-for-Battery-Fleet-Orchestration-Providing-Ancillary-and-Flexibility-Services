# Battery Energy Storage System (BESS) - PPO vs MILP Comparison

Sistema di confronto avanzato tra algoritmi PPO (Proximal Policy Optimization) e MILP (Mixed Integer Linear Programming) per l'ottimizzazione di sistemi di accumulo energetico nel mercato italiano dei servizi di flessibilità.

## 🏗️ Struttura del Progetto

### File Principali
- **`improved_annual_comparison.py`** - Sistema di confronto annuale migliorato (MAIN)
- **`milp_optimizer.py`** - Ottimizzatore MILP per BESS
- **`extended_ppo_environment.py`** - Ambiente PPO esteso con servizi di flessibilità
- **`flexibility_market.py`** - Modello mercato flessibilità italiano (ARERA)
- **`drl_flexibility_analysis.py`** - Analisi DRL e generatori di errore

### File di Supporto
- **`comparison_engine.py`** - Engine di confronto base
- **`italian_market_config.py`** - Configurazioni mercato italiano
- **`performance_metrics.py`** - Metriche di performance
- **`visualization_system.py`** - Sistema di visualizzazione
- **`statistical_analysis_reporting.py`** - Analisi statistiche
- **`market_data_integration.py`** - Integrazione dati di mercato
- **`export_functionality.py`** - Funzionalità di esportazione
- **`sensitivity_analysis.py`** - Analisi di sensibilità

### File Demo/Test
- **`demo_milp_results.py`** - Demo risultati MILP
- **`run_comparison.py`** - Esecuzione confronto base
- **`simple_milp_test.py`** - Test MILP semplificato

### Directory
- **`.kiro/specs/`** - Specifiche del progetto
- **`annual_results/`** - Risultati simulazioni annuali
- **`ppo_models/`** - Modelli PPO salvati
- **`results/`** - Altri risultati
- **`data/`** - Dati di input

## 🚀 Utilizzo

### Confronto Annuale Completo
```bash
python improved_annual_comparison.py
```

### Demo MILP
```bash
python demo_milp_results.py
```

### Confronto Base
```bash
python run_comparison.py
```

## 📊 Output

Il sistema genera:
- Grafici di confronto dettagliati
- Report di performance per mercato
- Analisi di degrado batteria
- Esportazione CSV dei risultati
- Modelli PPO salvati per riutilizzo

## 🔋 Caratteristiche

- **Mercati Italiani**: FCR, aFRR, mFRR secondo ARERA
- **PPO Realistico**: Pre-training + ritraining mensile
- **Degrado Batteria**: Modello realistico calendario + ciclico
- **Forecast Errors**: Simulazione errori di previsione
- **Analisi Dettagliate**: Breakdown per mercato e algoritmo