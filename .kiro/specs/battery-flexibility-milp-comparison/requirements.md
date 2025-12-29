# Requirements Document

## Introduction

Questo progetto estende il sistema esistente di trading di batterie con DRL PPO per includere i servizi di flessibilità secondo la normativa italiana, e confronta le performance con un ottimizzatore MILP tradizionale. L'obiettivo è valutare l'efficacia del PPO rispetto a un approccio di ottimizzazione deterministica in uno scenario più complesso che combina arbitraggio energetico e servizi di flessibilità.

## Glossary

- **BESS**: Battery Energy Storage System - Sistema di accumulo energetico a batterie
- **PPO**: Proximal Policy Optimization - Algoritmo di reinforcement learning
- **MILP**: Mixed Integer Linear Programming - Programmazione lineare mista intera
- **Arbitraggio**: Strategia di trading che sfrutta le differenze di prezzo dell'energia elettrica
- **Servizi_di_Flessibilità**: Servizi ancillari forniti al sistema elettrico per bilanciamento e stabilità
- **ARERA**: Autorità di Regolazione per Energia Reti e Ambiente
- **MSD**: Mercato dei Servizi di Dispacciamento
- **BSP**: Balancing Service Provider - Fornitore di servizi di bilanciamento
- **Forecast_Error**: Errore di previsione dei prezzi dell'energia
- **SOC**: State of Charge - Stato di carica della batteria
- **SOH**: State of Health - Stato di salute della batteria
- **Degradation_Model**: Modello di degrado della batteria basato sui cicli di utilizzo

## Requirements

### Requirement 1: Modellazione Servizi di Flessibilità Italiani

**User Story:** Come sviluppatore di sistemi energetici, voglio modellare i servizi di flessibilità secondo la normativa italiana, così da poter valutare le opportunità di revenue aggiuntive per il BESS.

#### Acceptance Criteria

1. WHEN il sistema riceve una richiesta di servizio di flessibilità, THE BESS SHALL rispondere secondo i parametri tecnici ARERA
2. WHEN si attiva un servizio di riserva primaria, THE BESS SHALL fornire la potenza richiesta entro 30 secondi
3. WHEN si attiva un servizio di riserva secondaria, THE BESS SHALL fornire la potenza richiesta entro 200 secondi  
4. WHEN si attiva un servizio di riserva terziaria, THE BESS SHALL fornire la potenza richiesta entro 15 minuti
5. THE Sistema_Flessibilità SHALL calcolare i ricavi secondo le tariffe ARERA vigenti
6. WHEN si verifica un conflitto tra arbitraggio e servizi di flessibilità, THE Sistema_Ottimizzazione SHALL prioritizzare secondo la strategia configurata

### Requirement 2: Implementazione Ottimizzatore MILP

**User Story:** Come ricercatore, voglio implementare un ottimizzatore MILP per il BESS, così da poter confrontare le performance con l'approccio PPO esistente.

#### Acceptance Criteria

1. THE MILP_Optimizer SHALL ottimizzare simultaneamente arbitraggio e servizi di flessibilità
2. WHEN riceve i prezzi di forecast, THE MILP_Optimizer SHALL calcolare la strategia ottima per l'orizzonte temporale definito
3. THE MILP_Optimizer SHALL rispettare tutti i vincoli fisici del BESS (potenza, capacità, SOC)
4. THE MILP_Optimizer SHALL includere il modello di degrado della batteria nella funzione obiettivo
5. WHEN si verificano errori di forecast, THE MILP_Optimizer SHALL ricalcolare la strategia con i nuovi dati
6. THE MILP_Optimizer SHALL supportare rolling horizon optimization con finestre temporali configurabili

### Requirement 3: Estensione Sistema PPO Esistente

**User Story:** Come data scientist, voglio estendere il sistema PPO esistente per includere i servizi di flessibilità, così da mantenere la coerenza con l'implementazione attuale.

#### Acceptance Criteria

1. THE PPO_Extended SHALL mantenere tutte le funzionalità esistenti di arbitraggio
2. WHEN si addestra il modello, THE PPO_Extended SHALL includere i servizi di flessibilità nello spazio delle azioni
3. THE PPO_Extended SHALL utilizzare le stesse distribuzioni di errore di forecast esistenti
4. WHEN calcola la reward, THE PPO_Extended SHALL includere i ricavi da servizi di flessibilità
5. THE PPO_Extended SHALL mantenere il sistema di monthly retraining esistente
6. THE Environment_Extended SHALL includere le opportunità di flessibilità nell'observation space

### Requirement 4: Sistema di Confronto e Valutazione

**User Story:** Come ricercatore, voglio confrontare sistematicamente PPO e MILP, così da quantificare i vantaggi e svantaggi di ciascun approccio.

#### Acceptance Criteria

1. THE Comparison_System SHALL eseguire entrambi gli algoritmi sugli stessi dataset di test
2. WHEN confronta le performance, THE Comparison_System SHALL utilizzare le stesse distribuzioni di errore di forecast
3. THE Comparison_System SHALL calcolare metriche di performance standardizzate per entrambi gli approcci
4. WHEN genera i risultati, THE Comparison_System SHALL produrre grafici comparativi dettagliati
5. THE Comparison_System SHALL analizzare la robustezza agli errori di forecast per entrambi gli algoritmi
6. THE Comparison_System SHALL valutare i tempi di calcolo e la scalabilità di entrambi gli approcci

### Requirement 5: Modellazione Mercati Italiani

**User Story:** Come esperto di mercati energetici, voglio modellare accuratamente i mercati italiani, così da ottenere risultati realistici e applicabili.

#### Acceptance Criteria

1. THE Market_Model SHALL utilizzare i prezzi storici del mercato elettrico italiano (PUN)
2. WHEN modella i servizi di flessibilità, THE Market_Model SHALL utilizzare le tariffe MSD reali
3. THE Market_Model SHALL includere i costi di abilitazione come BSP secondo normativa ARERA
4. WHEN calcola i ricavi, THE Market_Model SHALL applicare le regole di settlement del GME
5. THE Market_Model SHALL considerare i vincoli di partecipazione minimi per ciascun servizio
6. THE Market_Model SHALL modellare la probabilità di attivazione dei servizi basata su dati storici

### Requirement 6: Gestione Incertezza e Robustezza

**User Story:** Come ingegnere di sistema, voglio valutare la robustezza degli algoritmi all'incertezza, così da identificare l'approccio più affidabile in condizioni reali.

#### Acceptance Criteria

1. WHEN testa la robustezza, THE Test_System SHALL utilizzare le quattro distribuzioni di errore esistenti
2. THE Test_System SHALL valutare la performance in scenari di alta volatilità dei prezzi
3. WHEN si verificano errori di forecast estremi, THE Test_System SHALL misurare la degradazione delle performance
4. THE Test_System SHALL analizzare la sensibilità ai parametri di configurazione per entrambi gli algoritmi
5. WHEN confronta gli approcci, THE Test_System SHALL considerare il trade-off tra performance e robustezza
6. THE Test_System SHALL valutare l'adattabilità a cambiamenti nelle condizioni di mercato

### Requirement 7: Visualizzazione e Reporting

**User Story:** Come stakeholder del progetto, voglio visualizzazioni chiare e complete, così da comprendere facilmente i risultati del confronto.

#### Acceptance Criteria

1. THE Visualization_System SHALL generare grafici comparativi per profitti, SOC, e utilizzo batteria
2. WHEN produce i report, THE Visualization_System SHALL includere analisi statistiche dettagliate
3. THE Visualization_System SHALL creare heatmap per visualizzare i pattern di utilizzo del BESS
4. WHEN mostra i risultati di flessibilità, THE Visualization_System SHALL distinguere tra diversi tipi di servizi
5. THE Visualization_System SHALL produrre dashboard interattive per l'analisi dei risultati
6. THE Visualization_System SHALL esportare tutti i grafici in formato PDF ad alta risoluzione

### Requirement 8: Validazione e Testing

**User Story:** Come quality assurance engineer, voglio validare l'accuratezza delle implementazioni, così da garantire risultati affidabili.

#### Acceptance Criteria

1. THE Validation_System SHALL verificare la correttezza del modello MILP attraverso test unitari
2. WHEN testa l'integrazione, THE Validation_System SHALL verificare la coerenza tra PPO esteso e versione originale
3. THE Validation_System SHALL validare i calcoli dei ricavi da servizi di flessibilità contro benchmark noti
4. WHEN esegue i test, THE Validation_System SHALL verificare il rispetto di tutti i vincoli fisici del BESS
5. THE Validation_System SHALL testare la convergenza degli algoritmi in scenari limite
6. THE Validation_System SHALL validare la consistenza dei risultati attraverso multiple esecuzioni