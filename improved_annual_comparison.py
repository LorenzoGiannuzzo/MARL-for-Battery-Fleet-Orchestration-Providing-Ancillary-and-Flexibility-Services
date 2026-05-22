#!/usr/bin/env python3
"""
Sistema di Confronto Annuale PPO vs MILP Migliorato
Con PPO realistico, degrado corretto e training adattivo
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pickle
import os

# Import dei sistemi esistenti
from milp_optimizer import MILPOptimizer, BatteryParameters
from flexibility_market import FlexibilityMarket, FlexibilityService, ServiceType
from extended_ppo_environment import ExtendedBatteryTradingEnv
from drl_flexibility_analysis import ForecastErrorGenerator

# Import per PPO
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
    PPO_AVAILABLE = True
except ImportError:
    print("⚠️ stable-baselines3 non disponibile. Usando simulatore PPO avanzato.")
    PPO_AVAILABLE = False


@dataclass
class ImprovedAnnualConfig:
    """Configurazione migliorata per confronto annuale"""
    start_date: datetime = datetime(2024, 1, 1)
    days: int = 365
    battery_capacity_mwh: float = 4.0
    battery_max_power_mw: float = 2.0
    
    # PPO Training Configuration
    pretrain_months: int = 2  # Mesi di pre-training
    retrain_frequency_months: int = 1  # Ritraining ogni mese
    training_episodes: int = 1000  # Episodi per training
    
    # Degrado batteria realistico
    calendar_aging_per_day: float = 0.0001  # 0.01% al giorno
    cycle_aging_per_mwh: float = 0.00005   # 0.005% per MWh
    
    # Forecast errors
    use_forecast_errors: bool = True
    error_distribution: str = 'normal'
    
    # Output
    save_models: bool = True
    model_dir: str = "ppo_models"
    results_dir: str = "flexibility_results"


class RealisticBatteryModel:
    """Modello batteria realistico con degrado corretto"""
    
    def __init__(self, initial_params: BatteryParameters, config: ImprovedAnnualConfig):
        self.initial_params = initial_params
        self.config = config
        self.current_capacity = initial_params.capacity_mwh
        self.current_efficiency = initial_params.efficiency
        self.soh = 1.0
        self.total_throughput = 0.0
        self.days_elapsed = 0
        
    def update_daily_degradation(self, daily_throughput: float):
        """Aggiorna degrado con modello realistico"""
        # Calendar aging (tempo)
        calendar_degradation = self.config.calendar_aging_per_day
        
        # Cycle aging (uso)
        cycle_degradation = daily_throughput * self.config.cycle_aging_per_mwh
        
        # Aggiorna SOH (non può scendere sotto 0.8)
        total_degradation = calendar_degradation + cycle_degradation
        self.soh = max(0.8, self.soh - total_degradation)
        
        # Aggiorna parametri
        self.current_capacity = self.initial_params.capacity_mwh * self.soh
        self.current_efficiency = max(0.85, self.initial_params.efficiency * (0.95 + 0.05 * self.soh))
        
        # Tracking
        self.total_throughput += daily_throughput
        self.days_elapsed += 1
    
    def get_current_params(self) -> BatteryParameters:
        """Restituisce parametri batteria attuali"""
        return BatteryParameters(
            capacity_mwh=self.current_capacity,
            max_power_mw=self.initial_params.max_power_mw,
            efficiency=self.current_efficiency,
            soc_min=self.initial_params.soc_min,
            soc_max=self.initial_params.soc_max,
            initial_soc=0.5,
            degradation_cost_per_mwh=self.initial_params.degradation_cost_per_mwh
        )


class AdvancedPPOAgent:
    """Agente PPO avanzato con training adattivo"""
    
    def __init__(self, config: ImprovedAnnualConfig):
        self.config = config
        self.model = None
        self.env = None
        self.training_data = []
        
        # Crea directory per modelli
        os.makedirs(config.model_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)
        
    def create_training_environment(self, prices: List[float], 
                                  flexibility_services: List[List[FlexibilityService]],
                                  error_distribution: str = 'normal') -> ExtendedBatteryTradingEnv:
        """Crea ambiente di training"""
        error_gen = ForecastErrorGenerator(error_distribution) if self.config.use_forecast_errors else None
        
        env = ExtendedBatteryTradingEnv(
            prices=prices,
            error_generator=error_gen,
            flexibility_enabled=True,
            conflict_strategy='revenue_priority',
            random_seed=42
        )
        
        return env
    
    def pretrain_agent(self, pretrain_prices: List[float], 
                      pretrain_services: List[List[FlexibilityService]]):
        """Pre-training dell'agente PPO"""
        print(f"🤖 Pre-training PPO per {self.config.pretrain_months} mesi...")
        
        if not PPO_AVAILABLE:
            print("⚠️ Usando simulatore PPO avanzato invece di vero PPO")
            return
        
        # Crea ambiente di training
        env = self.create_training_environment(pretrain_prices, pretrain_services)
        vec_env = DummyVecEnv([lambda: env])
        
        # Inizializza modello PPO con parametri migliorati per esplorazione
        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,  # Aumentato ulteriormente per più esplorazione
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"{self.config.results_dir}/tensorboard/"
        )
        
        # Training
        total_timesteps = self.config.training_episodes * 480  # 480 steps per episode
        self.model.learn(total_timesteps=total_timesteps)
        
        # Salva modello pre-trained
        model_path = f"{self.config.model_dir}/ppo_pretrained.zip"
        self.model.save(model_path)
        print(f"✅ Pre-training completato. Modello salvato: {model_path}")
    
    def retrain_agent(self, recent_prices: List[float], 
                     recent_services: List[List[FlexibilityService]], 
                     month: int):
        """Ritraining mensile dell'agente"""
        print(f"🔄 Ritraining PPO per mese {month}...")
        
        if not PPO_AVAILABLE or self.model is None:
            print("⚠️ Ritraining non disponibile senza modello PPO")
            return
        
        # Crea nuovo ambiente con dati recenti
        env = self.create_training_environment(recent_prices, recent_services)
        vec_env = DummyVecEnv([lambda: env])
        
        # Aggiorna ambiente del modello
        self.model.set_env(vec_env)
        
        # Ritraining con meno episodi
        retrain_timesteps = (self.config.training_episodes // 2) * 480
        self.model.learn(total_timesteps=retrain_timesteps, reset_num_timesteps=False)
        
        # Salva modello aggiornato
        model_path = f"{self.config.model_dir}/ppo_month_{month:02d}.zip"
        self.model.save(model_path)
        print(f"✅ Ritraining completato. Modello salvato: {model_path}")
    
    def predict_action(self, observation: np.ndarray) -> int:
        """Predice azione usando il modello PPO con action masking per diversità"""
        if PPO_AVAILABLE and self.model is not None:
            action, _ = self.model.predict(observation, deterministic=True)
            
            # Action masking per evitare sempre la stessa azione
            if not hasattr(self, '_recent_actions'):
                self._recent_actions = []
            
            # Se l'azione è stata usata troppo di recente, prova alternative
            if len(self._recent_actions) >= 5 and self._recent_actions[-5:].count(int(action)) >= 3:
                # Prova azioni simili
                alternatives = []
                
                if action < 21:  # Arbitraggio
                    # Prova altre azioni arbitraggio
                    alternatives = [max(0, action-2), min(20, action+2)]
                elif 21 <= action <= 24:  # FCR
                    # Prova altre percentuali FCR o arbitraggio
                    alternatives = [21, 22, 23, 24, 10]  # Include arbitraggio 0MW
                elif 25 <= action <= 27:  # aFRR
                    # Prova arbitraggio o FCR
                    alternatives = [10, 22, 24]  # Arbitraggio 0MW, FCR 25%, FCR 75%
                elif 28 <= action <= 30:  # mFRR
                    # Prova arbitraggio o aFRR
                    alternatives = [10, 25, 27]  # Arbitraggio 0MW, aFRR 0%, aFRR 100%
                
                # Scegli alternativa non usata di recente
                for alt in alternatives:
                    if alt not in self._recent_actions[-3:]:
                        action = alt
                        break
            
            # Aggiorna storia
            self._recent_actions.append(int(action))
            if len(self._recent_actions) > 10:
                self._recent_actions.pop(0)
            
            return int(action)
        else:
            # Fallback: strategia euristica avanzata
            return self._advanced_heuristic_action(observation)
    
    def _advanced_heuristic_action(self, obs: np.ndarray) -> int:
        """Strategia euristica avanzata quando PPO non è disponibile"""
        # Estrai informazioni dall'osservazione
        soc = obs[0]
        current_price = obs[25]  # Prezzo corrente normalizzato
        forecast_prices = obs[1:25]  # Prezzi forecast
        
        # Denormalizza prezzi (assumendo normalizzazione 0-200 EUR/MWh)
        current_price_eur = current_price * 200.0
        forecast_avg = np.mean(forecast_prices) * 200.0
        
        # Strategia migliorata: considera sia arbitraggio che flessibilità
        
        # 1. Se prezzi sono stabili e SOC è ragionevole, usa servizi di flessibilità
        price_volatility = np.std(forecast_prices) * 200.0
        
        if price_volatility < 15.0 and 0.3 < soc < 0.7:
            # Bassa volatilità: i servizi di flessibilità potrebbero essere più redditizi
            # Sceglie un servizio di flessibilità in modo neutrale (senza bias)
            # Alterna tra FCR, aFRR e mFRR per dare all'algoritmo la possibilità di imparare
            flexibility_options = [22, 25, 28]  # FCR 25%, aFRR 50%, mFRR 50%
            return np.random.choice(flexibility_options)
        
        # 2. Altrimenti usa logica di arbitraggio esistente
        arbitrage_action = 10  # Default: no action (centro della scala)
        
        # Logica di trading più sofisticata
        if soc < 0.3 and current_price_eur < forecast_avg * 0.85:
            # SOC basso e prezzo conveniente: carica aggressivamente
            arbitrage_action = 2  # -1.6 MW (carica forte)
        elif soc < 0.5 and current_price_eur < forecast_avg * 0.95:
            # SOC medio-basso e prezzo buono: carica moderatamente
            arbitrage_action = 5  # -0.8 MW (carica media)
        elif soc > 0.7 and current_price_eur > forecast_avg * 1.15:
            # SOC alto e prezzo alto: scarica aggressivamente
            arbitrage_action = 18  # +1.6 MW (scarica forte)
        elif soc > 0.5 and current_price_eur > forecast_avg * 1.05:
            # SOC medio-alto e prezzo buono: scarica moderatamente
            arbitrage_action = 15  # +0.8 MW (scarica media)
        
        return arbitrage_action
    
    def simulate_episode(self, daily_prices: List[float], 
                        daily_services: List[List[FlexibilityService]],
                        battery_params: BatteryParameters) -> Dict[str, Any]:
        """Simula un episodio giornaliero con l'agente PPO"""
        start_time = time.time()
        
        # Crea ambiente per simulazione
        env = self.create_training_environment(daily_prices, daily_services)
        
        # Reset ambiente
        obs, _ = env.reset()
        
        # Metriche
        total_arbitrage_profit = 0.0
        total_flexibility_revenue = 0.0
        total_degradation_cost = 0.0
        soc_trajectory = [env.battery.soc]
        power_trajectory = []
        flexibility_reservations = {'FCR': [], 'aFRR': [], 'mFRR': []}
        
        # Breakdown ricavi per servizio
        fcr_revenue = 0.0
        afrr_revenue = 0.0
        mfrr_revenue = 0.0
        
        # Simula episodio
        for step in range(min(len(daily_prices), 24)):
            # Predici azione
            action = self.predict_action(obs)
            
            # Esegui azione
            obs, reward, done, truncated, info = env.step(action)
            
            # Accumula metriche
            total_arbitrage_profit += info.get('arbitrage_profit', 0.0)
            total_flexibility_revenue += info.get('flexibility_revenue', 0.0)
            total_degradation_cost += info.get('degradation_cost', 0.0)
            
            # Estrai breakdown ricavi per servizio dal flexibility_breakdown
            flexibility_breakdown = info.get('flexibility_breakdown', {})
            fcr_revenue += flexibility_breakdown.get('fcr_revenue', 0.0)
            afrr_revenue += flexibility_breakdown.get('afrr_revenue', 0.0)
            mfrr_revenue += flexibility_breakdown.get('mfrr_revenue', 0.0)
            
            # Tracking
            soc_trajectory.append(info.get('soc', env.battery.soc))
            power_trajectory.append(info.get('arbitrage_energy', 0.0))
            
            flexibility_reservations['FCR'].append(info.get('reserved_fcr', 0.0))
            flexibility_reservations['aFRR'].append(info.get('reserved_afrr', 0.0))
            flexibility_reservations['mFRR'].append(info.get('reserved_mfrr', 0.0))
            
            if done or truncated:
                break
        
        execution_time = time.time() - start_time
        
        # Calcola metriche finali
        net_profit = total_arbitrage_profit + total_flexibility_revenue - total_degradation_cost
        
        # Utilizzo batteria
        total_throughput = sum(abs(p) for p in power_trajectory)
        max_possible = battery_params.max_power_mw * len(power_trajectory)
        battery_utilization = total_throughput / max_possible if max_possible > 0 else 0.0
        
        return {
            'algorithm': 'PPO',
            'net_profit': net_profit,
            'arbitrage_profit': total_arbitrage_profit,
            'flexibility_revenue': total_flexibility_revenue,
            'degradation_cost': total_degradation_cost,
            'execution_time': execution_time,
            'battery_utilization': battery_utilization,
            'soc_trajectory': soc_trajectory,
            'power_trajectory': power_trajectory,
            'flexibility_reservations': flexibility_reservations,
            # Breakdown ricavi per servizio
            'fcr_revenue': fcr_revenue,
            'afrr_revenue': afrr_revenue,
            'mfrr_revenue': mfrr_revenue
        }


class ImprovedAnnualComparisonEngine:
    """Engine migliorato per confronto annuale equo"""
    
    def __init__(self, config: ImprovedAnnualConfig):
        self.config = config
        self.flexibility_market = FlexibilityMarket()
        
        # Agente PPO
        self.ppo_agent = AdvancedPPOAgent(config)
        
        # Risultati
        self.daily_results = []
        self.annual_summary = {}
        
    def generate_market_data(self) -> tuple:
        """Genera dati di mercato per l'anno completo + pre-training"""
        print("📊 Generazione dati di mercato...")
        
        # Genera dati per pre-training (2 mesi prima)
        pretrain_days = self.config.pretrain_months * 30
        total_days = pretrain_days + self.config.days
        
        all_prices = []
        all_services = []
        
        for day in range(total_days):
            # Effetto stagionale
            day_of_year = (day - pretrain_days) % 365
            seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
            
            for hour in range(24):
                # Pattern giornaliero
                base_price = 50.0 + 30.0 * np.sin(hour * np.pi / 12)
                seasonal_price = base_price * seasonal_factor
                
                # Rumore e weekend
                noise = np.random.normal(0, 5.0)
                weekend_factor = 0.9 if (day % 7) >= 5 else 1.0
                
                final_price = max(10.0, seasonal_price * weekend_factor + noise)
                all_prices.append(final_price)
                
                # Servizi flessibilità
                seasonal_multiplier = 1.0 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
                hour_services = [
                    FlexibilityService(
                        service_type=ServiceType.FCR,
                        capacity_price=30.0 * seasonal_multiplier,
                        energy_price=60.0,
                        activation_probability=0.1,
                        response_time=30,
                        min_capacity=1.0,
                        max_duration=1,
                        min_duration=1
                    ),
                    FlexibilityService(
                        service_type=ServiceType.AFRR,
                        capacity_price=25.0 * seasonal_multiplier,
                        energy_price=80.0,
                        activation_probability=0.05,
                        response_time=200,
                        min_capacity=1.0,
                        max_duration=4,
                        min_duration=1
                    )
                ]
                all_services.append(hour_services)
        
        # Separa dati pre-training da dati annuali
        pretrain_hours = pretrain_days * 24
        pretrain_prices = all_prices[:pretrain_hours]
        pretrain_services = all_services[:pretrain_hours]
        
        annual_prices = all_prices[pretrain_hours:]
        annual_services = all_services[pretrain_hours:]
        
        print(f"✅ Generati {len(pretrain_prices)} ore pre-training + {len(annual_prices)} ore annuali")
        
        return (pretrain_prices, pretrain_services, annual_prices, annual_services)
    
    def _calculate_milp_service_breakdown(self, milp_result, daily_services: List[List[FlexibilityService]]) -> Dict[str, float]:
        """Calcola il breakdown dei ricavi MILP per servizio usando i prezzi reali"""
        fcr_revenue = 0.0
        afrr_revenue = 0.0
        mfrr_revenue = 0.0
        
        # Per ogni ora del giorno
        for hour in range(min(len(daily_services), 24)):
            services = daily_services[hour]
            service_dict = {s.service_type.value: s for s in services}
            
            # FCR
            if 'FCR' in service_dict and hour < len(milp_result.flexibility_reservations.get('FCR', [])):
                service = service_dict['FCR']
                reserved_capacity = milp_result.flexibility_reservations['FCR'][hour]
                if reserved_capacity > 0:
                    capacity_revenue = reserved_capacity * service.capacity_price
                    energy_revenue = reserved_capacity * service.energy_price * service.activation_probability
                    fcr_revenue += capacity_revenue + energy_revenue
            
            # aFRR
            if 'aFRR' in service_dict and hour < len(milp_result.flexibility_reservations.get('aFRR', [])):
                service = service_dict['aFRR']
                reserved_capacity = milp_result.flexibility_reservations['aFRR'][hour]
                if reserved_capacity > 0:
                    capacity_revenue = reserved_capacity * service.capacity_price
                    energy_revenue = reserved_capacity * service.energy_price * service.activation_probability
                    afrr_revenue += capacity_revenue + energy_revenue
            
            # mFRR
            if 'mFRR' in service_dict and hour < len(milp_result.flexibility_reservations.get('mFRR', [])):
                service = service_dict['mFRR']
                reserved_capacity = milp_result.flexibility_reservations['mFRR'][hour]
                if reserved_capacity > 0:
                    capacity_revenue = reserved_capacity * service.capacity_price
                    energy_revenue = reserved_capacity * service.energy_price * service.activation_probability
                    mfrr_revenue += capacity_revenue + energy_revenue
        
        return {
            'fcr_revenue': fcr_revenue,
            'afrr_revenue': afrr_revenue,
            'mfrr_revenue': mfrr_revenue
        }
    
    def run_improved_annual_comparison(self) -> Dict[str, Any]:
        """Esegue confronto annuale migliorato"""
        print("🚀 CONFRONTO ANNUALE MIGLIORATO PPO vs MILP")
        print("=" * 70)
        print(f"📅 Periodo: {self.config.start_date.strftime('%Y-%m-%d')} - {self.config.days} giorni")
        print(f"🤖 PPO: Pre-training {self.config.pretrain_months} mesi, ritraining ogni {self.config.retrain_frequency_months} mese")
        print(f"🔋 Batteria: {self.config.battery_capacity_mwh} MWh, degrado realistico")
        print()
        
        # Genera dati di mercato
        pretrain_prices, pretrain_services, annual_prices, annual_services = self.generate_market_data()
        
        # Pre-training PPO
        self.ppo_agent.pretrain_agent(pretrain_prices, pretrain_services)
        
        # Inizializza modelli batteria
        initial_battery_params = BatteryParameters(
            capacity_mwh=self.config.battery_capacity_mwh,
            max_power_mw=self.config.battery_max_power_mw,
            efficiency=0.95,
            soc_min=0.1,
            soc_max=0.9,
            initial_soc=0.5,
            degradation_cost_per_mwh=25000.0
        )
        
        milp_battery = RealisticBatteryModel(initial_battery_params, self.config)
        ppo_battery = RealisticBatteryModel(initial_battery_params, self.config)
        
        # Risultati cumulativi
        milp_cumulative_profit = 0.0
        ppo_cumulative_profit = 0.0
        
        print("\n🔄 Inizio simulazione annuale...")
        
        # Simulazione giorno per giorno
        for day in range(self.config.days):
            # Ritraining mensile
            if day > 0 and day % (30 * self.config.retrain_frequency_months) == 0:
                month = day // 30
                # Usa ultimi 2 mesi per ritraining
                retrain_start = max(0, (day - 60) * 24)
                retrain_end = day * 24
                recent_prices = annual_prices[retrain_start:retrain_end]
                recent_services = annual_services[retrain_start:retrain_end]
                
                self.ppo_agent.retrain_agent(recent_prices, recent_services, month)
            
            # Dati giornalieri
            day_start = day * 24
            day_end = (day + 1) * 24
            daily_prices = annual_prices[day_start:day_end]
            daily_services = annual_services[day_start:day_end]
            
            # Parametri batteria attuali
            milp_params = milp_battery.get_current_params()
            ppo_params = ppo_battery.get_current_params()
            
            # Ottimizzazione MILP
            milp_optimizer = MILPOptimizer(milp_params, self.flexibility_market)
            milp_result = milp_optimizer.optimize(24, daily_prices, daily_services)
            
            # Simulazione PPO
            ppo_result = self.ppo_agent.simulate_episode(daily_prices, daily_services, ppo_params)
            
            # Aggiorna degrado batterie
            milp_throughput = sum(abs(p) for p in milp_result.power_trajectory)
            ppo_throughput = sum(abs(p) for p in ppo_result['power_trajectory'])
            
            milp_battery.update_daily_degradation(milp_throughput)
            ppo_battery.update_daily_degradation(ppo_throughput)
            
            # Accumula profitti
            milp_cumulative_profit += milp_result.net_profit
            ppo_cumulative_profit += ppo_result['net_profit']
            
            # Calcola breakdown MILP usando prezzi reali
            milp_breakdown = self._calculate_milp_service_breakdown(milp_result, daily_services)
            
            # Salva risultati dettagliati
            daily_result = {
                'day': day + 1,
                'date': self.config.start_date + timedelta(days=day),
                'milp_profit': milp_result.net_profit,
                'ppo_profit': ppo_result['net_profit'],
                'milp_cumulative': milp_cumulative_profit,
                'ppo_cumulative': ppo_cumulative_profit,
                'milp_soh': milp_battery.soh,
                'ppo_soh': ppo_battery.soh,
                'avg_price': np.mean(daily_prices),
                'min_price': np.min(daily_prices),
                'max_price': np.max(daily_prices),
                'price_volatility': np.std(daily_prices),
                'milp_utilization': milp_result.battery_utilization,
                'ppo_utilization': ppo_result['battery_utilization'],
                'milp_arbitrage': milp_result.arbitrage_profit,
                'ppo_arbitrage': ppo_result['arbitrage_profit'],
                'milp_flexibility': milp_result.flexibility_revenue,
                'ppo_flexibility': ppo_result['flexibility_revenue'],
                'milp_degradation': milp_result.degradation_cost,
                'ppo_degradation': ppo_result['degradation_cost'],
                # Breakdown servizi flessibilità MILP (CORRETTO)
                'milp_fcr_revenue': milp_breakdown['fcr_revenue'],
                'milp_afrr_revenue': milp_breakdown['afrr_revenue'],
                'milp_mfrr_revenue': milp_breakdown['mfrr_revenue'],
                # Breakdown servizi flessibilità PPO (CORRETTO)
                'ppo_fcr_revenue': ppo_result['fcr_revenue'],
                'ppo_afrr_revenue': ppo_result['afrr_revenue'],
                'ppo_mfrr_revenue': ppo_result['mfrr_revenue'],
                # Utilizzo per servizio
                'milp_fcr_utilization': np.mean(milp_result.flexibility_reservations.get('FCR', [0])) / self.config.battery_max_power_mw,
                'milp_afrr_utilization': np.mean(milp_result.flexibility_reservations.get('aFRR', [0])) / self.config.battery_max_power_mw,
                'milp_mfrr_utilization': np.mean(milp_result.flexibility_reservations.get('mFRR', [0])) / self.config.battery_max_power_mw,
                'ppo_fcr_utilization': np.mean(ppo_result['flexibility_reservations']['FCR']) / self.config.battery_max_power_mw,
                'ppo_afrr_utilization': np.mean(ppo_result['flexibility_reservations']['aFRR']) / self.config.battery_max_power_mw,
                'ppo_mfrr_utilization': np.mean(ppo_result['flexibility_reservations']['mFRR']) / self.config.battery_max_power_mw,
            }
            
            self.daily_results.append(daily_result)
            
            # Progress report
            if (day + 1) % 30 == 0 or day == 0:
                print(f"📈 Giorno {day + 1:3d}: "
                      f"MILP={milp_cumulative_profit:8.0f}€ (SOH={milp_battery.soh:.3f}), "
                      f"PPO={ppo_cumulative_profit:8.0f}€ (SOH={ppo_battery.soh:.3f})")
        
        # Genera summary
        self.annual_summary = self._generate_improved_summary()
        
        print(f"\n🎉 SIMULAZIONE ANNUALE MIGLIORATA COMPLETATA!")
        return self.annual_summary
    
    def _generate_improved_summary(self) -> Dict[str, Any]:
        """Genera summary migliorato con breakdown dettagliato"""
        df = pd.DataFrame(self.daily_results)
        
        final_milp_profit = df['milp_cumulative'].iloc[-1]
        final_ppo_profit = df['ppo_cumulative'].iloc[-1]
        final_milp_soh = df['milp_soh'].iloc[-1]
        final_ppo_soh = df['ppo_soh'].iloc[-1]
        
        # Analisi dettagliata per mercato
        summary = {
            'total_days': len(self.daily_results),
            'final_profits': {
                'milp': final_milp_profit,
                'ppo': final_ppo_profit,
                'advantage_milp': final_milp_profit - final_ppo_profit,
                'advantage_percent': ((final_milp_profit - final_ppo_profit) / final_ppo_profit * 100) if final_ppo_profit != 0 else 0
            },
            'final_soh': {
                'milp': final_milp_soh,
                'ppo': final_ppo_soh,
                'degradation_milp': (1.0 - final_milp_soh) * 100,
                'degradation_ppo': (1.0 - final_ppo_soh) * 100
            },
            'revenue_breakdown': {
                'milp': {
                    'arbitrage_total': df['milp_arbitrage'].sum(),
                    'flexibility_total': df['milp_flexibility'].sum(),
                    'fcr_total': df['milp_fcr_revenue'].sum(),
                    'afrr_total': df['milp_afrr_revenue'].sum(),
                    'mfrr_total': df['milp_mfrr_revenue'].sum(),
                    'degradation_total': df['milp_degradation'].sum()
                },
                'ppo': {
                    'arbitrage_total': df['ppo_arbitrage'].sum(),
                    'flexibility_total': df['ppo_flexibility'].sum(),
                    'fcr_total': df['ppo_fcr_revenue'].sum(),
                    'afrr_total': df['ppo_afrr_revenue'].sum(),
                    'mfrr_total': df['ppo_mfrr_revenue'].sum(),
                    'degradation_total': df['ppo_degradation'].sum()
                }
            },
            'average_daily_metrics': {
                'milp_profit': df['milp_profit'].mean(),
                'ppo_profit': df['ppo_profit'].mean(),
                'milp_utilization': df['milp_utilization'].mean(),
                'ppo_utilization': df['ppo_utilization'].mean(),
                'milp_arbitrage': df['milp_arbitrage'].mean(),
                'ppo_arbitrage': df['ppo_arbitrage'].mean(),
                'milp_flexibility': df['milp_flexibility'].mean(),
                'ppo_flexibility': df['ppo_flexibility'].mean()
            },
            'market_utilization': {
                'milp': {
                    'fcr_avg': df['milp_fcr_utilization'].mean(),
                    'afrr_avg': df['milp_afrr_utilization'].mean(),
                    'mfrr_avg': df['milp_mfrr_utilization'].mean()
                },
                'ppo': {
                    'fcr_avg': df['ppo_fcr_utilization'].mean(),
                    'afrr_avg': df['ppo_afrr_utilization'].mean(),
                    'mfrr_avg': df['ppo_mfrr_utilization'].mean()
                }
            },
            'price_analysis': {
                'avg_price': df['avg_price'].mean(),
                'min_price': df['min_price'].min(),
                'max_price': df['max_price'].max(),
                'avg_volatility': df['price_volatility'].mean()
            },
            'volatility': {
                'milp_profit_std': df['milp_profit'].std(),
                'ppo_profit_std': df['ppo_profit'].std()
            }
        }
        
        return summary
    
    def generate_improved_report(self):
        """Genera report migliorato con breakdown dettagliato"""
        if not self.daily_results:
            print("❌ Nessun risultato disponibile")
            return
        
        print("\n📋 REPORT ANNUALE DETTAGLIATO PPO vs MILP")
        print("=" * 70)
        
        s = self.annual_summary
        
        print(f"\n💰 PERFORMANCE FINANZIARIA TOTALE:")
        print(f"   MILP Totale:     {s['final_profits']['milp']:,.0f} EUR")
        print(f"   PPO Totale:      {s['final_profits']['ppo']:,.0f} EUR")
        print(f"   Vantaggio MILP:  {s['final_profits']['advantage_milp']:+,.0f} EUR ({s['final_profits']['advantage_percent']:+.1f}%)")
        
        print(f"\n� BREAKDOWNA RICAVI PER MERCATO:")
        milp_rev = s['revenue_breakdown']['milp']
        ppo_rev = s['revenue_breakdown']['ppo']
        
        print(f"   ARBITRAGGIO:")
        print(f"     MILP: {milp_rev['arbitrage_total']:,.0f} EUR")
        print(f"     PPO:  {ppo_rev['arbitrage_total']:,.0f} EUR")
        print(f"     Diff: {milp_rev['arbitrage_total'] - ppo_rev['arbitrage_total']:+,.0f} EUR")
        
        print(f"   SERVIZI FLESSIBILITÀ:")
        print(f"     MILP Totale: {milp_rev['flexibility_total']:,.0f} EUR")
        print(f"     PPO Totale:  {ppo_rev['flexibility_total']:,.0f} EUR")
        print(f"     Diff:        {milp_rev['flexibility_total'] - ppo_rev['flexibility_total']:+,.0f} EUR")
        
        print(f"   BREAKDOWN SERVIZI:")
        print(f"     FCR  - MILP: {milp_rev['fcr_total']:,.0f} EUR, PPO: {ppo_rev['fcr_total']:,.0f} EUR")
        print(f"     aFRR - MILP: {milp_rev['afrr_total']:,.0f} EUR, PPO: {ppo_rev['afrr_total']:,.0f} EUR")
        print(f"     mFRR - MILP: {milp_rev['mfrr_total']:,.0f} EUR, PPO: {ppo_rev['mfrr_total']:,.0f} EUR")
        
        print(f"   COSTI DEGRADO:")
        print(f"     MILP: {milp_rev['degradation_total']:,.0f} EUR")
        print(f"     PPO:  {ppo_rev['degradation_total']:,.0f} EUR")
        
        print(f"\n🔋 DEGRADO BATTERIA:")
        print(f"   MILP SOH:        {s['final_soh']['milp']:.3f} ({s['final_soh']['degradation_milp']:.1f}% degrado)")
        print(f"   PPO SOH:         {s['final_soh']['ppo']:.3f} ({s['final_soh']['degradation_ppo']:.1f}% degrado)")
        
        print(f"\n📈 UTILIZZO MERCATI (% capacità batteria):")
        milp_util = s['market_utilization']['milp']
        ppo_util = s['market_utilization']['ppo']
        print(f"   FCR  - MILP: {milp_util['fcr_avg']*100:.1f}%, PPO: {ppo_util['fcr_avg']*100:.1f}%")
        print(f"   aFRR - MILP: {milp_util['afrr_avg']*100:.1f}%, PPO: {ppo_util['afrr_avg']*100:.1f}%")
        print(f"   mFRR - MILP: {milp_util['mfrr_avg']*100:.1f}%, PPO: {ppo_util['mfrr_avg']*100:.1f}%")
        
        print(f"\n💹 ANALISI PREZZI:")
        price_analysis = s['price_analysis']
        print(f"   Prezzo Medio:    {price_analysis['avg_price']:.1f} EUR/MWh")
        print(f"   Prezzo Min:      {price_analysis['min_price']:.1f} EUR/MWh")
        print(f"   Prezzo Max:      {price_analysis['max_price']:.1f} EUR/MWh")
        print(f"   Volatilità Media: {price_analysis['avg_volatility']:.1f} EUR/MWh")
        
        print(f"\n📊 METRICHE GIORNALIERE MEDIE:")
        m = s['average_daily_metrics']
        print(f"   Profitto:        MILP={m['milp_profit']:.0f}€, PPO={m['ppo_profit']:.0f}€")
        print(f"   Utilizzo:        MILP={m['milp_utilization']*100:.1f}%, PPO={m['ppo_utilization']*100:.1f}%")
        print(f"   Arbitraggio:     MILP={m['milp_arbitrage']:.0f}€, PPO={m['ppo_arbitrage']:.0f}€")
        print(f"   Flessibilità:    MILP={m['milp_flexibility']:.0f}€, PPO={m['ppo_flexibility']:.0f}€")
        
        print(f"\n📈 VOLATILITÀ:")
        v = s['volatility']
        print(f"   Std Profitti:    MILP={v['milp_profit_std']:.0f}€, PPO={v['ppo_profit_std']:.0f}€")
        
        # Genera grafici migliorati
        self._create_improved_plots()
    
    def _create_improved_plots(self):
        """Crea grafici migliorati con analisi dettagliate"""
        df = pd.DataFrame(self.daily_results)
        
        # Crea figura con più subplot
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Confronto Annuale Dettagliato PPO vs MILP - Mercati Flessibilità Italiani', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Profitti cumulativi (grande)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(df['day'], df['milp_cumulative'], label='MILP', color='#4ECDC4', linewidth=3)
        ax1.plot(df['day'], df['ppo_cumulative'], label='PPO', color='#FF6B6B', linewidth=3)
        ax1.set_title('Profitti Cumulativi Annuali', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Giorno')
        ax1.set_ylabel('Profitto Cumulativo (EUR)')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k€'))
        
        # 2. SOH Evolution
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(df['day'], df['milp_soh'], label='MILP', color='#4ECDC4', linewidth=2)
        ax2.plot(df['day'], df['ppo_soh'], label='PPO', color='#FF6B6B', linewidth=2)
        ax2.set_title('Evoluzione SOH', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Giorno')
        ax2.set_ylabel('SOH')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Breakdown Ricavi per Fonte
        ax3 = fig.add_subplot(gs[1, 0])
        categories = ['Arbitraggio', 'FCR', 'aFRR', 'mFRR']
        milp_values = [
            df['milp_arbitrage'].sum(),
            df['milp_fcr_revenue'].sum(),
            df['milp_afrr_revenue'].sum(),
            df['milp_mfrr_revenue'].sum()
        ]
        ppo_values = [
            df['ppo_arbitrage'].sum(),
            df['ppo_fcr_revenue'].sum(),
            df['ppo_afrr_revenue'].sum(),
            df['ppo_mfrr_revenue'].sum()
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, milp_values, width, label='MILP', color='#4ECDC4', alpha=0.8)
        bars2 = ax3.bar(x + width/2, ppo_values, width, label='PPO', color='#FF6B6B', alpha=0.8)
        
        ax3.set_title('Ricavi Totali per Mercato', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Mercato')
        ax3.set_ylabel('Ricavi Totali (EUR)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Aggiungi valori sulle barre
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.annotate(f'{height/1000:.0f}k',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        # 4. Utilizzo Mercati
        ax4 = fig.add_subplot(gs[1, 1])
        milp_util = [
            df['milp_fcr_utilization'].mean() * 100,
            df['milp_afrr_utilization'].mean() * 100,
            df['milp_mfrr_utilization'].mean() * 100
        ]
        ppo_util = [
            df['ppo_fcr_utilization'].mean() * 100,
            df['ppo_afrr_utilization'].mean() * 100,
            df['ppo_mfrr_utilization'].mean() * 100
        ]
        
        services = ['FCR', 'aFRR', 'mFRR']
        x = np.arange(len(services))
        
        bars1 = ax4.bar(x - width/2, milp_util, width, label='MILP', color='#4ECDC4', alpha=0.8)
        bars2 = ax4.bar(x + width/2, ppo_util, width, label='PPO', color='#FF6B6B', alpha=0.8)
        
        ax4.set_title('Utilizzo Medio Mercati', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Servizio')
        ax4.set_ylabel('Utilizzo (% Capacità)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(services)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Analisi Prezzi
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(df['day'], df['avg_price'], color='orange', alpha=0.7, linewidth=1)
        ax5.fill_between(df['day'], df['min_price'], df['max_price'], 
                        color='orange', alpha=0.2, label='Range Prezzi')
        ax5.plot(df['day'], df['avg_price'], color='orange', linewidth=2, label='Prezzo Medio')
        ax5.set_title('Evoluzione Prezzi Energia', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Giorno')
        ax5.set_ylabel('Prezzo (EUR/MWh)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Profitti Giornalieri con Media Mobile
        ax6 = fig.add_subplot(gs[2, :2])
        # Media mobile 7 giorni
        milp_ma = df['milp_profit'].rolling(window=7, center=True).mean()
        ppo_ma = df['ppo_profit'].rolling(window=7, center=True).mean()
        
        ax6.plot(df['day'], df['milp_profit'], color='#4ECDC4', alpha=0.3, linewidth=1)
        ax6.plot(df['day'], df['ppo_profit'], color='#FF6B6B', alpha=0.3, linewidth=1)
        ax6.plot(df['day'], milp_ma, label='MILP (Media 7gg)', color='#4ECDC4', linewidth=2)
        ax6.plot(df['day'], ppo_ma, label='PPO (Media 7gg)', color='#FF6B6B', linewidth=2)
        ax6.set_title('Profitti Giornalieri con Media Mobile', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Giorno')
        ax6.set_ylabel('Profitto Giornaliero (EUR)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Volatilità Prezzi
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.plot(df['day'], df['price_volatility'], color='purple', linewidth=2)
        ax7.set_title('Volatilità Prezzi', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Giorno')
        ax7.set_ylabel('Volatilità (EUR/MWh)')
        ax7.grid(True, alpha=0.3)
        
        # 8. Vantaggio MILP vs PPO
        ax8 = fig.add_subplot(gs[3, 0])
        advantage = df['milp_profit'] - df['ppo_profit']
        advantage_ma = advantage.rolling(window=7, center=True).mean()
        
        ax8.plot(df['day'], advantage, color='gray', alpha=0.5, linewidth=1)
        ax8.plot(df['day'], advantage_ma, color='green', linewidth=2, label='Media 7gg')
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax8.fill_between(df['day'], advantage, 0, 
                        where=(advantage >= 0), color='green', alpha=0.3, label='MILP Vantaggio')
        ax8.fill_between(df['day'], advantage, 0, 
                        where=(advantage < 0), color='red', alpha=0.3, label='PPO Vantaggio')
        ax8.set_title('Vantaggio Giornaliero', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Giorno')
        ax8.set_ylabel('Vantaggio MILP (EUR)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Distribuzione Profitti
        ax9 = fig.add_subplot(gs[3, 1])
        ax9.hist(df['milp_profit'], bins=30, alpha=0.7, label='MILP', color='#4ECDC4', density=True)
        ax9.hist(df['ppo_profit'], bins=30, alpha=0.7, label='PPO', color='#FF6B6B', density=True)
        ax9.axvline(df['milp_profit'].mean(), color='#4ECDC4', linestyle='--', linewidth=2, label='Media MILP')
        ax9.axvline(df['ppo_profit'].mean(), color='#FF6B6B', linestyle='--', linewidth=2, label='Media PPO')
        ax9.set_title('Distribuzione Profitti', fontsize=12, fontweight='bold')
        ax9.set_xlabel('Profitto Giornaliero (EUR)')
        ax9.set_ylabel('Densità')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # 10. Correlazione Prezzo-Profitto
        ax10 = fig.add_subplot(gs[3, 2])
        ax10.scatter(df['avg_price'], df['milp_profit'], alpha=0.6, color='#4ECDC4', label='MILP', s=20)
        ax10.scatter(df['avg_price'], df['ppo_profit'], alpha=0.6, color='#FF6B6B', label='PPO', s=20)
        
        # Linee di tendenza
        z_milp = np.polyfit(df['avg_price'], df['milp_profit'], 1)
        z_ppo = np.polyfit(df['avg_price'], df['ppo_profit'], 1)
        p_milp = np.poly1d(z_milp)
        p_ppo = np.poly1d(z_ppo)
        
        price_range = np.linspace(df['avg_price'].min(), df['avg_price'].max(), 100)
        ax10.plot(price_range, p_milp(price_range), color='#4ECDC4', linestyle='--', linewidth=2)
        ax10.plot(price_range, p_ppo(price_range), color='#FF6B6B', linestyle='--', linewidth=2)
        
        ax10.set_title('Correlazione Prezzo-Profitto', fontsize=12, fontweight='bold')
        ax10.set_xlabel('Prezzo Medio (EUR/MWh)')
        ax10.set_ylabel('Profitto (EUR)')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        # Salva
        output_file = f"{self.config.results_dir}/improved_annual_comparison_detailed.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n📊 Grafici dettagliati salvati: {output_file}")
        plt.close()
        
        # Crea anche un grafico separato per il breakdown dei ricavi
        self._create_revenue_breakdown_chart()
    
    def _create_revenue_breakdown_chart(self):
        """Crea grafico dettagliato del breakdown dei ricavi"""
        df = pd.DataFrame(self.daily_results)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Breakdown Dettagliato Ricavi per Mercato', fontsize=16, fontweight='bold')
        
        # Grafico a torta MILP
        milp_revenues = [
            df['milp_arbitrage'].sum(),
            df['milp_fcr_revenue'].sum(),
            df['milp_afrr_revenue'].sum(),
            df['milp_mfrr_revenue'].sum()
        ]
        milp_labels = ['Arbitraggio', 'FCR', 'aFRR', 'mFRR']
        milp_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        
        # Rimuovi categorie con valore zero
        milp_nonzero = [(rev, label, color) for rev, label, color in zip(milp_revenues, milp_labels, milp_colors) if rev > 0]
        if milp_nonzero:
            milp_revenues_nz, milp_labels_nz, milp_colors_nz = zip(*milp_nonzero)
            
            wedges1, texts1, autotexts1 = ax1.pie(milp_revenues_nz, labels=milp_labels_nz, colors=milp_colors_nz,
                                                  autopct=lambda pct: f'{pct:.1f}%\n({pct*sum(milp_revenues_nz)/100:.0f}€)',
                                                  startangle=90, textprops={'fontsize': 10})
        ax1.set_title(f'MILP - Totale: {sum(milp_revenues):,.0f} EUR', fontsize=14, fontweight='bold')
        
        # Grafico a torta PPO
        ppo_revenues = [
            df['ppo_arbitrage'].sum(),
            df['ppo_fcr_revenue'].sum(),
            df['ppo_afrr_revenue'].sum(),
            df['ppo_mfrr_revenue'].sum()
        ]
        ppo_labels = ['Arbitraggio', 'FCR', 'aFRR', 'mFRR']
        ppo_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        
        # Rimuovi categorie con valore zero
        ppo_nonzero = [(rev, label, color) for rev, label, color in zip(ppo_revenues, ppo_labels, ppo_colors) if rev > 0]
        if ppo_nonzero:
            ppo_revenues_nz, ppo_labels_nz, ppo_colors_nz = zip(*ppo_nonzero)
            
            wedges2, texts2, autotexts2 = ax2.pie(ppo_revenues_nz, labels=ppo_labels_nz, colors=ppo_colors_nz,
                                                  autopct=lambda pct: f'{pct:.1f}%\n({pct*sum(ppo_revenues_nz)/100:.0f}€)',
                                                  startangle=90, textprops={'fontsize': 10})
        ax2.set_title(f'PPO - Totale: {sum(ppo_revenues):,.0f} EUR', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Salva
        output_file = f"{self.config.results_dir}/revenue_breakdown_chart.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Grafico breakdown ricavi salvato: {output_file}")
        plt.close()
    
    def export_results(self):
        """Esporta risultati migliorati"""
        if not self.daily_results:
            return
        
        df = pd.DataFrame(self.daily_results)
        output_file = f"{self.config.results_dir}/improved_annual_results.csv"
        df.to_csv(output_file, index=False)
        print(f"📁 Risultati esportati: {output_file}")


def main():
    """Test del sistema migliorato"""
    config = ImprovedAnnualConfig(
        start_date=datetime(2024, 1, 1),
        days=365,  # Anno completo!
        pretrain_months=2,
        retrain_frequency_months=1,
        training_episodes=1000,  # Episodi completi per training migliore
        use_forecast_errors=True
    )
    
    engine = ImprovedAnnualComparisonEngine(config)
    results = engine.run_improved_annual_comparison()
    
    engine.generate_improved_report()
    engine.export_results()
    
    return results


if __name__ == "__main__":
    print("🚀 Sistema di Confronto Annuale Migliorato PPO vs MILP")
    results = main()
    print("\n✅ Test completato!")