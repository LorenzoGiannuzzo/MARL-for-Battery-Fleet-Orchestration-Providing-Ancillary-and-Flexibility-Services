"""
Enhanced Annual Comparison: PPO vs MILP
Usa l'ambiente PPO migliorato per un confronto più equo con MILP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import time
import os
from pathlib import Path

# Import components
from enhanced_ppo_environment import EnhancedBatteryTradingEnv
from milp_optimizer import MILPOptimizer, BatteryParameters
from flexibility_market import FlexibilityMarket
from drl_flexibility_analysis import ForecastErrorGenerator
from italian_market_config import ITALIAN_CONFIG


def generate_price_data(n_hours):
    """Generate synthetic price data for comparison"""
    prices = []
    for hour in range(n_hours):
        # Daily pattern
        hour_of_day = hour % 24
        base_price = 50.0 + 30.0 * np.sin(hour_of_day * np.pi / 12)
        
        # Weekly pattern
        day_of_week = (hour // 24) % 7
        weekend_factor = 0.9 if day_of_week >= 5 else 1.0
        
        # Seasonal pattern
        day_of_year = (hour // 24) % 365
        seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Add noise
        noise = np.random.normal(0, 5.0)
        
        final_price = max(10.0, base_price * weekend_factor * seasonal_factor + noise)
        prices.append(final_price)
    
    return prices


def run_enhanced_annual_comparison():
    """Run enhanced annual comparison with improved PPO"""
    
    print("🚀 ENHANCED ANNUAL COMPARISON: PPO vs MILP")
    print("=" * 70)
    print("Confronto annuale con PPO migliorato")
    print()
    
    # Generate annual price data
    print("📊 Generazione dati annuali...")
    annual_prices = []
    for month in range(1, 13):
        monthly_prices = generate_price_data(24 * 30)  # 30 days per month
        annual_prices.extend(monthly_prices)
    
    print(f"   Generati {len(annual_prices)} prezzi orari per l'anno")
    print(f"   Prezzo medio: {np.mean(annual_prices):.2f} EUR/MWh")
    print(f"   Range prezzi: {np.min(annual_prices):.1f} - {np.max(annual_prices):.1f} EUR/MWh")
    
    # Initialize components
    flexibility_market = FlexibilityMarket(random_seed=42)
    battery_params = BatteryParameters()
    
    # Results storage
    results = {
        'ppo': {
            'daily_profits': [],
            'daily_arbitrage': [],
            'daily_flexibility': [],
            'daily_fcr': [],
            'daily_afrr': [],
            'daily_mfrr': [],
            'daily_degradation': [],
            'soc_trajectory': [],
            'power_trajectory': [],
            'monthly_profits': [0] * 12
        },
        'milp': {
            'daily_profits': [],
            'daily_arbitrage': [],
            'daily_flexibility': [],
            'daily_fcr': [],
            'daily_afrr': [],
            'daily_mfrr': [],
            'daily_degradation': [],
            'soc_trajectory': [],
            'power_trajectory': [],
            'monthly_profits': [0] * 12
        }
    }
    
    # Run PPO simulation
    print("\n🤖 SIMULAZIONE PPO ENHANCED")
    print("-" * 40)
    
    ppo_start_time = time.time()
    
    # Try to load enhanced model, fallback to monthly models
    try:
        ppo_model = PPO.load('ppo_models/enhanced_ppo_final')
        print("✅ Caricato modello enhanced_ppo_final")
        use_monthly_models = False
    except:
        print("⚠️  Modello enhanced non trovato, uso modelli mensili")
        use_monthly_models = True
        ppo_model = None
    
    current_day = 0
    for month in range(1, 13):
        print(f"\n📅 Mese {month}")
        
        # Load monthly model if needed
        if use_monthly_models:
            try:
                ppo_model = PPO.load(f'ppo_models/enhanced_ppo_month_{month}')
                print(f"   Caricato modello per mese {month}")
            except:
                print(f"   ⚠️  Modello mese {month} non trovato, uso modello base")
                try:
                    ppo_model = PPO.load('ppo_models/ppo_flexibility_final')
                except:
                    print("   ❌ Nessun modello disponibile!")
                    continue
        
        # Monthly data
        month_start = current_day * 24
        month_end = month_start + (30 * 24)
        monthly_prices = annual_prices[month_start:month_end]
        
        if len(monthly_prices) == 0:
            continue
        
        # Create enhanced environment for this month
        error_gen = ForecastErrorGenerator('normal')
        env = EnhancedBatteryTradingEnv(
            prices=monthly_prices,
            error_generator=error_gen,
            flexibility_enabled=True,
            random_seed=42
        )
        
        # Run PPO for this month
        obs, _ = env.reset()
        monthly_profit = 0
        monthly_arbitrage = 0
        monthly_flexibility = 0
        monthly_fcr = 0
        monthly_afrr = 0
        monthly_mfrr = 0
        monthly_degradation = 0
        
        daily_profit = 0
        daily_arbitrage = 0
        daily_flexibility = 0
        daily_fcr = 0
        daily_afrr = 0
        daily_mfrr = 0
        daily_degradation = 0
        
        step_count = 0
        done = False
        
        while not done and step_count < len(monthly_prices):
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            # Accumulate metrics
            arbitrage_profit = info.get('arbitrage_profit', 0)
            flexibility_revenue = info.get('flexibility_revenue', 0)
            fcr_revenue = info.get('flexibility_breakdown', {}).get('fcr_revenue', 0)
            afrr_revenue = info.get('flexibility_breakdown', {}).get('afrr_revenue', 0)
            mfrr_revenue = info.get('flexibility_breakdown', {}).get('mfrr_revenue', 0)
            degradation_cost = info.get('degradation_cost', 0)
            
            daily_profit += arbitrage_profit + flexibility_revenue - degradation_cost
            daily_arbitrage += arbitrage_profit
            daily_flexibility += flexibility_revenue
            daily_fcr += fcr_revenue
            daily_afrr += afrr_revenue
            daily_mfrr += mfrr_revenue
            daily_degradation += degradation_cost
            
            monthly_profit += arbitrage_profit + flexibility_revenue - degradation_cost
            monthly_arbitrage += arbitrage_profit
            monthly_flexibility += flexibility_revenue
            monthly_fcr += fcr_revenue
            monthly_afrr += afrr_revenue
            monthly_mfrr += mfrr_revenue
            monthly_degradation += degradation_cost
            
            # Store trajectories
            results['ppo']['soc_trajectory'].append(info.get('soc', 0.5))
            results['ppo']['power_trajectory'].append(info.get('arbitrage_energy', 0))
            
            step_count += 1
            
            # End of day (every 24 hours)
            if step_count % 24 == 0:
                results['ppo']['daily_profits'].append(daily_profit)
                results['ppo']['daily_arbitrage'].append(daily_arbitrage)
                results['ppo']['daily_flexibility'].append(daily_flexibility)
                results['ppo']['daily_fcr'].append(daily_fcr)
                results['ppo']['daily_afrr'].append(daily_afrr)
                results['ppo']['daily_mfrr'].append(daily_mfrr)
                results['ppo']['daily_degradation'].append(daily_degradation)
                
                # Reset daily counters
                daily_profit = daily_arbitrage = daily_flexibility = 0
                daily_fcr = daily_afrr = daily_mfrr = daily_degradation = 0
            
            if truncated:
                break
        
        results['ppo']['monthly_profits'][month - 1] = monthly_profit
        current_day += 30
        
        print(f"   PPO Mese {month}: {monthly_profit:.0f} EUR")
        print(f"     Arbitraggio: {monthly_arbitrage:.0f} EUR")
        print(f"     Flessibilità: {monthly_flexibility:.0f} EUR")
        print(f"       - FCR: {monthly_fcr:.0f} EUR")
        print(f"       - aFRR: {monthly_afrr:.0f} EUR")
        print(f"       - mFRR: {monthly_mfrr:.0f} EUR")
    
    ppo_time = time.time() - ppo_start_time
    ppo_total = sum(results['ppo']['monthly_profits'])
    
    print(f"\n✅ PPO completato in {ppo_time:.1f}s")
    print(f"   Profitto totale PPO: {ppo_total:.0f} EUR")
    
    # Run MILP simulation
    print("\n🔧 SIMULAZIONE MILP")
    print("-" * 40)
    
    milp_start_time = time.time()
    optimizer = MILPOptimizer(battery_params, flexibility_market)
    
    current_day = 0
    for month in range(1, 13):
        print(f"\n📅 Mese {month}")
        
        # Monthly data
        month_start = current_day * 24
        month_end = month_start + (30 * 24)
        monthly_prices = annual_prices[month_start:month_end]
        
        if len(monthly_prices) == 0:
            continue
        
        # Generate flexibility services for the month
        flexibility_services = []
        for hour in range(len(monthly_prices)):
            hour_of_day = hour % 24
            day_of_week = (hour // 24) % 7
            services = flexibility_market.get_flexibility_opportunities(hour_of_day, day_of_week)
            flexibility_services.append(services)
        
        # Run MILP optimization for the month
        result = optimizer.optimize(
            time_horizon=len(monthly_prices),
            energy_prices=monthly_prices,
            flexibility_services=flexibility_services
        )
        
        monthly_profit = result.net_profit
        monthly_arbitrage = result.arbitrage_profit
        monthly_flexibility = result.flexibility_revenue
        monthly_degradation = result.degradation_cost
        
        # Extract service-specific revenues
        monthly_fcr = 0
        monthly_afrr = 0
        monthly_mfrr = 0
        
        for t in range(len(monthly_prices)):
            services = flexibility_services[t]
            service_dict = {s.service_type.value: s for s in services}
            
            if 'FCR' in service_dict and t < len(result.flexibility_reservations['FCR']):
                fcr_capacity = result.flexibility_reservations['FCR'][t] or 0
                if fcr_capacity > 0:
                    monthly_fcr += fcr_capacity * service_dict['FCR'].capacity_price
            
            if 'aFRR' in service_dict and t < len(result.flexibility_reservations['aFRR']):
                afrr_capacity = result.flexibility_reservations['aFRR'][t] or 0
                if afrr_capacity > 0:
                    monthly_afrr += afrr_capacity * service_dict['aFRR'].capacity_price
            
            if 'mFRR' in service_dict and t < len(result.flexibility_reservations['mFRR']):
                mfrr_capacity = result.flexibility_reservations['mFRR'][t] or 0
                if mfrr_capacity > 0:
                    monthly_mfrr += mfrr_capacity * service_dict['mFRR'].capacity_price
        
        results['milp']['monthly_profits'][month - 1] = monthly_profit
        
        # Store daily results (approximate from monthly)
        days_in_month = len(monthly_prices) // 24
        daily_profit_avg = monthly_profit / days_in_month if days_in_month > 0 else 0
        daily_arbitrage_avg = monthly_arbitrage / days_in_month if days_in_month > 0 else 0
        daily_flexibility_avg = monthly_flexibility / days_in_month if days_in_month > 0 else 0
        daily_fcr_avg = monthly_fcr / days_in_month if days_in_month > 0 else 0
        daily_afrr_avg = monthly_afrr / days_in_month if days_in_month > 0 else 0
        daily_mfrr_avg = monthly_mfrr / days_in_month if days_in_month > 0 else 0
        daily_degradation_avg = monthly_degradation / days_in_month if days_in_month > 0 else 0
        
        for _ in range(days_in_month):
            results['milp']['daily_profits'].append(daily_profit_avg)
            results['milp']['daily_arbitrage'].append(daily_arbitrage_avg)
            results['milp']['daily_flexibility'].append(daily_flexibility_avg)
            results['milp']['daily_fcr'].append(daily_fcr_avg)
            results['milp']['daily_afrr'].append(daily_afrr_avg)
            results['milp']['daily_mfrr'].append(daily_mfrr_avg)
            results['milp']['daily_degradation'].append(daily_degradation_avg)
        
        # Store trajectories (simplified)
        if result.soc_trajectory:
            results['milp']['soc_trajectory'].extend(result.soc_trajectory)
        if result.power_trajectory:
            results['milp']['power_trajectory'].extend(result.power_trajectory)
        
        current_day += 30
        
        print(f"   MILP Mese {month}: {monthly_profit:.0f} EUR")
        print(f"     Arbitraggio: {monthly_arbitrage:.0f} EUR")
        print(f"     Flessibilità: {monthly_flexibility:.0f} EUR")
        print(f"       - FCR: {monthly_fcr:.0f} EUR")
        print(f"       - aFRR: {monthly_afrr:.0f} EUR")
        print(f"       - mFRR: {monthly_mfrr:.0f} EUR")
    
    milp_time = time.time() - milp_start_time
    milp_total = sum(results['milp']['monthly_profits'])
    
    print(f"\n✅ MILP completato in {milp_time:.1f}s")
    print(f"   Profitto totale MILP: {milp_total:.0f} EUR")
    
    # Calculate final comparison
    print("\n📊 RISULTATI FINALI ENHANCED COMPARISON")
    print("=" * 70)
    
    ppo_arbitrage_total = sum(results['ppo']['daily_arbitrage'])
    ppo_flexibility_total = sum(results['ppo']['daily_flexibility'])
    ppo_fcr_total = sum(results['ppo']['daily_fcr'])
    ppo_afrr_total = sum(results['ppo']['daily_afrr'])
    ppo_mfrr_total = sum(results['ppo']['daily_mfrr'])
    ppo_degradation_total = sum(results['ppo']['daily_degradation'])
    
    milp_arbitrage_total = sum(results['milp']['daily_arbitrage'])
    milp_flexibility_total = sum(results['milp']['daily_flexibility'])
    milp_fcr_total = sum(results['milp']['daily_fcr'])
    milp_afrr_total = sum(results['milp']['daily_afrr'])
    milp_mfrr_total = sum(results['milp']['daily_mfrr'])
    milp_degradation_total = sum(results['milp']['daily_degradation'])
    
    advantage = milp_total - ppo_total
    advantage_pct = (advantage / milp_total * 100) if milp_total > 0 else 0
    
    print(f"💰 PERFORMANCE FINANZIARIA TOTALE:")
    print(f"   MILP Totale:     {milp_total:,.0f} EUR")
    print(f"   PPO Totale:      {ppo_total:,.0f} EUR")
    print(f"   Vantaggio MILP:  {advantage:+,.0f} EUR ({advantage_pct:+.1f}%)")
    print()
    
    print(f"📈 BREAKDOWN RICAVI PER MERCATO:")
    print(f"   ARBITRAGGIO:")
    print(f"     MILP: {milp_arbitrage_total:,.0f} EUR")
    print(f"     PPO:  {ppo_arbitrage_total:,.0f} EUR")
    print(f"     Diff: {milp_arbitrage_total - ppo_arbitrage_total:+,.0f} EUR")
    print(f"   SERVIZI FLESSIBILITÀ:")
    print(f"     MILP Totale: {milp_flexibility_total:,.0f} EUR")
    print(f"     PPO Totale:  {ppo_flexibility_total:,.0f} EUR")
    print(f"     Diff:        {milp_flexibility_total - ppo_flexibility_total:+,.0f} EUR")
    print(f"   BREAKDOWN SERVIZI:")
    print(f"     FCR  - MILP: {milp_fcr_total:,.0f} EUR, PPO: {ppo_fcr_total:,.0f} EUR")
    print(f"     aFRR - MILP: {milp_afrr_total:,.0f} EUR, PPO: {ppo_afrr_total:,.0f} EUR")
    print(f"     mFRR - MILP: {milp_mfrr_total:,.0f} EUR, PPO: {ppo_mfrr_total:,.0f} EUR")
    print(f"   COSTI DEGRADO:")
    print(f"     MILP: {milp_degradation_total:,.0f} EUR")
    print(f"     PPO:  {ppo_degradation_total:,.0f} EUR")
    print()
    
    # Calculate market utilization
    total_capacity_hours = 2.0 * len(results['ppo']['daily_profits']) * 24  # 2MW * hours
    
    ppo_fcr_utilization = (ppo_fcr_total / 45.0 / total_capacity_hours * 100) if total_capacity_hours > 0 else 0
    ppo_afrr_utilization = (ppo_afrr_total / 35.0 / total_capacity_hours * 100) if total_capacity_hours > 0 else 0
    ppo_mfrr_utilization = (ppo_mfrr_total / 25.0 / total_capacity_hours * 100) if total_capacity_hours > 0 else 0
    
    milp_fcr_utilization = (milp_fcr_total / 45.0 / total_capacity_hours * 100) if total_capacity_hours > 0 else 0
    milp_afrr_utilization = (milp_afrr_total / 35.0 / total_capacity_hours * 100) if total_capacity_hours > 0 else 0
    milp_mfrr_utilization = (milp_mfrr_total / 25.0 / total_capacity_hours * 100) if total_capacity_hours > 0 else 0
    
    print(f"📈 UTILIZZO MERCATI (% capacità batteria):")
    print(f"   FCR  - MILP: {milp_fcr_utilization:.1f}%, PPO: {ppo_fcr_utilization:.1f}%")
    print(f"   aFRR - MILP: {milp_afrr_utilization:.1f}%, PPO: {ppo_afrr_utilization:.1f}%")
    print(f"   mFRR - MILP: {milp_mfrr_utilization:.1f}%, PPO: {ppo_mfrr_utilization:.1f}%")
    print()
    
    # Performance metrics
    print(f"⚡ METRICHE PERFORMANCE:")
    print(f"   Tempo esecuzione MILP: {milp_time:.1f}s")
    print(f"   Tempo esecuzione PPO:  {ppo_time:.1f}s")
    print(f"   Speedup PPO:           {milp_time/ppo_time:.1f}x")
    print()
    
    # Daily averages
    ppo_daily_avg = np.mean(results['ppo']['daily_profits']) if results['ppo']['daily_profits'] else 0
    milp_daily_avg = np.mean(results['milp']['daily_profits']) if results['milp']['daily_profits'] else 0
    
    print(f"📊 METRICHE GIORNALIERE MEDIE:")
    print(f"   Profitto:        MILP={milp_daily_avg:.0f}€, PPO={ppo_daily_avg:.0f}€")
    print(f"   Arbitraggio:     MILP={np.mean(results['milp']['daily_arbitrage']):.0f}€, PPO={np.mean(results['ppo']['daily_arbitrage']):.0f}€")
    print(f"   Flessibilità:    MILP={np.mean(results['milp']['daily_flexibility']):.0f}€, PPO={np.mean(results['ppo']['daily_flexibility']):.0f}€")
    print()
    
    # Volatility
    ppo_volatility = np.std(results['ppo']['daily_profits']) if results['ppo']['daily_profits'] else 0
    milp_volatility = np.std(results['milp']['daily_profits']) if results['milp']['daily_profits'] else 0
    
    print(f"📈 VOLATILITÀ:")
    print(f"   Std Profitti:    MILP={milp_volatility:.0f}€, PPO={ppo_volatility:.0f}€")
    print()
    
    # Create visualizations
    create_enhanced_comparison_charts(results, annual_prices)
    
    # Export results
    export_enhanced_results(results, {
        'ppo_total': ppo_total,
        'milp_total': milp_total,
        'advantage': advantage,
        'advantage_pct': advantage_pct,
        'ppo_time': ppo_time,
        'milp_time': milp_time
    })
    
    print("📊 Grafici dettagliati salvati: enhanced_results/enhanced_annual_comparison.png")
    print("📁 Risultati esportati: enhanced_results/enhanced_annual_results.csv")
    print()
    print("✅ Enhanced comparison completato!")
    
    return results


def create_enhanced_comparison_charts(results, prices):
    """Create enhanced comparison charts"""
    
    os.makedirs('enhanced_results', exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Enhanced PPO vs MILP Annual Comparison', fontsize=16, fontweight='bold')
    
    # 1. Monthly profits comparison
    months = list(range(1, 13))
    axes[0, 0].bar([m - 0.2 for m in months], results['milp']['monthly_profits'], 
                   width=0.4, label='MILP', alpha=0.8, color='blue')
    axes[0, 0].bar([m + 0.2 for m in months], results['ppo']['monthly_profits'], 
                   width=0.4, label='Enhanced PPO', alpha=0.8, color='green')
    axes[0, 0].set_title('Monthly Profits Comparison')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Profit (EUR)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Daily profits over time
    days = list(range(len(results['ppo']['daily_profits'])))
    axes[0, 1].plot(days, results['milp']['daily_profits'], label='MILP', alpha=0.8, color='blue')
    axes[0, 1].plot(days, results['ppo']['daily_profits'], label='Enhanced PPO', alpha=0.8, color='green')
    axes[0, 1].set_title('Daily Profits Over Time')
    axes[0, 1].set_xlabel('Day')
    axes[0, 1].set_ylabel('Daily Profit (EUR)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Service utilization comparison
    services = ['FCR', 'aFRR', 'mFRR']
    milp_services = [sum(results['milp']['daily_fcr']), 
                     sum(results['milp']['daily_afrr']), 
                     sum(results['milp']['daily_mfrr'])]
    ppo_services = [sum(results['ppo']['daily_fcr']), 
                    sum(results['ppo']['daily_afrr']), 
                    sum(results['ppo']['daily_mfrr'])]
    
    x = np.arange(len(services))
    axes[0, 2].bar(x - 0.2, milp_services, width=0.4, label='MILP', alpha=0.8, color='blue')
    axes[0, 2].bar(x + 0.2, ppo_services, width=0.4, label='Enhanced PPO', alpha=0.8, color='green')
    axes[0, 2].set_title('Flexibility Services Revenue')
    axes[0, 2].set_xlabel('Service Type')
    axes[0, 2].set_ylabel('Total Revenue (EUR)')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(services)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. SOC trajectories (first 30 days)
    if results['ppo']['soc_trajectory'] and results['milp']['soc_trajectory']:
        hours = list(range(min(720, len(results['ppo']['soc_trajectory']))))  # 30 days
        ppo_soc = results['ppo']['soc_trajectory'][:len(hours)]
        milp_soc = results['milp']['soc_trajectory'][:len(hours)]
        
        axes[1, 0].plot(hours, milp_soc, label='MILP', alpha=0.8, color='blue')
        axes[1, 0].plot(hours, ppo_soc, label='Enhanced PPO', alpha=0.8, color='green')
        axes[1, 0].set_title('SOC Trajectory (First 30 Days)')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('SOC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Cumulative profits
    milp_cumulative = np.cumsum(results['milp']['daily_profits'])
    ppo_cumulative = np.cumsum(results['ppo']['daily_profits'])
    
    axes[1, 1].plot(days, milp_cumulative, label='MILP', alpha=0.8, color='blue')
    axes[1, 1].plot(days, ppo_cumulative, label='Enhanced PPO', alpha=0.8, color='green')
    axes[1, 1].set_title('Cumulative Profits')
    axes[1, 1].set_xlabel('Day')
    axes[1, 1].set_ylabel('Cumulative Profit (EUR)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Performance gap over time
    profit_gap = [m - p for m, p in zip(results['milp']['daily_profits'], results['ppo']['daily_profits'])]
    axes[1, 2].plot(days, profit_gap, color='red', alpha=0.8)
    axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 2].set_title('Daily Performance Gap (MILP - PPO)')
    axes[1, 2].set_xlabel('Day')
    axes[1, 2].set_ylabel('Profit Gap (EUR)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_results/enhanced_annual_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def export_enhanced_results(results, summary):
    """Export enhanced results to CSV"""
    
    os.makedirs('enhanced_results', exist_ok=True)
    
    # Create comprehensive results DataFrame
    max_days = max(len(results['ppo']['daily_profits']), len(results['milp']['daily_profits']))
    
    df = pd.DataFrame({
        'day': range(1, max_days + 1),
        'ppo_profit': results['ppo']['daily_profits'] + [0] * (max_days - len(results['ppo']['daily_profits'])),
        'milp_profit': results['milp']['daily_profits'] + [0] * (max_days - len(results['milp']['daily_profits'])),
        'ppo_arbitrage': results['ppo']['daily_arbitrage'] + [0] * (max_days - len(results['ppo']['daily_arbitrage'])),
        'milp_arbitrage': results['milp']['daily_arbitrage'] + [0] * (max_days - len(results['milp']['daily_arbitrage'])),
        'ppo_flexibility': results['ppo']['daily_flexibility'] + [0] * (max_days - len(results['ppo']['daily_flexibility'])),
        'milp_flexibility': results['milp']['daily_flexibility'] + [0] * (max_days - len(results['milp']['daily_flexibility'])),
        'ppo_fcr': results['ppo']['daily_fcr'] + [0] * (max_days - len(results['ppo']['daily_fcr'])),
        'milp_fcr': results['milp']['daily_fcr'] + [0] * (max_days - len(results['milp']['daily_fcr'])),
        'ppo_afrr': results['ppo']['daily_afrr'] + [0] * (max_days - len(results['ppo']['daily_afrr'])),
        'milp_afrr': results['milp']['daily_afrr'] + [0] * (max_days - len(results['milp']['daily_afrr'])),
        'ppo_mfrr': results['ppo']['daily_mfrr'] + [0] * (max_days - len(results['ppo']['daily_mfrr'])),
        'milp_mfrr': results['milp']['daily_mfrr'] + [0] * (max_days - len(results['milp']['daily_mfrr'])),
    })
    
    df.to_csv('enhanced_results/enhanced_annual_results.csv', index=False)
    
    # Create summary file
    with open('enhanced_results/enhanced_summary.txt', 'w') as f:
        f.write("ENHANCED PPO vs MILP Annual Comparison Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"PPO Total Profit: {summary['ppo_total']:,.0f} EUR\n")
        f.write(f"MILP Total Profit: {summary['milp_total']:,.0f} EUR\n")
        f.write(f"MILP Advantage: {summary['advantage']:+,.0f} EUR ({summary['advantage_pct']:+.1f}%)\n\n")
        f.write(f"Execution Times:\n")
        f.write(f"  PPO: {summary['ppo_time']:.1f}s\n")
        f.write(f"  MILP: {summary['milp_time']:.1f}s\n")
        f.write(f"  PPO Speedup: {summary['milp_time']/summary['ppo_time']:.1f}x\n")


def main():
    """Main function"""
    
    # Check if enhanced models exist
    if not os.path.exists('ppo_models/enhanced_ppo_final.zip') and not os.path.exists('ppo_models/enhanced_ppo_month_1.zip'):
        print("⚠️  Modelli enhanced non trovati!")
        print("   Esegui prima: python enhanced_ppo_pretraining.py")
        return
    
    # Run enhanced comparison
    results = run_enhanced_annual_comparison()
    
    return results


if __name__ == "__main__":
    main()