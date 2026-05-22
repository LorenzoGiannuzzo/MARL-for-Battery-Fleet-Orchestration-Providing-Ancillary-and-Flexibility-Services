#!/usr/bin/env python3
"""
Script principale per eseguire il confronto PPO vs MILP
"""

from comparison_engine import ComparisonEngine, ComparisonConfig
from visualization_system import VisualizationSystem
from statistical_analysis_reporting import StatisticalAnalysisReporter
from export_functionality import ExportManager

def main():
    print("🚀 Avvio confronto PPO vs MILP per servizi di flessibilità italiani")
    print("=" * 70)
    
    # Configurazione del confronto
    config = ComparisonConfig(
        time_horizon=24,  # 24 ore
        error_distributions=['uniform', 'normal'],  # 2 distribuzioni di errore
        n_episodes_per_distribution=3,  # 3 episodi per distribuzione
        battery_capacity_mwh=4.0,
        battery_max_power_mw=2.0,
        flexibility_enabled=True
    )
    
    try:
        # 1. Esegui confronto
        print("📊 Esecuzione confronto...")
        engine = ComparisonEngine(config)
        
        # Genera prezzi di esempio per 24 ore
        import numpy as np
        base_prices = [50.0 + 30.0 * np.sin(h * np.pi / 12) + 5.0 * np.random.normal() for h in range(24)]
        
        summary = engine.run_comparison(base_prices)
        episode_results = engine.get_episode_results()
        
        print(f"✅ Confronto completato: {summary.total_episodes} episodi")
        print(f"   - Episodi riusciti: {summary.successful_episodes}")
        print(f"   - Tasso di successo: {summary.successful_episodes/summary.total_episodes*100:.1f}%")
        
        # 2. Genera visualizzazioni
        print("\n🎨 Generazione visualizzazioni...")
        viz_system = VisualizationSystem(output_dir="results")
        viz_files = viz_system.generate_comprehensive_report(summary, episode_results)
        
        print(f"✅ Generate {len(viz_files)} visualizzazioni")
        
        # 3. Analisi statistica
        print("\n📈 Analisi statistica...")
        stats_reporter = StatisticalAnalysisReporter()
        stats_report = stats_reporter.generate_comprehensive_report(summary, episode_results)
        
        print(f"✅ Analisi statistica completata")
        print(f"   - Test di significatività: {len(stats_report.significance_tests)}")
        print(f"   - Servizi di flessibilità analizzati: {len(stats_report.flexibility_analysis)}")
        
        # 4. Export completo
        print("\n📦 Export risultati...")
        export_manager = ExportManager(output_dir="results")
        
        # Export dati
        data_exports = export_manager.export_data_formats(summary, episode_results, stats_report)
        
        # Dashboard interattiva
        dashboard_path = export_manager.create_interactive_dashboard(summary, episode_results, stats_report)
        
        # PDF alta risoluzione
        pdf_exports = export_manager.export_high_resolution_pdfs(viz_files)
        
        # Archivio completo
        archive_path = export_manager.create_export_archive(
            viz_files, pdf_exports, data_exports, dashboard_path
        )
        
        print(f"✅ Export completato")
        print(f"   - Dashboard interattiva: {dashboard_path}")
        print(f"   - Archivio completo: {archive_path}")
        
        # 5. Risultati principali
        print("\n🏆 RISULTATI PRINCIPALI:")
        print("-" * 40)
        
        for dist, advantage in summary.profit_advantage.items():
            winner = "MILP" if advantage > 0 else "PPO"
            print(f"   {dist}: {advantage:+.1f} EUR (vantaggio {winner})")
        
        print(f"\nTempo di esecuzione medio:")
        ppo_time = summary.ppo_results.get('execution_time', {}).get('mean', 0)
        milp_time = summary.milp_results.get('execution_time', {}).get('mean', 0)
        print(f"   PPO: {ppo_time:.4f} secondi")
        print(f"   MILP: {milp_time:.4f} secondi")
        
        if milp_time > 0 and ppo_time > 0:
            ratio = milp_time / ppo_time
            print(f"   Rapporto MILP/PPO: {ratio:.1f}x")
        
        print(f"\n🎉 Confronto completato con successo!")
        print(f"📁 Tutti i risultati salvati in: results/")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)