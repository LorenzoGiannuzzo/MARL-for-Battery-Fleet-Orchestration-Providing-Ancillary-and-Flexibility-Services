"""
Enhanced PPO Pre-training with Improved Environment
Addestra il PPO con l'ambiente migliorato per competere meglio con MILP
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import os
from pathlib import Path
import torch.nn as nn

# Import components
from enhanced_ppo_environment import EnhancedBatteryTradingEnv, BackwardCompatibilityWrapper
from drl_flexibility_analysis import ForecastErrorGenerator
from flexibility_market import FlexibilityMarket


def generate_price_data(n_hours):
    """Generate synthetic price data for training"""
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


def create_enhanced_training_env(prices, error_distribution='normal', random_seed=42):
    """Create enhanced training environment"""
    error_gen = ForecastErrorGenerator(error_distribution)
    env = EnhancedBatteryTradingEnv(
        prices=prices,
        error_generator=error_gen,
        flexibility_enabled=True,
        random_seed=random_seed
    )
    return env


def create_curriculum_training_data():
    """Create curriculum learning data with increasing complexity"""
    
    print("📚 Creazione curriculum di training...")
    
    # Stage 1: Simple patterns (high FCR prices)
    simple_prices = []
    for _ in range(100):  # 100 episodes
        base_price = 50.0
        daily_pattern = [base_price + 10 * np.sin(2 * np.pi * h / 24) for h in range(24)]
        simple_prices.extend(daily_pattern * 20)  # 20 days per episode
    
    # Stage 2: Moderate complexity (realistic price variations)
    moderate_prices = []
    for _ in range(150):  # 150 episodes
        moderate_prices.extend(generate_price_data(480))  # 20 days per episode
    
    # Stage 3: High complexity (volatile markets)
    complex_prices = []
    for _ in range(100):  # 100 episodes
        base_prices = generate_price_data(480)
        # Add volatility
        volatility = np.random.normal(0, 15, len(base_prices))
        volatile_prices = [max(10, p + v) for p, v in zip(base_prices, volatility)]
        complex_prices.extend(volatile_prices)
    
    return {
        'simple': simple_prices,
        'moderate': moderate_prices,
        'complex': complex_prices
    }


def train_enhanced_ppo_curriculum():
    """Train PPO with curriculum learning for better performance"""
    
    print("🚀 ENHANCED PPO TRAINING CON CURRICULUM LEARNING")
    print("=" * 60)
    
    # Create curriculum data
    curriculum_data = create_curriculum_training_data()
    
    # Training configuration
    config = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,  # Encourage exploration
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': {
            'net_arch': [256, 256, 128],  # Larger network
            'activation_fn': nn.Tanh
        }
    }
    
    # Create directories
    os.makedirs('ppo_models', exist_ok=True)
    os.makedirs('enhanced_training_logs', exist_ok=True)
    
    # Stage 1: Simple patterns training
    print("\n📖 STAGE 1: Training su pattern semplici (FCR focus)")
    print("-" * 50)
    
    env_simple = create_enhanced_training_env(curriculum_data['simple'], 'normal', 42)
    env_simple = Monitor(env_simple, 'enhanced_training_logs/stage1')
    
    model = PPO(
        'MlpPolicy',
        env_simple,
        verbose=1,
        tensorboard_log='enhanced_training_logs/tensorboard',
        **config
    )
    
    # Train stage 1
    model.learn(
        total_timesteps=200000,  # 200k steps
        tb_log_name='enhanced_ppo_stage1',
        progress_bar=True
    )
    
    model.save('ppo_models/enhanced_ppo_stage1')
    print("✅ Stage 1 completato!")
    
    # Stage 2: Moderate complexity
    print("\n📖 STAGE 2: Training su complessità moderata")
    print("-" * 50)
    
    env_moderate = create_enhanced_training_env(curriculum_data['moderate'], 'normal', 43)
    env_moderate = Monitor(env_moderate, 'enhanced_training_logs/stage2')
    
    # Continue training from stage 1
    model.set_env(env_moderate)
    model.learn(
        total_timesteps=300000,  # 300k additional steps
        tb_log_name='enhanced_ppo_stage2',
        progress_bar=True,
        reset_num_timesteps=False
    )
    
    model.save('ppo_models/enhanced_ppo_stage2')
    print("✅ Stage 2 completato!")
    
    # Stage 3: High complexity with multiple error distributions
    print("\n📖 STAGE 3: Training su alta complessità (multi-distribution)")
    print("-" * 50)
    
    # Train on multiple error distributions
    error_distributions = ['normal', 'uniform', 'ornstein-uhlenbeck']
    
    for i, dist in enumerate(error_distributions):
        print(f"  Training con distribuzione: {dist}")
        
        env_complex = create_enhanced_training_env(curriculum_data['complex'], dist, 44 + i)
        env_complex = Monitor(env_complex, f'enhanced_training_logs/stage3_{dist}')
        
        model.set_env(env_complex)
        model.learn(
            total_timesteps=200000,  # 200k steps per distribution
            tb_log_name=f'enhanced_ppo_stage3_{dist}',
            progress_bar=True,
            reset_num_timesteps=False
        )
    
    model.save('ppo_models/enhanced_ppo_stage3')
    print("✅ Stage 3 completato!")
    
    # Final fine-tuning stage
    print("\n📖 STAGE 4: Fine-tuning finale")
    print("-" * 50)
    
    # Create final training environment with realistic annual data
    annual_prices = []
    for month in range(1, 13):
        monthly_prices = generate_price_data(24 * 30)  # 30 days per month
        annual_prices.extend(monthly_prices)
    
    env_final = create_enhanced_training_env(annual_prices, 'normal', 50)
    env_final = Monitor(env_final, 'enhanced_training_logs/final')
    
    # Fine-tuning with reduced learning rate
    model.learning_rate = 1e-4  # Reduced learning rate for fine-tuning
    model.set_env(env_final)
    
    model.learn(
        total_timesteps=500000,  # 500k steps for final training
        tb_log_name='enhanced_ppo_final',
        progress_bar=True,
        reset_num_timesteps=False
    )
    
    # Save final model
    model.save('ppo_models/enhanced_ppo_final')
    print("✅ Fine-tuning completato!")
    
    return model


def evaluate_enhanced_model():
    """Evaluate the enhanced model performance"""
    
    print("\n🔍 VALUTAZIONE MODELLO ENHANCED")
    print("=" * 40)
    
    # Load the final model
    model = PPO.load('ppo_models/enhanced_ppo_final')
    
    # Create evaluation environment
    eval_prices = generate_price_data(480)  # 20 days
    eval_env = create_enhanced_training_env(eval_prices, 'normal', 100)
    
    # Run evaluation episodes
    n_eval_episodes = 10
    episode_rewards = []
    episode_fcr_usage = []
    episode_revenues = []
    
    for episode in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        episode_fcr_total = 0
        episode_revenue_total = 0
        steps = 0
        
        done = False
        while not done and steps < 480:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            
            episode_reward += reward
            episode_fcr_total += info.get('reserved_fcr', 0)
            episode_revenue_total += info.get('flexibility_revenue', 0)
            steps += 1
            
            if truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_fcr_usage.append(episode_fcr_total / steps if steps > 0 else 0)
        episode_revenues.append(episode_revenue_total)
        
        print(f"  Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"FCR Usage={episode_fcr_usage[-1]:.3f} MW, "
              f"Revenue={episode_revenue_total:.2f} EUR")
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    avg_fcr_usage = np.mean(episode_fcr_usage)
    avg_revenue = np.mean(episode_revenues)
    
    print(f"\n📊 RISULTATI VALUTAZIONE:")
    print(f"  Reward medio: {avg_reward:.2f}")
    print(f"  FCR usage medio: {avg_fcr_usage:.3f} MW")
    print(f"  Revenue medio: {avg_revenue:.2f} EUR")
    print(f"  Std reward: {np.std(episode_rewards):.2f}")
    
    return {
        'avg_reward': avg_reward,
        'avg_fcr_usage': avg_fcr_usage,
        'avg_revenue': avg_revenue,
        'episode_rewards': episode_rewards
    }


def create_monthly_models():
    """Create specialized models for each month"""
    
    print("\n📅 CREAZIONE MODELLI MENSILI SPECIALIZZATI")
    print("=" * 50)
    
    # Load base model
    base_model = PPO.load('ppo_models/enhanced_ppo_final')
    
    # Create monthly models
    for month in range(1, 13):
        print(f"\n🗓️  Training modello per mese {month}")
        
        # Generate month-specific training data
        monthly_prices = []
        for _ in range(50):  # 50 episodes per month
            episode_prices = generate_price_data(480)
            # Add seasonal variation based on month
            if month in [12, 1, 2]:  # Winter
                seasonal_multiplier = 1.2
            elif month in [6, 7, 8]:  # Summer
                seasonal_multiplier = 0.9
            else:
                seasonal_multiplier = 1.0
            
            adjusted_prices = [p * seasonal_multiplier for p in episode_prices]
            monthly_prices.extend(adjusted_prices)
        
        # Create month-specific environment
        env_monthly = create_enhanced_training_env(monthly_prices, 'normal', month)
        env_monthly = Monitor(env_monthly, f'enhanced_training_logs/month_{month}')
        
        # Clone base model for monthly training
        monthly_model = PPO.load('ppo_models/enhanced_ppo_final')
        monthly_model.set_env(env_monthly)
        monthly_model.learning_rate = 5e-5  # Very low learning rate for fine-tuning
        
        # Fine-tune for this month
        monthly_model.learn(
            total_timesteps=100000,  # 100k steps per month
            tb_log_name=f'enhanced_ppo_month_{month}',
            progress_bar=True
        )
        
        # Save monthly model
        monthly_model.save(f'ppo_models/enhanced_ppo_month_{month}')
        print(f"✅ Modello mese {month} salvato!")
    
    print("\n✅ Tutti i modelli mensili creati!")


def main():
    """Main training pipeline"""
    
    print("🎯 ENHANCED PPO TRAINING PIPELINE")
    print("=" * 60)
    print("Obiettivo: Migliorare le performance PPO per competere con MILP")
    print("Strategia: Curriculum learning + reward shaping + FCR focus")
    print()
    
    # Step 1: Curriculum training
    print("STEP 1: Curriculum Learning Training")
    model = train_enhanced_ppo_curriculum()
    
    # Step 2: Evaluation
    print("\nSTEP 2: Model Evaluation")
    eval_results = evaluate_enhanced_model()
    
    # Step 3: Monthly specialization
    print("\nSTEP 3: Monthly Model Specialization")
    create_monthly_models()
    
    # Step 4: Final summary
    print("\n🎉 TRAINING COMPLETATO!")
    print("=" * 60)
    print("Modelli creati:")
    print("  - enhanced_ppo_final.zip (modello principale)")
    print("  - enhanced_ppo_month_X.zip (modelli mensili specializzati)")
    print()
    print("Miglioramenti implementati:")
    print("  ✅ Reward shaping per preferire FCR")
    print("  ✅ Action space semplificato e focalizzato")
    print("  ✅ Observation space arricchito con revenue density")
    print("  ✅ Curriculum learning per apprendimento graduale")
    print("  ✅ Modelli mensili specializzati")
    print()
    print(f"Performance finale: {eval_results['avg_reward']:.2f} reward medio")
    print(f"FCR usage medio: {eval_results['avg_fcr_usage']:.3f} MW")
    print()
    print("🚀 Pronto per il confronto con MILP!")


if __name__ == "__main__":
    main()