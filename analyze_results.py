"""
Análisis y visualización de resultados de entrenamiento
=======================================================

Herramientas para analizar y visualizar:
- Curvas de aprendizaje
- Comparación de modelos
- Matrices de confusión
- Mapas de calor de acciones

Uso:
    python analyze_results.py --plot-learning
    python analyze_results.py --compare-models
    python analyze_results.py --visualize-policy modelo.zip
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import load_results
from touristbot_env import TouristBotEnv
import pandas as pd
import json


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_learning_curves(log_dir, save_path="./plots/learning_curves.png"):
    """
    Plotea curvas de aprendizaje desde logs de Monitor
    """
    print(f"📊 Generando curvas de aprendizaje desde {log_dir}...")
    
    try:
        df = load_results(log_dir)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Curvas de Aprendizaje - TouristBot PPO', fontsize=16, fontweight='bold')
        
        # Rewards por episodio
        axes[0, 0].plot(df['r'].rolling(window=10).mean(), label='Reward (MA-10)')
        axes[0, 0].fill_between(
            range(len(df)),
            df['r'].rolling(window=10).mean() - df['r'].rolling(window=10).std(),
            df['r'].rolling(window=10).mean() + df['r'].rolling(window=10).std(),
            alpha=0.3
        )
        axes[0, 0].set_xlabel('Episodio')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Recompensa por Episodio')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Longitud de episodio
        axes[0, 1].plot(df['l'].rolling(window=10).mean(), color='orange', label='Steps (MA-10)')
        axes[0, 1].set_xlabel('Episodio')
        axes[0, 1].set_ylabel('Pasos')
        axes[0, 1].set_title('Pasos por Episodio')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Tiempo acumulado
        axes[1, 0].plot(df['t'].cumsum() / 3600, color='green')
        axes[1, 0].set_xlabel('Episodio')
        axes[1, 0].set_ylabel('Horas')
        axes[1, 0].set_title('Tiempo Acumulado de Entrenamiento')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distribución de rewards
        axes[1, 1].hist(df['r'], bins=50, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(df['r'].mean(), color='red', linestyle='--', label=f'Media: {df["r"].mean():.2f}')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].set_title('Distribución de Rewards')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {save_path}")
        plt.show()
        
    except Exception as e:
        print(f"❌ Error generando curvas: {e}")


def compare_models(results_json="./models/comparison/results.json", save_path="./plots/model_comparison.png"):
    """
    Compara rendimiento de diferentes modelos
    """
    print(f"⚖️  Comparando modelos desde {results_json}...")
    
    try:
        with open(results_json, 'r') as f:
            results = json.load(f)
        
        # Preparar datos
        models = []
        success_rates = []
        mean_steps = []
        mean_rewards = []
        
        for model_name, stats in results.items():
            if "error" not in stats:
                models.append(model_name)
                success_rates.append(stats['success_rate'] * 100)
                mean_steps.append(stats['mean_steps'])
                mean_rewards.append(stats['mean_reward'])
        
        # Crear gráfico
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Comparación de Algoritmos', fontsize=16, fontweight='bold')
        
        # Tasa de éxito
        axes[0].bar(models, success_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0].set_ylabel('Tasa de Éxito (%)')
        axes[0].set_title('Tasa de Éxito por Algoritmo')
        axes[0].set_ylim([0, 100])
        for i, v in enumerate(success_rates):
            axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # Pasos promedio
        axes[1].bar(models, mean_steps, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1].set_ylabel('Pasos Promedio')
        axes[1].set_title('Eficiencia (menos es mejor)')
        for i, v in enumerate(mean_steps):
            axes[1].text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')
        
        # Reward promedio
        axes[2].bar(models, mean_rewards, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[2].set_ylabel('Reward Promedio')
        axes[2].set_title('Reward Total por Episodio')
        for i, v in enumerate(mean_rewards):
            axes[2].text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {save_path}")
        plt.show()
        
    except Exception as e:
        print(f"❌ Error comparando modelos: {e}")


def visualize_policy(model_path, n_episodes=10, save_path="./plots/policy_heatmap.png"):
    """
    Visualiza la política del agente como mapa de calor de acciones
    """
    print(f"🗺️  Visualizando política de {model_path}...")
    
    try:
        model = PPO.load(model_path)
        env = TouristBotEnv(use_partial_obs=False, render_mode=None)
        
        # Matriz para contar visitas a cada posición
        visit_count = np.zeros((10, 10))
        action_count = np.zeros((10, 10, 4))  # [x, y, acción]
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            
            while not done:
                x, y = env.agent_pos
                visit_count[y, x] += 1
                
                action, _ = model.predict(obs, deterministic=True)
                action_count[y, x, action] += 1
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
        
        env.close()
        
        # Crear visualización
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Análisis de Política', fontsize=16, fontweight='bold')
        
        # Mapa de calor de visitas
        sns.heatmap(visit_count, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Visitas'})
        axes[0].set_title('Frecuencia de Visitas por Celda')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        
        # Acción preferida por celda
        preferred_action = np.argmax(action_count, axis=2)
        action_names = ['↑', '↓', '←', '→']
        
        # Crear matriz de texto con flechas
        action_text = np.empty_like(preferred_action, dtype=object)
        for i in range(10):
            for j in range(10):
                if visit_count[i, j] > 0:
                    action_text[i, j] = action_names[preferred_action[i, j]]
                else:
                    action_text[i, j] = ''
        
        sns.heatmap(visit_count > 0, annot=action_text, fmt='', cmap='Blues', 
                    ax=axes[1], cbar=False, linewidths=0.5)
        axes[1].set_title('Acción Preferida por Celda')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {save_path}")
        plt.show()
        
    except Exception as e:
        print(f"❌ Error visualizando política: {e}")


def analyze_trajectories(model_path, n_episodes=5, save_path="./plots/trajectories.png"):
    """
    Visualiza las trayectorias del agente en el grid
    """
    print(f"🛤️  Analizando trayectorias de {model_path}...")
    
    try:
        model = PPO.load(model_path)
        env = TouristBotEnv(use_partial_obs=False, render_mode=None)
        
        fig, axes = plt.subplots(1, n_episodes, figsize=(4*n_episodes, 4))
        if n_episodes == 1:
            axes = [axes]
        
        fig.suptitle('Trayectorias del Agente', fontsize=16, fontweight='bold')
        
        for ep, ax in enumerate(axes):
            obs, info = env.reset(seed=ep)
            trajectory = [env.agent_pos.copy()]
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                trajectory.append(env.agent_pos.copy())
                done = terminated or truncated
            
            # Plotear trayectoria
            trajectory = np.array(trajectory)
            
            # Grid
            for i in range(11):
                ax.axhline(i, color='gray', linewidth=0.5, alpha=0.3)
                ax.axvline(i, color='gray', linewidth=0.5, alpha=0.3)
            
            # Lugares
            for place_type, pos in env.places.items():
                color = 'orange' if place_type == 'restaurant' else 'red'
                ax.scatter(pos[0]+0.5, pos[1]+0.5, s=300, marker='s', 
                          color=color, alpha=0.6, label=place_type)
            
            # Trayectoria
            ax.plot(trajectory[:, 0]+0.5, trajectory[:, 1]+0.5, 
                   'b-', linewidth=2, alpha=0.7, label='Trayectoria')
            ax.scatter(trajectory[0, 0]+0.5, trajectory[0, 1]+0.5, 
                      s=200, marker='o', color='green', label='Inicio', zorder=5)
            ax.scatter(trajectory[-1, 0]+0.5, trajectory[-1, 1]+0.5, 
                      s=200, marker='*', color='gold', label='Fin', zorder=5)
            
            ax.set_xlim([0, 10])
            ax.set_ylim([0, 10])
            ax.set_aspect('equal')
            ax.set_title(f'Episodio {ep+1} ({len(trajectory)} pasos)')
            ax.legend(loc='upper right', fontsize=8)
            ax.invert_yaxis()
        
        env.close()
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {save_path}")
        plt.show()
        
    except Exception as e:
        print(f"❌ Error analizando trayectorias: {e}")


def generate_full_report(model_path, log_dir):
    """
    Genera un reporte completo con todas las visualizaciones
    """
    print("="*70)
    print("📈 GENERANDO REPORTE COMPLETO")
    print("="*70)
    
    os.makedirs("./plots/", exist_ok=True)
    
    print("\n1. Curvas de aprendizaje...")
    plot_learning_curves(log_dir)
    
    print("\n2. Visualización de política...")
    visualize_policy(model_path)
    
    print("\n3. Análisis de trayectorias...")
    analyze_trajectories(model_path)
    
    print("\n✅ Reporte completo generado en ./plots/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Análisis de resultados de TouristBot")
    parser.add_argument("--plot-learning", type=str, help="Plotear curvas de aprendizaje (log_dir)")
    parser.add_argument("--compare-models", type=str, help="Comparar modelos (results.json)")
    parser.add_argument("--visualize-policy", type=str, help="Visualizar política (modelo.zip)")
    parser.add_argument("--trajectories", type=str, help="Analizar trayectorias (modelo.zip)")
    parser.add_argument("--full-report", nargs=2, metavar=('MODEL', 'LOGDIR'), 
                       help="Generar reporte completo")
    parser.add_argument("--episodes", type=int, default=10, help="Episodios para análisis")
    
    args = parser.parse_args()
    
    if args.plot_learning:
        plot_learning_curves(args.plot_learning)
    elif args.compare_models:
        compare_models(args.compare_models)
    elif args.visualize_policy:
        visualize_policy(args.visualize_policy, n_episodes=args.episodes)
    elif args.trajectories:
        analyze_trajectories(args.trajectories, n_episodes=args.episodes)
    elif args.full_report:
        generate_full_report(args.full_report[0], args.full_report[1])
    else:
        print("Uso:")
        print("  python analyze_results.py --plot-learning logs/ppo_basic/")
        print("  python analyze_results.py --compare-models models/comparison/results.json")
        print("  python analyze_results.py --visualize-policy models/ppo_basic/best_model.zip")
        print("  python analyze_results.py --trajectories models/ppo_basic/best_model.zip")
        print("  python analyze_results.py --full-report modelo.zip logs/")
