"""
Entrenamiento básico de TouristBot con PPO
===========================================

Script para entrenar un agente con Proximal Policy Optimization (PPO)
usando Stable-Baselines3.

Uso:
    python train_ppo_basic.py
"""

import os
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
import gymnasium as gym

from touristbot_env import TouristBotEnv, PLACE_TYPES


# ============================================
# CONFIGURACIÓN DEL ENTRENAMIENTO
# ============================================

CONFIG = {
    # Entorno
    "use_partial_obs": True,
    "view_size": 5,
    "n_envs": 4,  # Entrenamiento paralelo
    
    # PPO Hyperparameters
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,  # Coeficiente de entropía para exploración
    
    # Entrenamiento
    "total_timesteps": 200000,
    "eval_freq": 10000,
    "save_freq": 20000,
    
    # Paths
    "log_dir": "./logs/ppo_basic/",
    "model_dir": "./models/ppo_basic/",
    "tensorboard_log": "./tensorboard/ppo_basic/",
}


def make_env(rank, seed=0):
    """
    Crea una función para crear el entorno (útil para paralelización)
    
    Args:
        rank: ID del entorno (para logging)
        seed: Seed base para reproducibilidad
    """
    def _init():
        env = TouristBotEnv(
            use_partial_obs=CONFIG["use_partial_obs"],
            view_size=CONFIG["view_size"],
            render_mode=None
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_ppo_basic():
    """
    Entrenamiento básico con PPO
    """
    print("="*70)
    print("🚀 ENTRENAMIENTO PPO BÁSICO - TOURISTBOT")
    print("="*70)
    print(f"\n⚙️  Configuración:")
    print(f"   - Vista parcial: {CONFIG['use_partial_obs']} ({CONFIG['view_size']}x{CONFIG['view_size']})")
    print(f"   - Entornos paralelos: {CONFIG['n_envs']}")
    print(f"   - Total timesteps: {CONFIG['total_timesteps']:,}")
    print(f"   - Learning rate: {CONFIG['learning_rate']}")
    print(f"   - Batch size: {CONFIG['batch_size']}")
    
    # Crear directorios
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    os.makedirs(CONFIG["tensorboard_log"], exist_ok=True)
    
    # Crear entornos vectorizados para entrenamiento
    print(f"\n📦 Creando {CONFIG['n_envs']} entornos vectorizados...")
    env = make_vec_env(
        make_env(rank=0),
        n_envs=CONFIG["n_envs"],
        seed=42
    )
    
    # Entorno de evaluación separado
    print(f"📊 Creando entorno de evaluación...")
    eval_env = make_vec_env(
        make_env(rank=100),
        n_envs=1,
        seed=123
    )
    
    # Callbacks para guardar y evaluar
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=CONFIG["model_dir"],
        log_path=CONFIG["log_dir"],
        eval_freq=CONFIG["eval_freq"],
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=CONFIG["save_freq"],
        save_path=CONFIG["model_dir"],
        name_prefix="ppo_touristbot"
    )
    
    # Crear modelo PPO
    print(f"\n🤖 Creando modelo PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=CONFIG["learning_rate"],
        n_steps=CONFIG["n_steps"],
        batch_size=CONFIG["batch_size"],
        n_epochs=CONFIG["n_epochs"],
        gamma=CONFIG["gamma"],
        gae_lambda=CONFIG["gae_lambda"],
        clip_range=CONFIG["clip_range"],
        ent_coef=CONFIG["ent_coef"],
        verbose=1,
        tensorboard_log=CONFIG["tensorboard_log"],
        device="auto"  # Usa GPU si está disponible
    )
    
    print(f"\n📊 Arquitectura de la política:")
    print(f"   - Input: {env.observation_space.shape[0]} (observación)")
    print(f"   - Output: {env.action_space.n} (acciones)")
    print(f"   - Policy: MLP (Red neuronal fully-connected)")
    
    # Entrenar
    print(f"\n🎓 Iniciando entrenamiento...")
    print(f"   TensorBoard: tensorboard --logdir {CONFIG['tensorboard_log']}")
    print(f"\n{'='*70}\n")
    
    timestamp_start = datetime.now()
    
    try:
        model.learn(
            total_timesteps=CONFIG["total_timesteps"],
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        
        timestamp_end = datetime.now()
        duration = timestamp_end - timestamp_start
        
        print(f"\n{'='*70}")
        print(f"✅ ENTRENAMIENTO COMPLETADO")
        print(f"{'='*70}")
        print(f"   Duración: {duration}")
        print(f"   Timesteps: {CONFIG['total_timesteps']:,}")
        
        # Guardar modelo final
        final_model_path = os.path.join(CONFIG["model_dir"], "ppo_touristbot_final")
        model.save(final_model_path)
        print(f"   Modelo guardado: {final_model_path}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Entrenamiento interrumpido por el usuario")
        interrupted_model_path = os.path.join(CONFIG["model_dir"], "ppo_touristbot_interrupted")
        model.save(interrupted_model_path)
        print(f"   Modelo guardado: {interrupted_model_path}")
    
    finally:
        env.close()
        eval_env.close()
    
    return model


def test_trained_model(model_path, n_episodes=5, render=True):
    """
    Prueba un modelo entrenado
    
    Args:
        model_path: Ruta al modelo guardado
        n_episodes: Número de episodios a ejecutar
        render: Si mostrar visualización
    """
    print(f"\n{'='*70}")
    print(f"🧪 PROBANDO MODELO ENTRENADO")
    print(f"{'='*70}")
    print(f"   Modelo: {model_path}")
    print(f"   Episodios: {n_episodes}\n")
    
    # Cargar modelo
    model = PPO.load(model_path)
    
    # Crear entorno
    env = TouristBotEnv(
        use_partial_obs=CONFIG["use_partial_obs"],
        view_size=CONFIG["view_size"],
        render_mode="human" if render else None
    )
    
    stats = {
        "success": 0,
        "timeout": 0,
        "steps": [],
        "rewards": []
    }
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        print(f"\n--- Episodio {episode + 1}/{n_episodes} ---")
        print(f"Objetivo: {env.goal_type} en {env.goal_pos}")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            if render:
                env.render()
        
        # Estadísticas
        if terminated:
            stats["success"] += 1
            result = "✅ Éxito"
        else:
            stats["timeout"] += 1
            result = "❌ Timeout"
        
        stats["steps"].append(steps)
        stats["rewards"].append(episode_reward)
        
        print(f"Resultado: {result}")
        print(f"Pasos: {steps}")
        print(f"Reward: {episode_reward:.2f}")
    
    env.close()
    
    # Resumen
    print(f"\n{'='*70}")
    print(f"📊 ESTADÍSTICAS")
    print(f"{'='*70}")
    print(f"   Tasa de éxito: {stats['success']}/{n_episodes} ({stats['success']/n_episodes*100:.1f}%)")
    print(f"   Pasos promedio: {np.mean(stats['steps']):.1f} ± {np.std(stats['steps']):.1f}")
    print(f"   Reward promedio: {np.mean(stats['rewards']):.2f} ± {np.std(stats['rewards']):.2f}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenar TouristBot con PPO")
    parser.add_argument("--train", action="store_true", help="Entrenar modelo")
    parser.add_argument("--test", type=str, help="Probar modelo (ruta al .zip)")
    parser.add_argument("--episodes", type=int, default=5, help="Episodios de prueba")
    parser.add_argument("--no-render", action="store_true", help="No mostrar visualización")
    
    args = parser.parse_args()
    
    if args.train:
        train_ppo_basic()
    elif args.test:
        test_trained_model(args.test, n_episodes=args.episodes, render=not args.no_render)
    else:
        print("Uso:")
        print("  python train_ppo_basic.py --train              # Entrenar")
        print("  python train_ppo_basic.py --test modelo.zip    # Probar")
