import os
import numpy as np
from datetime import datetime
import json
from typing import Dict, Any

from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import optuna

from touristbot_env import TouristBotEnv


# Configuration
CONFIGS = {
    "ppo_optimized": {
        "algorithm": "PPO",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    
    "ppo_exploration": {
        "algorithm": "PPO",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    
    "sac": {
        "algorithm": "SAC",
        "learning_rate": 3e-4,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "ent_coef": "auto",
    },
    
    "dqn": {
        "algorithm": "DQN",
        "learning_rate": 1e-4,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "batch_size": 32,
        "tau": 1.0,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.02,
    }
}


# Callbacks
class CurriculumCallback(BaseCallback):
   # Callback for curriculum learning
    def __init__(self, milestones: Dict[int, Dict[str, Any]], verbose=0):
        super().__init__(verbose)
        self.milestones = milestones
        
    def _on_step(self) -> bool:
        for timestep, changes in self.milestones.items():
            if self.num_timesteps == timestep:
                if self.verbose > 0:
                    print(f"\n🎓 Curriculum milestone {timestep}: {changes}")
                
                
                
        return True

# Callback to scale rewards dinamically
class RewardScalingCallback(BaseCallback):
    
    def __init__(self, scale_factor=1.0, verbose=0):
        super().__init__(verbose)
        self.scale_factor = scale_factor
        
    def _on_step(self) -> bool:
        return True



# Training functions for Curriculum learning
def train_with_curriculum():
    """
    Entrenamiento con Curriculum Learning
    
    Estrategia modificada (observation space compatible):
    1. Fácil: Vista parcial 5x5, max_steps=150 (más tiempo)
    2. Medio: Vista parcial 5x5, max_steps=100 (tiempo normal)
    3. Difícil: Vista parcial 5x5, max_steps=75 (menos tiempo)
    
    """
    print("="*70)
    print("🎓 CURRICULUM LEARNING")
    print("="*70)
    print("\nEstrategia: Reducir gradualmente el tiempo disponible")
    print("(Mantiene observation space compatible entre fases)\n")
    
    phases = [
        {
            "name": "Fase 1: Tiempo extendido (fácil)",
            "use_partial_obs": True,
            "view_size": 5,
            "max_steps": 150,
            "timesteps": 50000,
            "learning_rate": 3e-4,
        },
        {
            "name": "Fase 2: Tiempo normal (medio)",
            "use_partial_obs": True,
            "view_size": 5,
            "max_steps": 100,
            "timesteps": 75000,
            "learning_rate": 1e-4,
        },
        {
            "name": "Fase 3: Tiempo reducido (difícil)",
            "use_partial_obs": True,
            "view_size": 5,
            "max_steps": 75,
            "timesteps": 100000,
            "learning_rate": 5e-5,
        }
    ]
    
    model = None
    os.makedirs("./models/curriculum/", exist_ok=True)
    
    for i, phase in enumerate(phases):
        print(f"\n{'='*70}")
        print(f"{phase['name']}")
        print(f"   Max steps: {phase['max_steps']}")
        print(f"   Timesteps: {phase['timesteps']:,}")
        print(f"{'='*70}\n")
        
        # Crear entorno para esta fase
        def make_env_phase():
            env = TouristBotEnv(
                use_partial_obs=phase["use_partial_obs"],
                view_size=phase.get("view_size", 5),
                render_mode=None
            )
            # Modificar max_steps para esta fase
            env.max_steps = phase["max_steps"]
            return Monitor(env)
        
        env = make_vec_env(make_env_phase, n_envs=4, seed=42)
        
        if model is None:
            # Primera fase: crear modelo nuevo
            print("Creando modelo nuevo...")
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=phase["learning_rate"],
                verbose=1,
                tensorboard_log=f"./tensorboard/curriculum/phase_{i+1}/"
            )
        else:
            # Fases siguientes: continuar entrenamiento con nuevo entorno
            print("Continuando entrenamiento con nueva dificultad...")
            model.set_env(env)
            model.learning_rate = phase["learning_rate"]
        
        # Entrenar esta fase
        model.learn(
            total_timesteps=phase["timesteps"],
            progress_bar=True,
            reset_num_timesteps=False
        )
        
        # Guardar checkpoint
        checkpoint_path = f"./models/curriculum/phase_{i+1}_final"
        model.save(checkpoint_path)
        print(f"\nFase {i+1} completada. Modelo guardado: {checkpoint_path}")
        
        env.close()
    
    print(f"\n{'='*70}")
    print(f"CURRICULUM LEARNING COMPLETADO")
    print(f"{'='*70}")
    print(f"Modelos guardados en ./models/curriculum/")
    print(f"Mejor modelo: phase_3_final.zip (entrenado en todas las fases)")
    
    return model


def compare_algorithms(timesteps=100000):
    """
    Compara diferentes algoritmos de RL
    """
    print("="*70)
    print("COMPARACIÓN DE ALGORITMOS")
    print("="*70)
    
    algorithms = {
        "PPO": PPO,
        "SAC": SAC,
        "DQN": DQN
    }
    
    results = {}
    
    for algo_name, AlgoClass in algorithms.items():
        print(f"\n{'='*70}")
        print(f"Entrenando {algo_name}")
        print(f"{'='*70}\n")
        
        # Crear entorno
        def make_env():
            env = TouristBotEnv(use_partial_obs=True, view_size=5, render_mode=None)
            return Monitor(env)
        
        env = make_vec_env(make_env, n_envs=4, seed=42)
        eval_env = make_vec_env(make_env, n_envs=1, seed=123)
        
        # Crear directorio
        log_dir = f"./logs/comparison/{algo_name.lower()}/"
        os.makedirs(log_dir, exist_ok=True)
        
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"./models/comparison/{algo_name.lower()}/",
            log_path=log_dir,
            eval_freq=10000,
            deterministic=True,
            verbose=1
        )
        
        try:
            # Create model with specific configuration
            config = CONFIGS.get(algo_name.lower(), {})
            config_filtered = {k: v for k, v in config.items() if k != "algorithm"}
            
            model = AlgoClass(
                "MlpPolicy",
                env,
                **config_filtered,
                verbose=1,
                tensorboard_log=f"./tensorboard/comparison/"
            )
            
            # Entrenar
            model.learn(
                total_timesteps=timesteps,
                callback=eval_callback,
                progress_bar=True
            )
            
            # Guardar
            model.save(f"./models/comparison/{algo_name.lower()}_final")
            
            # Evaluar
            stats = evaluate_model(model, env=eval_env, n_episodes=50)
            results[algo_name] = stats
            
        except Exception as e:
            print(f"Error entrenando {algo_name}: {e}")
            results[algo_name] = {"error": str(e)}
        
        finally:
            env.close()
            eval_env.close()
    
    # Show comparison
    print(f"\n{'='*70}")
    print(f"RESULTADOS DE LA COMPARACIÓN")
    print(f"{'='*70}\n")
    
    for algo_name, stats in results.items():
        if "error" not in stats:
            print(f"{algo_name}:")
            print(f"  Tasa de éxito: {stats['success_rate']*100:.1f}%")
            print(f"  Pasos promedio: {stats['mean_steps']:.1f}")
            print(f"  Reward promedio: {stats['mean_reward']:.2f}\n")
        else:
            print(f"{algo_name}: Error - {stats['error']}\n")
    
    # Guardar resultados
    with open("./models/comparison/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def tune_hyperparameters(n_trials=50):
    """
    Optimización de hiperparámetros con Optuna
    """
    print("="*70)
    print("HYPERPARAMETER TUNING CON OPTUNA")
    print("="*70)
    
    def objective(trial):
        """Función objetivo para Optuna"""
        
        # Suggest hyperparameters
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        n_epochs = trial.suggest_int("n_epochs", 5, 15)
        gamma = trial.suggest_uniform("gamma", 0.95, 0.999)
        ent_coef = trial.suggest_loguniform("ent_coef", 1e-4, 1e-1)
        clip_range = trial.suggest_uniform("clip_range", 0.1, 0.3)
        
        # Crear entorno
        def make_env():
            env = TouristBotEnv(use_partial_obs=True, view_size=5, render_mode=None)
            return Monitor(env)
        
        env = make_vec_env(make_env, n_envs=4, seed=42)
        
        # Crear modelo
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            ent_coef=ent_coef,
            clip_range=clip_range,
            verbose=0
        )
        
        # Entrenar
        model.learn(total_timesteps=50000)
        
        # Evaluar
        eval_env = TouristBotEnv(use_partial_obs=True, view_size=5, render_mode=None)
        stats = evaluate_model(model, env=eval_env, n_episodes=20)
        
        env.close()
        eval_env.close()
        
        # Return metric to optimize (success rate)
        return stats["success_rate"]
    
    # Crear estudio
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n{'='*70}")
    print(f"MEJORES HIPERPARÁMETROS")
    print(f"{'='*70}\n")
    print(f"Tasa de éxito: {study.best_value*100:.1f}%\n")
    
    for param, value in study.best_params.items():
        print(f"{param}: {value}")
    
    # Guardar resultados
    os.makedirs("./models/tuning/", exist_ok=True)
    with open("./models/tuning/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    
    return study.best_params


def evaluate_model(model, env, n_episodes=50):
    """
    Evalúa un modelo entrenado
    """
    successes = 0
    steps_list = []
    rewards_list = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Para VecEnv
            if isinstance(done, np.ndarray):
                done = done[0]
            if isinstance(info, list):
                info = info[0]
        
        # Check if success
        if "TimeLimit.truncated" not in info or not info.get("TimeLimit.truncated", False):
            successes += 1
        
        steps_list.append(steps)
        rewards_list.append(episode_reward)
    
    return {
        "success_rate": successes / n_episodes,
        "mean_steps": np.mean(steps_list),
        "std_steps": np.std(steps_list),
        "mean_reward": np.mean(rewards_list),
        "std_reward": np.std(rewards_list)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenamiento avanzado de TouristBot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["curriculum", "compare", "tune"],
        required=True,
        help="Modo de entrenamiento"
    )
    parser.add_argument("--timesteps", type=int, default=100000, help="Timesteps totales")
    parser.add_argument("--trials", type=int, default=50, help="Trials para tuning")
    
    args = parser.parse_args()
    
    # Crear directorios
    os.makedirs("./models/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./tensorboard/", exist_ok=True)
    
    if args.mode == "curriculum":
        train_with_curriculum()
    elif args.mode == "compare":
        compare_algorithms(timesteps=args.timesteps)
    elif args.mode == "tune":
        tune_hyperparameters(n_trials=args.trials)
