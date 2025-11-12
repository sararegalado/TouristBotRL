"""
Utilidades para entrenamiento y evaluación
==========================================

Funciones auxiliares para:
- Configuración de entornos
- Wrappers personalizados
- Callbacks útiles
- Métricas customizadas
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import cv2


# ============================================
# WRAPPERS PERSONALIZADOS
# ============================================

class RewardScalingWrapper(gym.Wrapper):
    """
    Escala los rewards para mejorar el aprendizaje
    """
    def __init__(self, env, scale_factor=0.01):
        super().__init__(env)
        self.scale_factor = scale_factor
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        scaled_reward = reward * self.scale_factor
        return obs, scaled_reward, terminated, truncated, info


class TimeAwareObservationWrapper(gym.ObservationWrapper):
    """
    Añade información temporal a la observación
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Expandir observation space para incluir tiempo normalizado
        low = np.append(self.observation_space.low, 0.0)
        high = np.append(self.observation_space.high, 1.0)
        
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )
    
    def observation(self, obs):
        # Añadir tiempo normalizado (steps / max_steps)
        time_normalized = self.env.steps / self.env.max_steps
        return np.append(obs, time_normalized)


class FrameStackWrapper(gym.Wrapper):
    """
    Apila las últimas N observaciones
    Útil para dar información temporal al agente
    """
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = None
        
        # Modificar observation space
        low = np.repeat(self.observation_space.low, n_stack)
        high = np.repeat(self.observation_space.high, n_stack)
        
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.n_stack
        return self._get_stacked_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        self.frames.pop(0)
        return self._get_stacked_obs(), reward, terminated, truncated, info
    
    def _get_stacked_obs(self):
        return np.concatenate(self.frames)


# ============================================
# CALLBACKS PERSONALIZADOS
# ============================================

class SuccessRateCallback(BaseCallback):
    """
    Callback para trackear la tasa de éxito durante el entrenamiento
    """
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.success_count = 0
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Verificar si el episodio terminó
        if self.locals.get("dones") is not None:
            dones = self.locals["dones"]
            infos = self.locals["infos"]
            
            for done, info in zip(dones, infos):
                if done:
                    self.episode_count += 1
                    
                    # Verificar si fue éxito (no timeout)
                    if "TimeLimit.truncated" not in info or not info["TimeLimit.truncated"]:
                        self.success_count += 1
        
        # Log cada check_freq steps
        if self.n_calls % self.check_freq == 0 and self.episode_count > 0:
            success_rate = self.success_count / self.episode_count
            self.logger.record("custom/success_rate", success_rate)
            
            if self.verbose > 0:
                print(f"Success rate: {success_rate*100:.1f}% ({self.success_count}/{self.episode_count})")
        
        return True


class ProgressBarCallback(BaseCallback):
    """
    Callback para mostrar barra de progreso personalizada
    """
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.last_print = 0
        
    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        
        # Actualizar cada 5%
        if progress - self.last_print >= 0.05:
            self.last_print = progress
            bar_length = 40
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\rProgreso: [{bar}] {progress*100:.1f}% - {self.num_timesteps}/{self.total_timesteps}", end='')
            
            if progress >= 1.0:
                print()  # Nueva línea al terminar
        
        return True


class SaveBestModelCallback(BaseCallback):
    """
    Guarda el mejor modelo basado en reward promedio
    """
    def __init__(self, save_path, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # Recolectar rewards
        if "rewards" in self.locals:
            self.episode_rewards.extend(self.locals["rewards"])
        
        # Verificar cada check_freq steps
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards)
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                
                if self.verbose > 0:
                    print(f"\n🏆 Nuevo mejor modelo! Reward promedio: {mean_reward:.2f}")
                
                self.model.save(self.save_path)
            
            self.episode_rewards = []
        
        return True


# ============================================
# FUNCIONES DE EVALUACIÓN
# ============================================

def evaluate_policy_detailed(model, env, n_eval_episodes=100, deterministic=True):
    """
    Evaluación detallada de una política
    
    Returns:
        dict con métricas completas
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    timeout_count = 0
    
    for i in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Para VecEnv
            if isinstance(done, np.ndarray):
                done = done[0]
            if isinstance(info, list):
                info = info[0]
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Clasificar resultado
        if "TimeLimit.truncated" in info and info["TimeLimit.truncated"]:
            timeout_count += 1
        else:
            success_count += 1
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": success_count / n_eval_episodes,
        "timeout_rate": timeout_count / n_eval_episodes,
        "episodes": n_eval_episodes
    }


def visualize_episode(model, env, save_video=False, video_path="episode.mp4"):
    """
    Visualiza un episodio completo y opcionalmente guarda video
    """
    obs, info = env.reset()
    done = False
    frames = []
    
    print(f"Objetivo: {env.goal_type} en {env.goal_pos}")
    print(f"Inicio: {env.agent_pos}\n")
    
    while not done:
        # Renderizar
        frame = env.render()
        if save_video and frame is not None:
            frames.append(frame)
        
        # Acción del modelo
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        print(f"Step {env.steps}: Pos={env.agent_pos}, Action={action}, Reward={reward:.2f}")
    
    print(f"\nResultado: {'✅ Éxito' if terminated else '❌ Timeout'}")
    print(f"Pasos totales: {env.steps}")
    print(f"Reward total: {env.total_reward:.2f}")
    
    # Guardar video
    if save_video and len(frames) > 0:
        import cv2
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"\n🎥 Video guardado: {video_path}")


def compare_policies(models_dict, env, n_episodes=50):
    """
    Compara múltiples políticas
    
    Args:
        models_dict: Dict {nombre: modelo}
        env: Entorno de evaluación
        n_episodes: Episodios por modelo
    
    Returns:
        DataFrame con comparación
    """
    import pandas as pd
    
    results = []
    
    for name, model in models_dict.items():
        print(f"Evaluando {name}...")
        stats = evaluate_policy_detailed(model, env, n_episodes)
        stats["model"] = name
        results.append(stats)
    
    df = pd.DataFrame(results)
    df = df.set_index("model")
    
    return df


# ============================================
# UTILIDADES DE CONFIGURACIÓN
# ============================================

def create_env_with_wrappers(base_env_fn, wrappers=None):
    """
    Crea entorno con wrappers aplicados
    
    Args:
        base_env_fn: Función que crea el entorno base
        wrappers: Lista de wrappers a aplicar
    
    Returns:
        Entorno con wrappers
    """
    env = base_env_fn()
    
    if wrappers:
        for wrapper_class, kwargs in wrappers:
            env = wrapper_class(env, **kwargs)
    
    return env


def get_optimal_n_envs():
    """
    Determina número óptimo de entornos paralelos
    basado en CPUs disponibles
    """
    import multiprocessing
    n_cpus = multiprocessing.cpu_count()
    
    # Usar mitad de los CPUs disponibles (reservar para otros procesos)
    optimal = max(1, n_cpus // 2)
    
    print(f"CPUs detectados: {n_cpus}")
    print(f"Entornos paralelos recomendados: {optimal}")
    
    return optimal


# ============================================
# MAIN (EJEMPLOS)
# ============================================

if __name__ == "__main__":
    print("Utilidades de entrenamiento cargadas")
    print("\nEjemplos de uso:")
    print("\n1. Wrapper de reward scaling:")
    print("   env = RewardScalingWrapper(env, scale_factor=0.01)")
    print("\n2. Callback de tasa de éxito:")
    print("   callback = SuccessRateCallback(check_freq=1000)")
    print("\n3. Evaluación detallada:")
    print("   stats = evaluate_policy_detailed(model, env, n_eval_episodes=100)")
    print("\n4. Número óptimo de entornos:")
    print(f"   n_envs = {get_optimal_n_envs()}")
