"""
Demo Interactiva de TouristBot
==============================

Script para demostración interactiva del agente entrenado.
Permite al usuario ver el agente en acción con diferentes objetivos.

Uso:
    python demo.py --model models/ppo_basic/best_model.zip
    python demo.py --model models/ppo_basic/best_model.zip --continuous
"""

import argparse
import time
import numpy as np
from stable_baselines3 import PPO
from touristbot_env import TouristBotEnv, PLACE_TYPES
import cv2


def print_banner():
    """Imprime banner de inicio"""
    print("\n" + "="*70)
    print(" " * 20 + "🌆 TOURISTBOT DEMO 🌆")
    print("="*70 + "\n")


def print_menu():
    """Muestra menú de opciones"""
    print("\n📋 MENÚ:")
    print("  1. Episodio con objetivo restaurant")
    print("  2. Episodio con objetivo museum")
    print("  3. Episodio aleatorio")
    print("  4. Comparar 5 episodios")
    print("  5. Modo continuo (hasta interrumpir)")
    print("  q. Salir")
    print()


def run_episode(model, goal_type=None, render=True, verbose=True):
    """
    Ejecuta un episodio completo
    
    Args:
        model: Modelo entrenado
        goal_type: Tipo de objetivo o None para aleatorio
        render: Si mostrar visualización
        verbose: Si mostrar información detallada
    
    Returns:
        dict con estadísticas del episodio
    """
    if goal_type is None:
        goal_type = np.random.choice(PLACE_TYPES)
    
    env = TouristBotEnv(
        goal_type=goal_type,
        use_partial_obs=True,
        view_size=5,
        render_mode="human" if render else None
    )
    
    obs, info = env.reset()
    done = False
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🎯 NUEVO EPISODIO")
        print(f"{'='*70}")
        print(f"Objetivo: {env.goal_type.upper()} en posición {env.goal_pos}")
        print(f"Inicio: {env.agent_pos}")
        print(f"Distancia inicial: {env._manhattan_distance(env.agent_pos, env.goal_pos)} celdas")
        print(f"{'='*70}\n")
    
    trajectory = [env.agent_pos.copy()]
    actions_taken = []
    rewards_received = []
    
    action_names = ["↑ Arriba", "↓ Abajo", "← Izquierda", "→ Derecha"]
    
    while not done:
        # Predicción del modelo
        action, _states = model.predict(obs, deterministic=True)
        
        # Ejecutar acción
        obs, reward, terminated, truncated, info = env.step(action)
        
        trajectory.append(env.agent_pos.copy())
        actions_taken.append(action)
        rewards_received.append(reward)
        
        done = terminated or truncated
        
        # Mostrar información si verbose
        if verbose and env.steps % 5 == 0:  # Cada 5 pasos
            print(f"Paso {env.steps:3d}: {action_names[action]:12s} | "
                  f"Pos {env.agent_pos} | "
                  f"Dist {info['distance_to_goal']:2d} | "
                  f"Reward {reward:+6.2f}")
        
        # Renderizar
        if render:
            env.render()
            time.sleep(0.1)  # Pausa para visualización
    
    # Resultado
    if verbose:
        print(f"\n{'='*70}")
        if terminated:
            print("🎉 ¡OBJETIVO ALCANZADO!")
        else:
            print("⏰ Tiempo agotado")
        print(f"{'='*70}")
        print(f"Pasos totales: {env.steps}")
        print(f"Reward total: {env.total_reward:.2f}")
        print(f"Celdas exploradas: {len(env.visited_cells)}")
        print(f"{'='*70}\n")
    
    env.close()
    
    return {
        "success": terminated,
        "steps": env.steps,
        "total_reward": env.total_reward,
        "cells_explored": len(env.visited_cells),
        "trajectory": trajectory,
        "actions": actions_taken,
        "rewards": rewards_received
    }


def compare_episodes(model, n_episodes=5):
    """
    Ejecuta múltiples episodios y compara resultados
    """
    print(f"\n{'='*70}")
    print(f"📊 COMPARANDO {n_episodes} EPISODIOS")
    print(f"{'='*70}\n")
    
    results = []
    
    for i in range(n_episodes):
        print(f"\n--- Episodio {i+1}/{n_episodes} ---")
        stats = run_episode(model, goal_type=None, render=False, verbose=False)
        results.append(stats)
        
        result_icon = "✅" if stats["success"] else "❌"
        print(f"{result_icon} {stats['steps']:2d} pasos | "
              f"Reward: {stats['total_reward']:6.2f} | "
              f"Exploradas: {stats['cells_explored']:2d} celdas")
    
    # Estadísticas agregadas
    success_count = sum(1 for r in results if r["success"])
    avg_steps = np.mean([r["steps"] for r in results])
    avg_reward = np.mean([r["total_reward"] for r in results])
    
    print(f"\n{'='*70}")
    print(f"📈 ESTADÍSTICAS AGREGADAS")
    print(f"{'='*70}")
    print(f"Tasa de éxito: {success_count}/{n_episodes} ({success_count/n_episodes*100:.1f}%)")
    print(f"Pasos promedio: {avg_steps:.1f} ± {np.std([r['steps'] for r in results]):.1f}")
    print(f"Reward promedio: {avg_reward:.2f} ± {np.std([r['total_reward'] for r in results]):.2f}")
    print(f"{'='*70}\n")


def continuous_mode(model):
    """
    Modo continuo: ejecuta episodios hasta interrumpir
    """
    print(f"\n{'='*70}")
    print("🔄 MODO CONTINUO")
    print("{'='*70}")
    print("Presiona Ctrl+C para detener\n")
    
    episode_count = 0
    success_count = 0
    
    try:
        while True:
            episode_count += 1
            print(f"\n{'─'*70}")
            print(f"Episodio #{episode_count}")
            print(f"{'─'*70}")
            
            stats = run_episode(model, goal_type=None, render=True, verbose=False)
            
            if stats["success"]:
                success_count += 1
            
            result = "✅ Éxito" if stats["success"] else "❌ Timeout"
            print(f"{result} | {stats['steps']} pasos | Reward: {stats['total_reward']:.2f}")
            print(f"Tasa de éxito acumulada: {success_count}/{episode_count} ({success_count/episode_count*100:.1f}%)")
            
            time.sleep(1)  # Pausa entre episodios
            
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print("🛑 Modo continuo detenido")
        print(f"{'='*70}")
        print(f"Episodios ejecutados: {episode_count}")
        print(f"Tasa de éxito final: {success_count}/{episode_count} ({success_count/episode_count*100:.1f}%)")
        print(f"{'='*70}\n")


def interactive_demo(model_path):
    """
    Demo interactiva con menú
    """
    print_banner()
    
    # Cargar modelo
    print("📦 Cargando modelo...")
    try:
        model = PPO.load(model_path)
        print(f"✅ Modelo cargado: {model_path}\n")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # Loop del menú
    while True:
        print_menu()
        choice = input("Selecciona una opción: ").strip().lower()
        
        if choice == "1":
            run_episode(model, goal_type="restaurant", render=True, verbose=True)
        
        elif choice == "2":
            run_episode(model, goal_type="museum", render=True, verbose=True)
        
        elif choice == "3":
            run_episode(model, goal_type=None, render=True, verbose=True)
        
        elif choice == "4":
            compare_episodes(model, n_episodes=5)
        
        elif choice == "5":
            continuous_mode(model)
        
        elif choice == "q":
            print("\n👋 ¡Hasta luego!\n")
            break
        
        else:
            print("\n❌ Opción inválida. Intenta de nuevo.")


def main():
    parser = argparse.ArgumentParser(description="Demo interactiva de TouristBot")
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo (.zip)")
    parser.add_argument("--continuous", action="store_true", help="Modo continuo directo")
    parser.add_argument("--episodes", type=int, default=1, help="Número de episodios (si no es interactivo)")
    parser.add_argument("--goal", type=str, choices=PLACE_TYPES, help="Tipo de objetivo")
    
    args = parser.parse_args()
    
    if args.continuous:
        # Modo continuo directo
        print_banner()
        print(f"Cargando modelo: {args.model}")
        model = PPO.load(args.model)
        continuous_mode(model)
    
    elif args.episodes > 1:
        # Múltiples episodios no interactivos
        print_banner()
        print(f"Cargando modelo: {args.model}")
        model = PPO.load(args.model)
        compare_episodes(model, n_episodes=args.episodes)
    
    else:
        # Modo interactivo (default)
        interactive_demo(args.model)


if __name__ == "__main__":
    main()
