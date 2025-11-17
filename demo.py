import argparse
import time
import numpy as np
from stable_baselines3 import PPO
from touristbot_env import TouristBotEnv, PLACE_TYPES
import cv2


def print_banner():
    """Print startup banner"""
    print("\n" + "="*70)
    print(" " * 20 + "TOURISTBOT DEMO ")
    print("="*70 + "\n")


def print_menu():
    """Show options menu"""
    print("\nMENU:")
    print("  1. Episode with restaurant goal")
    print("  2. Episode with museum goal")
    print("  3. Random episode")
    print("  4. Compare 5 episodes")
    print("  5. Continuous mode (until interrupted)")
    print("  q. Exit")
    print()


# Function to run complete episode
def run_episode(model, goal_type=None, render=True, verbose=True, max_steps=None):
    # Use only restaurant and museum for old models
    old_places = ["restaurant", "museum"]
    if goal_type is None:
        goal_type = np.random.choice(old_places)
    
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
        print(f"NEW EPISODE")
        print(f"{'='*70}")
        print(f"Goal: {env.goal_type.upper()} at position {env.goal_pos}")
        print(f"Start: {env.agent_pos}")
        print(f"Initial distance: {env._manhattan_distance(env.agent_pos, env.goal_pos)} cells")
        print(f"{'='*70}\n")
    
    trajectory = [env.agent_pos.copy()]
    actions_taken = []
    rewards_received = []
    
    action_names = ["↑ Up", "↓ Down", "← Left", "→ Right"]
    
    # Step limit to avoid infinite loops
    if max_steps is None:
        max_steps = env.max_steps + 50
    
    steps_taken = 0
    stuck_counter = 0
    last_position = env.agent_pos.copy()
    
    while not done and steps_taken < max_steps:
        # Model prediction
        try:
            action, _states = model.predict(obs, deterministic=True)
        except Exception as e:
            print(f"\nPrediction error: {e}")
            print("Using random action as fallback")
            action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        trajectory.append(env.agent_pos.copy())
        actions_taken.append(action)
        rewards_received.append(reward)
        
        done = terminated or truncated
        steps_taken += 1
        
        # Detect if stuck at same position
        if env.agent_pos == last_position:
            stuck_counter += 1
            if stuck_counter > 20:
                if verbose:
                    print(f"\nAgent stuck at {env.agent_pos} for {stuck_counter} steps")
                # Force termination
                done = True
                truncated = True
        else:
            stuck_counter = 0
            last_position = env.agent_pos.copy()
        
        # Show info every 5 steps
        if verbose and env.steps % 5 == 0:
            print(f"Paso {env.steps:3d}: {action_names[action]:12s} | "
                  f"Pos {env.agent_pos} | "
                  f"Dist {info['distance_to_goal']:2d} | "
                  f"Reward {reward:+6.2f}")
        
        if render:
            env.render()
            time.sleep(0.1)
    
    # Result
    if verbose:
        print(f"\n{'='*70}")
        if terminated and stuck_counter <= 20:
            print("GOAL REACHED!")
        elif stuck_counter > 20:
            print("Agent stuck - Episode terminated")
        elif steps_taken >= max_steps:
            print("Safety limit reached (possible loop)")
        else:
            print("Timeout")
        print(f"{'='*70}")
        print(f"Total steps: {env.steps}")
        print(f"Total reward: {env.total_reward:.2f}")
        print(f"Cells explored: {len(env.visited_cells)}")
        if stuck_counter > 0:
            print(f"Steps without movement: {stuck_counter}")
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
    """Run multiple episodes and compare results"""
    print(f"\n{'='*70}")
    print(f"COMPARING {n_episodes} EPISODES")
    print(f"{'='*70}\n")
    
    results = []
    
    for i in range(n_episodes):
        print(f"\n--- Episode {i+1}/{n_episodes} ---")
        stats = run_episode(model, goal_type=None, render=False, verbose=False)
        results.append(stats)
        
        print(f" {stats['steps']:2d} steps | "
              f"Reward: {stats['total_reward']:6.2f} | "
              f"Explored: {stats['cells_explored']:2d} cells")
    
    # Aggregate statistics
    success_count = sum(1 for r in results if r["success"])
    avg_steps = np.mean([r["steps"] for r in results])
    avg_reward = np.mean([r["total_reward"] for r in results])
    
    print(f"\n{'='*70}")
    print(f"AGGREGATE STATISTICS")
    print(f"{'='*70}")
    print(f"Success rate: {success_count}/{n_episodes} ({success_count/n_episodes*100:.1f}%)")
    print(f"Average steps: {avg_steps:.1f} ± {np.std([r['steps'] for r in results]):.1f}")
    print(f"Average reward: {avg_reward:.2f} ± {np.std([r['total_reward'] for r in results]):.2f}")
    print(f"{'='*70}\n")


def continuous_mode(model):
    """Continuous mode: run episodes until interrupted"""
    print(f"\n{'='*70}")
    print("CONTINUOUS MODE")
    print("{'='*70}")
    print("Press Ctrl+C to stop\n")
    
    episode_count = 0
    success_count = 0
    
    try:
        while True:
            episode_count += 1
            print(f"\n{'─'*70}")
            print(f"Episode #{episode_count}")
            print(f"{'─'*70}")
            
            stats = run_episode(model, goal_type=None, render=True, verbose=False)
            
            if stats["success"]:
                success_count += 1
            
            result = "Success" if stats["success"] else "Timeout"
            print(f"{result} | {stats['steps']} steps | Reward: {stats['total_reward']:.2f}")
            print(f"Cumulative success rate: {success_count}/{episode_count} ({success_count/episode_count*100:.1f}%)")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print("Continuous mode stopped")
        print(f"{'='*70}")
        print(f"Episodes run: {episode_count}")
        print(f"Final success rate: {success_count}/{episode_count} ({success_count/episode_count*100:.1f}%)")
        print(f"{'='*70}\n")


def interactive_demo(model_path):
    """Interactive demo with menu"""
    print_banner()
    
    # Load model
    print("📦 Loading model...")
    try:
        model = PPO.load(model_path)
        print(f"Model loaded: {model_path}\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Menu loop
    while True:
        print_menu()
        choice = input("Select an option: ").strip().lower()
        
        if choice == "1":
            run_episode(model, goal_type="restaurant", render=True, verbose=True, max_steps=250)
        
        elif choice == "2":
            run_episode(model, goal_type="museum", render=True, verbose=True, max_steps=250)
        
        elif choice == "3":
            run_episode(model, goal_type=None, render=True, verbose=True, max_steps=250)
        
        elif choice == "4":
            compare_episodes(model, n_episodes=5)
        
        elif choice == "5":
            continuous_mode(model)
        
        elif choice == "q":
            print("\nGoodbye!\n")
            break
        
        else:
            print("\nInvalid option. Try again.")


def main():
    parser = argparse.ArgumentParser(description="Demo interactiva de TouristBot")
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo (.zip)")
    parser.add_argument("--continuous", action="store_true", help="Modo continuo directo")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes (if not interactive)")
    parser.add_argument("--goal", type=str, choices=PLACE_TYPES, help="Goal type")
    
    args = parser.parse_args()
    
    if args.continuous:
        print_banner()
        print(f"Loading model: {args.model}")
        model = PPO.load(args.model)
        continuous_mode(model)
    
    elif args.episodes > 1:
        print_banner()
        print(f"Loading model: {args.model}")
        model = PPO.load(args.model)
        compare_episodes(model, n_episodes=args.episodes)
    
    else:
        interactive_demo(args.model)


if __name__ == "__main__":
    main()
