import argparse
import glob
import os
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO, A2C, DQN

from touristbot_env import TouristBotEnv

# Load model, auto-detecting algorithm type
def load_model(model_path):
    # Try to infer from path
    path_lower = model_path.lower()
    
    try:
        if "ppo" in path_lower:
            return PPO.load(model_path), "PPO"
        elif "a2c" in path_lower:
            return A2C.load(model_path), "A2C"
        elif "dqn" in path_lower:
            return DQN.load(model_path), "DQN"
        else:
            # Try each algorithm
            for algo_class, algo_name in [(PPO, "PPO"), (A2C, "A2C"), (DQN, "DQN")]:
                try:
                    return algo_class.load(model_path), algo_name
                except:
                    continue
            raise ValueError(f"Could not load model from {model_path}")
    except Exception as e:
        raise ValueError(f"Error loading model {model_path}: {e}")


# Test a single model and return statistics
def test_single_model(model_path, n_episodes=10, verbose=False):
    if verbose:
        print(f"\nTesting: {model_path}")
    
    # Load model
    try:
        model, algo_name = load_model(model_path)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None
    
    # Create environment
    env = TouristBotEnv(
        use_partial_obs=True,
        view_size=5,
        render_mode=None
    )
    
    # Test statistics
    successes = 0
    steps_list = []
    rewards_list = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        if terminated:
            successes += 1
        
        steps_list.append(steps)
        rewards_list.append(episode_reward)
    
    env.close()
    
    # Calculate statistics
    stats = {
        "model_path": model_path,
        "algorithm": algo_name,
        "n_episodes": n_episodes,
        "success_rate": successes / n_episodes,
        "avg_steps": np.mean(steps_list),
        "std_steps": np.std(steps_list),
        "avg_reward": np.mean(rewards_list),
        "std_reward": np.std(rewards_list),
        "min_steps": np.min(steps_list),
        "max_steps": np.max(steps_list),
        "min_reward": np.min(rewards_list),
        "max_reward": np.max(rewards_list),
    }
    
    return stats

# Compare multiple models and generate report
def compare_models(model_paths, n_episodes=10, output_file=None):
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"Episodes per model: {n_episodes}")
    print(f"Total models: {len(model_paths)}\n")
    
    # Test all models
    results = []
    for i, model_path in enumerate(model_paths, 1):
        print(f"[{i}/{len(model_paths)}] Testing {os.path.basename(model_path)}...", end=" ")
        stats = test_single_model(model_path, n_episodes=n_episodes)
        
        if stats:
            results.append(stats)
            print(f"Success rate: {stats['success_rate']*100:.1f}%")
        else:
            print("Failed")
    
    if not results:
        print("\nNo models could be tested successfully.")
        return
    
    # Sort by success rate, then by average steps (fewer is better)
    results.sort(key=lambda x: (-x['success_rate'], x['avg_steps']))
    
    # Generate report
    report_lines = []
    report_lines.append(f"\n{'='*80}")
    report_lines.append("COMPARISON RESULTS")
    report_lines.append(f"{'='*80}")
    report_lines.append(f"Tested: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Episodes per model: {n_episodes}")
    report_lines.append(f"Total models: {len(results)}\n")
    
    # Summary table
    report_lines.append(f"{'Rank':<6} {'Algorithm':<8} {'Success Rate':<14} {'Avg Steps':<14} {'Avg Reward':<14} {'Model'}")
    report_lines.append("-" * 80)
    
    for rank, stats in enumerate(results, 1):
        model_name = os.path.basename(os.path.dirname(stats['model_path']))
        report_lines.append(
            f"{rank:<6} "
            f"{stats['algorithm']:<8} "
            f"{stats['success_rate']*100:>6.1f}% "
            f"({stats['n_episodes']}/{stats['n_episodes']})  "
            f"{stats['avg_steps']:>6.1f} ± {stats['std_steps']:<4.1f}  "
            f"{stats['avg_reward']:>6.1f} ± {stats['std_reward']:<4.1f}  "
            f"{model_name}"
        )
    
    # Detailed statistics
    report_lines.append(f"\n{'='*80}")
    report_lines.append("DETAILED STATISTICS")
    report_lines.append(f"{'='*80}\n")
    
    for rank, stats in enumerate(results, 1):
        report_lines.append(f"Rank {rank}: {os.path.basename(stats['model_path'])}")
        report_lines.append(f"  Path: {stats['model_path']}")
        report_lines.append(f"  Algorithm: {stats['algorithm']}")
        report_lines.append(f"  Success Rate: {stats['success_rate']*100:.1f}% ({int(stats['success_rate']*stats['n_episodes'])}/{stats['n_episodes']})")
        report_lines.append(f"  Steps: {stats['avg_steps']:.1f} ± {stats['std_steps']:.1f} (min: {stats['min_steps']}, max: {stats['max_steps']})")
        report_lines.append(f"  Reward: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f} (min: {stats['min_reward']:.2f}, max: {stats['max_reward']:.2f})")
        report_lines.append("")
    
    # Best model summary
    best = results[0]
    report_lines.append(f"{'='*80}")
    report_lines.append("BEST MODEL")
    report_lines.append(f"{'='*80}")
    report_lines.append(f"Model: {best['model_path']}")
    report_lines.append(f"Algorithm: {best['algorithm']}")
    report_lines.append(f"Success Rate: {best['success_rate']*100:.1f}%")
    report_lines.append(f"Average Steps: {best['avg_steps']:.1f} ± {best['std_steps']:.1f}")
    report_lines.append(f"Average Reward: {best['avg_reward']:.2f} ± {best['std_reward']:.2f}")
    report_lines.append(f"{'='*80}")
    
    # Print report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple trained TouristBot models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "models",
        nargs="+",
        help="Model paths (can use wildcards like runs/*/models/final_model.zip)"
    )
    
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=10,
        help="Number of episodes to test each model (default: 10)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save comparison report to file"
    )
    
    args = parser.parse_args()
    
    # Expand wildcards
    model_paths = []
    for pattern in args.models:
        matches = glob.glob(pattern)
        if matches:
            model_paths.extend(matches)
        else:
            # If no matches, try as literal path
            if os.path.exists(pattern):
                model_paths.append(pattern)
            else:
                print(f"No matches found for pattern: {pattern}")
    
    if not model_paths:
        print("Error: No model files found.")
        print(f"Searched for: {args.models}")
        return
    
    # Remove duplicates and sort
    model_paths = sorted(set(model_paths))
    
    # Run comparison
    compare_models(model_paths, n_episodes=args.episodes, output_file=args.output)


if __name__ == "__main__":
    main()
