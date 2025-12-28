#!/usr/bin/env python3
"""
Quick demo script to visualize a trained TouristBot model.

Usage:
    # Visualize best model from a training run
    python demo.py runs/ppo_basic_*/models/best_model.zip
    
    # Visualize with custom number of episodes
    python demo.py runs/ppo_basic_*/models/final_model.zip --episodes 3
    
    # Run demo with slower speed
    python demo.py models/my_model.zip --delay 0.5
"""

import argparse
import glob
import time
import sys
from stable_baselines3 import PPO, A2C, DQN
from touristbot_env import TouristBotEnv


def load_model(model_path):
    """Load model, auto-detecting algorithm type."""
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
        print(f"❌ Error loading model: {e}")
        sys.exit(1)


def demo(model_path, n_episodes=5, delay=0.1, view_size=5):
    """Run demo with visualization."""
    print(f"\n{'='*70}")
    print("TOURISTBOT DEMO")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Delay: {delay}s per step")
    print()
    
    # Load model
    print("Loading model...", end=" ")
    model, algo_name = load_model(model_path)
    print(f"✓ ({algo_name})")
    
    # Create environment with rendering
    print("Creating environment...", end=" ")
    env = TouristBotEnv(
        use_partial_obs=True,
        view_size=view_size,
        render_mode="human"
    )
    print("✓")
    
    print(f"\n{'='*70}")
    print("Starting demo... (Press Ctrl+C to stop)")
    print(f"{'='*70}\n")
    
    # Run episodes
    total_successes = 0
    total_steps = []
    total_rewards = []
    
    try:
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            print(f"\n{'─'*70}")
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"{'─'*70}")
            print(f"🎯 Goal: Find a {env.goal_type.upper()}")
            print(f"📍 Location: {env.goal_pos}")
            print()
            
            while not done:
                # Get action from model
                action, _states = model.predict(obs, deterministic=True)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated
                
                # Render
                env.render()
                
                # Delay for visualization
                if delay > 0:
                    time.sleep(delay)
            
            # Episode results
            if terminated:
                total_successes += 1
                print(f"\n✅ SUCCESS! Found {env.goal_type} in {steps} steps")
            else:
                print(f"\n⏱️  TIMEOUT after {steps} steps")
            
            print(f"💰 Reward: {episode_reward:.2f}")
            
            total_steps.append(steps)
            total_rewards.append(episode_reward)
            
            # Brief pause between episodes
            if episode < n_episodes - 1:
                print(f"\nNext episode starting in 2 seconds...")
                time.sleep(2)
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Demo interrupted by user")
    
    finally:
        env.close()
    
    # Summary
    if total_steps:
        import numpy as np
        print(f"\n{'='*70}")
        print("DEMO SUMMARY")
        print(f"{'='*70}")
        print(f"Episodes completed: {len(total_steps)}/{n_episodes}")
        print(f"Success rate: {total_successes}/{len(total_steps)} ({total_successes/len(total_steps)*100:.1f}%)")
        print(f"Average steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
        print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trained TouristBot model in action",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo the best model
  python demo.py runs/ppo_basic_*/models/best_model.zip
  
  # Demo with 10 episodes
  python demo.py runs/ppo_basic_*/models/final_model.zip --episodes 10
  
  # Slower demo for better observation
  python demo.py models/my_model.zip --delay 0.5
  
  # Faster demo
  python demo.py models/my_model.zip --delay 0.05
        """
    )
    
    parser.add_argument(
        "model",
        help="Path to trained model (.zip file, can use wildcards)"
    )
    
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)"
    )
    
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.1,
        help="Delay between steps in seconds (default: 0.1)"
    )
    
    parser.add_argument(
        "--view-size", "-v",
        type=int,
        default=5,
        help="Agent's view size (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Expand wildcards
    matches = glob.glob(args.model)
    if matches:
        model_path = matches[0]
        if len(matches) > 1:
            print(f"⚠️  Multiple models found, using: {model_path}")
    elif args.model.endswith('.zip'):
        model_path = args.model
    else:
        print(f"❌ No model found: {args.model}")
        sys.exit(1)
    
    demo(model_path, n_episodes=args.episodes, delay=args.delay, view_size=args.view_size)


if __name__ == "__main__":
    main()
