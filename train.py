"""
Unified training script for TouristBot with support for:
- Multiple RL algorithms (PPO, A2C, DQN)
- Different hyperparameter configurations
- Curriculum learning strategies
- Easy command-line interface

Usage:
    # Train with specific algorithm
    python train.py --algorithm ppo_basic
    
    # Train with curriculum learning
    python train.py --algorithm ppo_basic --curriculum easy_to_hard
    
    # List all available options
    python train.py --list
    
    # Test a trained model
    python train.py --test models/ppo_basic/best_model.zip --episodes 10
"""

import os
import argparse
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
import gymnasium as gym

from touristbot_env import TouristBotEnv
from training_configs import (
    get_config, get_curriculum, get_env_config,
    list_available_configs, ALGORITHMS, CURRICULUM_STRATEGIES
)


class CurriculumCallback:
    """Callback to manage curriculum learning stages."""
    
    def __init__(self, stages, env_creator, model_creator, base_log_dir):
        self.stages = stages
        self.env_creator = env_creator
        self.model_creator = model_creator
        self.base_log_dir = base_log_dir
        self.current_stage = 0
        
    def get_stage_info(self):
        """Get current stage information."""
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return None


def make_env_creator(env_config, rank=0, seed=0):
    """Create environment factory function."""
    def _init():
        env = TouristBotEnv(
            use_partial_obs=env_config.get("use_partial_obs", True),
            view_size=env_config.get("view_size", 5),
            render_mode=None
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_algorithm(algo_name, config, env, tensorboard_log):
    """Create RL algorithm instance based on name and config."""
    algo_type = config["algorithm"]
    
    if algo_type == "PPO":
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            policy_kwargs={"net_arch": config["net_arch"]},
            verbose=1,
            tensorboard_log=tensorboard_log,
            device="auto"
        )
    
    elif algo_type == "A2C":
        return A2C(
            "MlpPolicy",
            env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            rms_prop_eps=config.get("rms_prop_eps", 1e-5),
            use_rms_prop=config.get("use_rms_prop", True),
            policy_kwargs={"net_arch": config["net_arch"]},
            verbose=1,
            tensorboard_log=tensorboard_log,
            device="auto"
        )
    
    elif algo_type == "DQN":
        return DQN(
            "MlpPolicy",
            env,
            learning_rate=config["learning_rate"],
            buffer_size=config["buffer_size"],
            learning_starts=config["learning_starts"],
            batch_size=config["batch_size"],
            tau=config["tau"],
            gamma=config["gamma"],
            train_freq=config["train_freq"],
            gradient_steps=config["gradient_steps"],
            target_update_interval=config["target_update_interval"],
            exploration_fraction=config["exploration_fraction"],
            exploration_initial_eps=config["exploration_initial_eps"],
            exploration_final_eps=config["exploration_final_eps"],
            policy_kwargs={"net_arch": config["net_arch"]},
            verbose=1,
            tensorboard_log=tensorboard_log,
            device="auto"
        )
    
    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")


def train(algorithm_name, curriculum_name="none", custom_name=None):
    """
    Main training function with support for curriculum learning.
    
    Args:
        algorithm_name: Name of algorithm config (e.g., 'ppo_basic')
        curriculum_name: Name of curriculum strategy (e.g., 'easy_to_hard')
        custom_name: Custom name for this training run
    """
    # Get configurations
    config = get_config(algorithm_name)
    curriculum = get_curriculum(curriculum_name)
    
    # Setup directories
    run_name = custom_name or f"{algorithm_name}_{curriculum_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir = f"./runs/{run_name}"
    log_dir = f"{base_dir}/logs"
    model_dir = f"{base_dir}/models"
    tensorboard_log = f"{base_dir}/tensorboard"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    
    # Print training info
    print("\n" + "="*70)
    print(f"TOURISTBOT TRAINING - {run_name.upper()}")
    print("="*70)
    print(f"\nAlgorithm: {algorithm_name} ({config['algorithm']})")
    print(f"Curriculum: {curriculum['name']} - {curriculum['description']}")
    print(f"Total Stages: {len(curriculum['stages'])}")
    print(f"Total Timesteps: {sum(s['timesteps'] for s in curriculum['stages']):,}")
    print(f"\nOutput Directory: {base_dir}")
    print(f"TensorBoard: tensorboard --logdir {tensorboard_log}")
    
    # Save configuration
    with open(f"{base_dir}/config.txt", "w") as f:
        f.write(f"Algorithm: {algorithm_name}\n")
        f.write(f"Curriculum: {curriculum_name}\n")
        f.write(f"Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nCurriculum Stages:\n")
        for i, stage in enumerate(curriculum['stages'], 1):
            f.write(f"  Stage {i}: {stage['description']}\n")
            f.write(f"    Timesteps: {stage['timesteps']}\n")
            f.write(f"    Env Config: {stage['env_config']}\n")
    
    # Training with curriculum
    model = None
    total_timesteps_trained = 0
    
    for stage_idx, stage in enumerate(curriculum['stages']):
        print(f"\n{'='*70}")
        print(f"CURRICULUM STAGE {stage_idx + 1}/{len(curriculum['stages'])}")
        print(f"{'='*70}")
        print(f"Description: {stage['description']}")
        print(f"Timesteps: {stage['timesteps']:,}")
        print(f"Environment: {stage['env_config']}")
        print()
        
        # Create environments for this stage
        env_config = stage['env_config']
        
        if config['algorithm'] == 'DQN':
            # DQN doesn't use vectorized environments
            env = make_env_creator(env_config, rank=0, seed=42)()
            env = DummyVecEnv([lambda: env])
            eval_env = make_env_creator(env_config, rank=100, seed=123)()
            eval_env = DummyVecEnv([lambda: eval_env])
        else:
            # PPO and A2C use vectorized environments
            n_envs = config.get("n_envs", 8)
            env = make_vec_env(
                make_env_creator(env_config, rank=0),
                n_envs=n_envs,
                seed=42
            )
            eval_env = make_vec_env(
                make_env_creator(env_config, rank=100),
                n_envs=1,
                seed=123
            )
        
        # Create or update model
        if model is None:
            # First stage: create new model
            print("Creating new model...")
            model = create_algorithm(algorithm_name, config, env, tensorboard_log)
        else:
            # Subsequent stages: update environment
            print("Updating model environment for new stage...")
            model.set_env(env)
        
        # Setup callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{model_dir}/stage_{stage_idx + 1}",
            log_path=log_dir,
            eval_freq=max(5000 // config.get("n_envs", 1), 1),
            deterministic=True,
            render=False,
            verbose=1
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(25000 // config.get("n_envs", 1), 1000),
            save_path=f"{model_dir}/stage_{stage_idx + 1}",
            name_prefix=f"checkpoint_stage{stage_idx + 1}"
        )
        
        # Train this stage
        print(f"\n🎓 Training Stage {stage_idx + 1}...")
        try:
            model.learn(
                total_timesteps=stage['timesteps'],
                callback=[eval_callback, checkpoint_callback],
                reset_num_timesteps=(stage_idx == 0),  # Reset only for first stage
                progress_bar=True
            )
            total_timesteps_trained += stage['timesteps']
            
            # Save stage model
            stage_model_path = f"{model_dir}/stage_{stage_idx + 1}_final"
            model.save(stage_model_path)
            print(f"✓ Stage {stage_idx + 1} completed. Model saved: {stage_model_path}")
            
        except KeyboardInterrupt:
            print(f"\n⚠ Training interrupted at stage {stage_idx + 1}")
            interrupted_path = f"{model_dir}/interrupted_stage_{stage_idx + 1}"
            model.save(interrupted_path)
            print(f"Model saved: {interrupted_path}")
            break
        
        finally:
            env.close()
            eval_env.close()
    
    # Save final model
    if model is not None:
        final_path = f"{model_dir}/final_model"
        model.save(final_path)
        print(f"\n{'='*70}")
        print("TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Total Timesteps: {total_timesteps_trained:,}")
        print(f"Final Model: {final_path}.zip")
        print(f"Best Models: {model_dir}/stage_*/best_model.zip")
    
    return model


def test_model(model_path, n_episodes=5, render=True, env_config=None):
    """
    Test a trained model.
    
    Args:
        model_path: Path to saved model (.zip file)
        n_episodes: Number of episodes to run
        render: Whether to render visualization
        env_config: Environment configuration (default: standard config)
    """
    print(f"\n{'='*70}")
    print("TESTING TRAINED MODEL")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}\n")
    
    # Load model
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path)
    elif "dqn" in model_path.lower():
        model = DQN.load(model_path)
    else:
        # Try to infer from file
        print("⚠ Algorithm not detected from filename, trying PPO...")
        try:
            model = PPO.load(model_path)
        except:
            print("Failed with PPO, trying A2C...")
            try:
                model = A2C.load(model_path)
            except:
                print("Failed with A2C, trying DQN...")
                model = DQN.load(model_path)
    
    # Create environment
    if env_config is None:
        env_config = {"use_partial_obs": True, "view_size": 5}
    
    env = TouristBotEnv(
        use_partial_obs=env_config.get("use_partial_obs", True),
        view_size=env_config.get("view_size", 5),
        render_mode="human" if render else None
    )
    
    # Test statistics
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
        
        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")
        print(f"Goal: {env.goal_type} at {env.goal_pos}")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            if render:
                env.render()
        
        # Record statistics
        if terminated:
            stats["success"] += 1
            result = "✓ Success"
        else:
            stats["timeout"] += 1
            result = "✗ Timeout"
        
        stats["steps"].append(steps)
        stats["rewards"].append(episode_reward)
        
        print(f"Result: {result}")
        print(f"Steps: {steps}")
        print(f"Reward: {episode_reward:.2f}")
    
    env.close()
    
    # Print summary
    print(f"\n{'='*70}")
    print("TEST STATISTICS")
    print(f"{'='*70}")
    print(f"Success Rate: {stats['success']}/{n_episodes} ({stats['success']/n_episodes*100:.1f}%)")
    print(f"Average Steps: {np.mean(stats['steps']):.1f} ± {np.std(stats['steps']):.1f}")
    print(f"Average Reward: {np.mean(stats['rewards']):.2f} ± {np.std(stats['rewards']):.2f}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Train TouristBot with multiple algorithms and curriculum learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available options
  python train.py --list
  
  # Train with PPO basic configuration
  python train.py --algorithm ppo_basic
  
  # Train with curriculum learning
  python train.py --algorithm ppo_basic --curriculum easy_to_hard
  
  # Train with A2C
  python train.py --algorithm a2c_basic
  
  # Train with DQN
  python train.py --algorithm dqn_basic
  
  # Test a trained model
  python train.py --test runs/ppo_basic_none_*/models/final_model.zip
  
  # Test with more episodes and no rendering
  python train.py --test models/best_model.zip --episodes 20 --no-render
        """
    )
    
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        choices=list(ALGORITHMS.keys()),
        help="Algorithm configuration to use"
    )
    
    parser.add_argument(
        "--curriculum", "-c",
        type=str,
        default="none",
        choices=list(CURRICULUM_STRATEGIES.keys()),
        help="Curriculum learning strategy (default: none)"
    )
    
    parser.add_argument(
        "--name", "-n",
        type=str,
        help="Custom name for this training run"
    )
    
    parser.add_argument(
        "--test", "-t",
        type=str,
        help="Path to model to test (.zip file)"
    )
    
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=5,
        help="Number of episodes for testing (default: 5)"
    )
    
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering during testing"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available algorithms and curriculum strategies"
    )
    
    args = parser.parse_args()
    
    # List available options
    if args.list:
        list_available_configs()
        return
    
    # Test mode
    if args.test:
        test_model(
            args.test,
            n_episodes=args.episodes,
            render=not args.no_render
        )
        return
    
    # Train mode
    if args.algorithm:
        train(
            algorithm_name=args.algorithm,
            curriculum_name=args.curriculum,
            custom_name=args.name
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
