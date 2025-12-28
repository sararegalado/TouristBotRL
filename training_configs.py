"""
Training configurations for different algorithms and strategies.
Contains hyperparameters for PPO, A2C, DQN, and curriculum learning strategies.
"""

# ==============================================================================
# PPO CONFIGURATIONS
# ==============================================================================

PPO_BASIC = {
    "algorithm": "PPO",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
    "total_timesteps": 500000,
    "n_envs": 8,
}

PPO_AGGRESSIVE = {
    "algorithm": "PPO",
    "learning_rate": 1e-3,  # Higher learning rate
    "n_steps": 1024,  # Fewer steps per update
    "batch_size": 128,
    "n_epochs": 15,  # More epochs
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.3,  # Larger clip range
    "ent_coef": 0.02,  # More exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
    "total_timesteps": 500000,
    "n_envs": 8,
}

PPO_CONSERVATIVE = {
    "algorithm": "PPO",
    "learning_rate": 1e-4,  # Lower learning rate
    "n_steps": 4096,  # More steps per update
    "batch_size": 512,  # Larger batches
    "n_epochs": 5,  # Fewer epochs
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,  # Smaller clip range
    "ent_coef": 0.005,  # Less exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "net_arch": [dict(pi=[512, 512], vf=[512, 512])],  # Larger network
    "total_timesteps": 500000,
    "n_envs": 4,  # Fewer parallel envs
}

# ==============================================================================
# A2C CONFIGURATIONS
# ==============================================================================

A2C_BASIC = {
    "algorithm": "A2C",
    "learning_rate": 7e-4,
    "n_steps": 5,
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "rms_prop_eps": 1e-5,
    "use_rms_prop": True,
    "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
    "total_timesteps": 500000,
    "n_envs": 16,  # A2C benefits from more parallel envs
}

A2C_FAST = {
    "algorithm": "A2C",
    "learning_rate": 1e-3,
    "n_steps": 3,
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "ent_coef": 0.02,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "rms_prop_eps": 1e-5,
    "use_rms_prop": True,
    "net_arch": [dict(pi=[128, 128], vf=[128, 128])],
    "total_timesteps": 500000,
    "n_envs": 16,
}

# ==============================================================================
# DQN CONFIGURATIONS
# ==============================================================================

DQN_BASIC = {
    "algorithm": "DQN",
    "learning_rate": 1e-4,
    "buffer_size": 100000,
    "learning_starts": 10000,
    "batch_size": 128,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 1000,
    "exploration_fraction": 0.3,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "net_arch": [256, 256],
    "total_timesteps": 500000,
    "n_envs": 1,  # DQN doesn't use parallel envs
}

DQN_DOUBLE = {
    "algorithm": "DQN",
    "learning_rate": 1e-4,
    "buffer_size": 150000,
    "learning_starts": 10000,
    "batch_size": 256,
    "tau": 0.01,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 2,  # Double gradient steps
    "target_update_interval": 2000,
    "exploration_fraction": 0.2,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.02,
    "net_arch": [512, 512],  # Larger network
    "total_timesteps": 500000,
    "n_envs": 1,
}

# ==============================================================================
# CURRICULUM LEARNING STRATEGIES
# ==============================================================================

# ==============================================================================
# NEW CURRICULUM LEARNING STRATEGIES (Beyond View Size)
# ==============================================================================

CURRICULUM_TIME_PRESSURE = {
    "name": "time_pressure",
    "description": "Gradually reduce max steps (time limit) to increase difficulty",
    "stages": [
        {
            "timesteps": 150000,
            "env_config": {
                "use_partial_obs": True,
                "view_size": 5,
                "max_steps": 300,  # Generous time limit
            },
            "description": "Stage 1: 300 steps - Learn without time pressure"
        },
        {
            "timesteps": 200000,
            "env_config": {
                "use_partial_obs": True,
                "view_size": 5,
                "max_steps": 200,  # Standard time
            },
            "description": "Stage 2: 200 steps - Moderate time pressure"
        },
        {
            "timesteps": 150000,
            "env_config": {
                "use_partial_obs": True,
                "view_size": 5,
                "max_steps": 120,  # Tight time limit
            },
            "description": "Stage 3: 120 steps - High time pressure"
        },
    ]
}

CURRICULUM_SPARSE_REWARDS = {
    "name": "sparse_rewards",
    "description": "Gradually reduce auxiliary rewards (shaping, exploration) for harder learning",
    "stages": [
        {
            "timesteps": 150000,
            "env_config": {
                "use_partial_obs": True,
                "view_size": 5,
                "reward_shaping_weight": 10.0,  # Full shaping
                "exploration_bonus": 0.5,  # Full exploration bonus
                "step_penalty": -0.1,  # Moderate penalty
            },
            "description": "Stage 1: Dense rewards - Strong guidance"
        },
        {
            "timesteps": 200000,
            "env_config": {
                "use_partial_obs": True,
                "view_size": 5,
                "reward_shaping_weight": 5.0,  # Reduced shaping
                "exploration_bonus": 0.2,  # Reduced exploration
                "step_penalty": -0.15,  # Higher penalty
            },
            "description": "Stage 2: Medium rewards - Less guidance"
        },
        {
            "timesteps": 150000,
            "env_config": {
                "use_partial_obs": True,
                "view_size": 5,
                "reward_shaping_weight": 1.0,  # Minimal shaping
                "exploration_bonus": 0.0,  # No exploration bonus
                "step_penalty": -0.2,  # High penalty
            },
            "description": "Stage 3: Sparse rewards - Minimal guidance"
        },
    ]
}


# ==============================================================================
# AVAILABLE CONFIGURATIONS REGISTRY
# ==============================================================================

ALGORITHMS = {
    "ppo_basic": PPO_BASIC,
    "ppo_aggressive": PPO_AGGRESSIVE,
    "ppo_conservative": PPO_CONSERVATIVE,
    "a2c_basic": A2C_BASIC,
    "a2c_fast": A2C_FAST,
    "dqn_basic": DQN_BASIC,
    "dqn_double": DQN_DOUBLE,
}

CURRICULUM_STRATEGIES = {
    "time_pressure": CURRICULUM_TIME_PRESSURE,
    "sparse_rewards": CURRICULUM_SPARSE_REWARDS,
}

# ==============================================================================
# ENVIRONMENT CONFIGURATIONS
# ==============================================================================

ENV_CONFIGS = {
    "default": {
        "use_partial_obs": True,
        "view_size": 5,
    },
    "easy": {
        "use_partial_obs": True,
        "view_size": 7,
    },
    "hard": {
        "use_partial_obs": True,
        "view_size": 3,
    },
    "full_obs": {
        "use_partial_obs": False,
        "view_size": 20,
    }
}


def get_config(algorithm_name):
    """Get configuration by name."""
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {list(ALGORITHMS.keys())}")
    return ALGORITHMS[algorithm_name].copy()


def get_curriculum(curriculum_name):
    """Get curriculum strategy by name."""
    if curriculum_name not in CURRICULUM_STRATEGIES:
        raise ValueError(f"Unknown curriculum: {curriculum_name}. Available: {list(CURRICULUM_STRATEGIES.keys())}")
    return CURRICULUM_STRATEGIES[curriculum_name].copy()


def get_env_config(env_name):
    """Get environment configuration by name."""
    if env_name not in ENV_CONFIGS:
        raise ValueError(f"Unknown env config: {env_name}. Available: {list(ENV_CONFIGS.keys())}")
    return ENV_CONFIGS[env_name].copy()


def list_available_configs():
    """Print all available configurations."""
    print("\n" + "="*70)
    print("AVAILABLE ALGORITHMS")
    print("="*70)
    for name, config in ALGORITHMS.items():
        print(f"\n{name}:")
        print(f"  Algorithm: {config['algorithm']}")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  Total Timesteps: {config['total_timesteps']:,}")
        if config['algorithm'] == 'PPO':
            print(f"  Batch Size: {config['batch_size']}")
            print(f"  N Steps: {config['n_steps']}")
        elif config['algorithm'] == 'DQN':
            print(f"  Buffer Size: {config['buffer_size']}")
            print(f"  Batch Size: {config['batch_size']}")
    
    print("\n" + "="*70)
    print("CURRICULUM STRATEGIES")
    print("="*70)
    for name, curriculum in CURRICULUM_STRATEGIES.items():
        print(f"\n{name}:")
        print(f"  Description: {curriculum['description']}")
        print(f"  Stages: {len(curriculum['stages'])}")
        for i, stage in enumerate(curriculum['stages'], 1):
            print(f"    Stage {i}: {stage['timesteps']:,} timesteps - {stage['description']}")
    
    print("\n" + "="*70)
    print("ENVIRONMENT CONFIGS")
    print("="*70)
    for name, config in ENV_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Partial Obs: {config['use_partial_obs']}")
        print(f"  View Size: {config['view_size']}x{config['view_size']}")


if __name__ == "__main__":
    list_available_configs()
