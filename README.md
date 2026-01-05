# 🌆 TouristBot - Navigation with RL and NLP

Agent that navigates in a 2D city through specific places (restaurants, museums, shops, cinemas) using Reinforcement Learning and NLP.

## 📋 Table of Contents
- [Installation](#-installation)
- [Training System](#-training-system)
- [Available Algorithms](#-available-algorithms)
- [Curriculum Learning](#-curriculum-learning)
- [Testing & Comparison](#-testing--comparison)
- [Interactive Application](#-interactive-application)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [TensorBoard](#-tensorboard---training-visualization)
- [Additional Information](#-additional-information)
- [Contributing](#-contributing)
- [License](#-license)



## 🛠️ Installation

### Requirements
- Python 3.8+
- numpy < 2.0.0 (important for stable-baselines3 compatibility)

### Setup
```bash
# Clone or navigate to the project directory
cd TouristBot_RL


# Install dependencies
pip install -r requirements.txt
```

**Main Dependencies:**
- `stable-baselines3` - RL algorithms (PPO, A2C, DQN)
- `gymnasium` - RL environment interface
- `transformers` - NLP models for zero-shot classification
- `opencv-python` - Visualization
- `torch` - Deep learning backend
- `tensorboard` - Training monitoring

---

## 🚀 Training System

The project includes a flexible training system supporting multiple algorithms, hyperparameter configurations, and curriculum learning.

### Basic Training Commands

```bash
# List all available options
python train.py --list

# Monitor training with TensorBoard
tensorboard --logdir runs/

# PPO
python train.py --algorithm ppo_basic
python train.py --algorithm ppo_aggressive
python train.py --algorithm ppo_conservative

# A2C
python train.py --algorithm a2c_basic
python train.py --algorithm a2c_fast

# DQN
python train.py --algorithm dqn_basic
python train.py --algorithm dqn_double

# Train with curriculum learning
python train.py --algorithm ppo_basic --curriculum curriculum_time_pressure

python train.py --algorithm ppo_basic --curriculum curriculum_sparse_rewards

# Train with custom name
python train.py --algorithm ppo_basic --name my_experiment

```

### Configuration Files

All configurations are defined in `training_configs.py`:
- **Algorithm configurations**: Hyperparameters for each algorithm variant
- **Curriculum strategies**: Different progressive learning approaches
- **Environment configs**: Various observation settings

---

## 🤖 Available Algorithms

### PPO (Proximal Policy Optimization)

| Config | Speed | Description | Best For |
|--------|-------|-------------|----------|
| `ppo_basic` | ⚡⚡⚡ | Balanced configuration | General purpose (recommended) |
| `ppo_aggressive` | ⚡⚡⚡⚡ | Higher learning rate, more exploration | Quick experiments |
| `ppo_conservative` | ⚡⚡ | Lower learning rate, larger network | Stable, final training |


### A2C (Advantage Actor-Critic)

| Config | Speed | Description | Best For |
|--------|-------|-------------|----------|
| `a2c_basic` | ⚡⚡⚡⚡ | Standard A2C with 16 envs | Fast iterations |
| `a2c_fast` | ⚡⚡⚡⚡⚡ | Smaller network, faster | Rapid prototyping |


### DQN (Deep Q-Network)

| Config | Speed | Description | Best For |
|--------|-------|-------------|----------|
| `dqn_basic` | ⚡⚡ | Standard DQN with replay buffer | Discrete actions |
| `dqn_double` | ⚡ | Larger network and buffer | Best performance |


### Algorithm Comparison Table

| Algorithm | Training Speed | Sample Efficiency | Stability | Parallel Envs |
|-----------|----------------|-------------------|-----------|---------------|
| PPO Basic | Medium | Medium | High | 8 |
| PPO Aggressive | Fast | Medium | Medium | 8 |
| PPO Conservative | Slow | High | Very High | 4 |
| A2C Basic | Fast | Low | Medium | 16 |
| A2C Fast | Very Fast | Low | Medium | 16 |
| DQN Basic | Slow | High | Medium | 1 |
| DQN Double | Slow | High | High | 1 |

---

## 📚 Curriculum Learning

Curriculum learning progressively increases task difficulty to improve learning.

### Available Strategies

#### 1. `curriculum_time_pressure` 
#### 2. `curriculum_sparse_rewards`
---

## 🧪 Testing & Comparison

### Test a Trained Model

```bash
# Test with default settings (5 episodes)
python train.py --test runs/ppo_basic_*/models/final_model.zip

# Test with more episodes
python train.py --test runs/ppo_basic_*/models/best_model.zip --episodes 20

# Test without rendering (faster)
python train.py --test runs/ppo_basic_*/models/final_model.zip --episodes 50 --no-render
```

### Compare Multiple Models

```bash
# Compare all final models
python compare_models.py runs/*/models/final_model.zip

# Compare with more episodes for accuracy
python compare_models.py --episodes 20 runs/*/models/final_model.zip

# Save comparison report
python compare_models.py --output results.txt runs/*/models/final_model.zip
```

**Output includes:**
- Success rates
- Average steps to goal
- Average rewards
- Performance rankings


---

## 🎮 Interactive Application

Run the GUI application with natural language interface:

```bash
# Interactive mode (default) -> uses basic PPO
python touristbot_app.py


# Use specific model
python touristbot_app.py --model runs/ppo_basic_curriculum_time_pressure_20251228_181333/models/final_model.zip
```

### How It Works

1. **Natural Language Processing**
   - Uses zero-shot BERT classification
   - Converts text to goal category (restaurant, museum, shop, cinema, park)
   - Works in Spanish and English

2. **RL Navigation**
   - Trained agent navigates to the goal
   - Partial observation (5x5 view) simulates limited vision
   - Real-time visualization with OpenCV

3. **Interactive Interface**
   - Press 'T' to enter text
   - ESC or EXIT button to quit
   - Real-time agent visualization

---

## 📁 Project Structure

```
TouristBot_RL/
├── 🎯 MAIN SCRIPTS
│   ├── touristbot_app.py          # Interactive GUI application
│   ├── train.py                   # Unified training script (multiple algorithms + curriculum)
│   ├── compare_models.py          # Compare multiple trained models
│
├── ⚙️ CONFIGURATION & ENVIRONMENT
│   ├── training_configs.py        # All algorithm configurations
│   ├── touristbot_env.py          # RL environment (20x20 city, 5x5 partial view)
│
├── 📦 OUTPUTS
│   └── runs/                      # Training runs (organized by algorithm/curriculum)
│       └── <algo>_<curriculum>_<timestamp>/
│           ├── models/
│           │   ├── stage_*/best_model.zip    # Best model from each stage
│           │   └── final_model.zip           # Final trained model
│           ├── logs/                         # Evaluation logs
│           ├── tensorboard/                  # TensorBoard logs
│           └── config.txt                    # Configuration used
│
│
└── 📄 CONFIG
    ├── requirements.txt           # Python dependencies
    └── README.md                  # This file
```

### Key Files

- **train.py**: Main training script with support for 7 algorithm variants and 3 curriculum strategies
- **training_configs.py**: Configuration definitions for all algorithms and curricula
- **compare_models.py**: Batch testing and comparison of multiple models
- **demo.py**: Interactive visualization of trained agents
- **touristbot_app.py**: GUI application with natural language interface
- **touristbot_env.py**: Gymnasium environment implementation


---

## 🎯 Features

### ✨ Training System
- **7 Algorithm Variants**: PPO (3), A2C (2), DQN (2)
- **2 Curriculum Strategies**: Progressive learning approaches
- **Easy CLI**: Simple command-line interface
- **Organized Output**: All runs saved with timestamps
- **TensorBoard Integration**: Real-time training monitoring

### 🧠 Environment
- **20x20 Grid City**: Discrete navigation space
- **Partial Observation**: 5x5 view (configurable)
- **5 Place Types**: Restaurant, Museum, Shop, Cinema, Park
- **Random Goal Selection**: Each episode randomly selects a goal type for better generalization
- **Reward Shaping**: Distance-based + time penalty
- **Episode Limit**: 200 steps (configurable)

### 🗣️ Natural Language Interface
- **Zero-Shot Classification**: No additional training needed
- **Bilingual**: Spanish and English support
- **5 Categories**: Automatic mapping to place types
- **BERT-based**: Uses transformer models

### 📊 Monitoring & Analysis
- **TensorBoard**: Real-time metrics visualization
- **Model Comparison**: Batch testing and ranking
- **Success Rate Tracking**: Episode outcomes
- **Performance Metrics**: Steps, rewards, efficiency

---

## � TensorBoard - Training Visualization

TensorBoard is automatically enabled for all training runs and provides real-time visualization of the learning process.

### Starting TensorBoard
```bash
# This allows comparing multiple experiments
tensorboard --logdir runs/
```

Then open your browser to: **http://localhost:6006**


---

## 📖 Additional Information

### Environment Details

The TouristBot environment simulates a 20x20 grid city with:
- **Places**: 5 types (restaurant, museum, shop, cinema, park)
- **Agent**: Single navigating agent with partial observation
- **Actions**: 4 discrete (up, down, left, right)
- **Observation**: Partial grid view (default 5x5) or full grid
- **Reward**: -0.1 per step, +10.0 for reaching goal
- **Episode**: Ends on goal reached or 150 steps
- **Goal Selection**: Random goal type per episode (unless specified via options)

### Algorithms Overview

- **PPO**: On-policy, good balance of speed and stability
- **A2C**: On-policy synchronous, faster but less sample efficient  
- **DQN**: Off-policy with replay, sample efficient but slower


---

## 🤝 Contributing

This is an educational project for reinforcement learning experimentation. Feel free to:
- Add new algorithm configurations
- Implement new curriculum strategies
- Improve the environment
- Enhance visualization
- Add new features

---

## 📄 License

Educational project for reinforcement learning research and learning.

---

**Authors**: Sara Regalado | Zaloa Fernandez | Universidad de Deusto 2025-2026
