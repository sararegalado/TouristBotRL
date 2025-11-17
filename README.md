# 🌆 TouristBot - Navigation Agent with RL

Agent that learns to navigate a 2D city to reach specific places (restaurants, museums, shops, cinemas, parks) using Reinforcement Learning.

## 📁 Project Structure

```
project/
├── touristbot_env.py       # Environment (20x20 grid, partial view 5x5)
├── train_ppo_basic.py      # Basic PPO training
├── train_advanced.py       # Curriculum + Comparison + Tuning
├── analyze_results.py      # Analysis and visualization
├── demo.py                 # Interactive demo
├── utils.py                # Utilities (wrappers, callbacks)
├── show_places.py          # Show all places on map
```

## 🎮 Quick Start

### 1. Train basic model

```bash
python train_ppo_basic.py --train
```

### 2. Test trained model

```bash
python demo.py --model models/ppo_basic/ppo_touristbot_final.zip
```

### 3. View progress in TensorBoard

```bash
tensorboard --logdir ./tensorboard/ppo_basic/
```

## 🎓 Advanced Techniques

### Curriculum Learning
```bash
python train_advanced.py --mode curriculum
```
Progressive training by reducing available time (150→100→75 steps).

### Compare Algorithms (PPO vs SAC vs DQN)
```bash
python train_advanced.py --mode compare --timesteps 100000
```

### Hyperparameter Tuning (Optuna)
```bash
python train_advanced.py --mode tune --trials 50
```

## 📊 Results Analysis

```bash
# Learning curves
python analyze_results.py --plot-learning logs/ppo_basic/

# Visualize policy
python analyze_results.py --visualize-policy models/ppo_basic/best_model.zip

# Full report
python analyze_results.py --full-report models/ppo_basic/best_model.zip logs/ppo_basic/
```

## 🏙️ Visualize City Structure

```bash
python show_places.py
```
This script shows the map of streets, buildings, and all places both in text and visually.

## 📈 Configuration

Edit `CONFIG` in `train_ppo_basic.py`:

```python
CONFIG = {
    "use_partial_obs": True,    # Partial view
    "view_size": 5,              # View size
    "n_envs": 4,                 # Parallel environments
    "total_timesteps": 200000,   # Timesteps
    "learning_rate": 3e-4,       # Learning rate
}
```

## 🏆 Expected Results

- **Success rate**: 70-95%
- **Average steps**: 10-25
- **Training time**: 10-30 min (CPU)

---

**Authors**: Sara Regalado | Zaloa Fernandez | Universidad de Deusto 2025-2026
