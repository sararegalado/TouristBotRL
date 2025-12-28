# 🌆 TouristBot - Navegación con RL y Lenguaje Natural

Agent that navigates in a 2D city through specific places (restaurants, museums, shops, cinemas) using Reinforcement Learning and NLP.

## 📋 Table of Contents
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Training System](#-training-system)
- [Available Algorithms](#-available-algorithms)
- [Curriculum Learning](#-curriculum-learning)
- [Testing & Comparison](#-testing--comparison)
- [Interactive Application](#-interactive-application)
- [Project Structure](#-project-structure)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)


## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive App
```bash
python touristbot_app.py
```

Press **'T'** to type your destination in natural language:
- "Quiero comer algo" / "I want to eat something"
- "Necesito ir a una tienda" / "I need to go to a shop"  
- "Llévame al museo" / "Take me to the museum"
- "Busca un cine" / "Find a cinema"

### 3. Train Your Own Model
```bash
# See all available options
python train.py --list

# Train with PPO (recommended)
python train.py --algorithm ppo_basic

# Train with curriculum learning
python train.py --algorithm ppo_basic --curriculum easy_to_hard
```

### 4. Test & Compare Models
```bash
# Test a model
python train.py --test runs/ppo_basic_*/models/final_model.zip

# Compare multiple models
python compare_models.py runs/*/models/final_model.zip

# Visualize a model
python demo.py runs/ppo_basic_*/models/best_model.zip
```

---

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

# Train with default PPO
python train.py --algorithm ppo_basic

# Train with a specific algorithm
python train.py --algorithm a2c_basic
python train.py --algorithm dqn_basic

# Train with curriculum learning
python train.py --algorithm ppo_basic --curriculum easy_to_hard

# Train with custom name
python train.py --algorithm ppo_basic --name my_experiment

# Monitor training with TensorBoard
tensorboard --logdir runs/
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

**Example:**
```bash
python train.py --algorithm ppo_basic
```

### A2C (Advantage Actor-Critic)

| Config | Speed | Description | Best For |
|--------|-------|-------------|----------|
| `a2c_basic` | ⚡⚡⚡⚡ | Standard A2C with 16 envs | Fast iterations |
| `a2c_fast` | ⚡⚡⚡⚡⚡ | Smaller network, faster | Rapid prototyping |

**Example:**
```bash
python train.py --algorithm a2c_basic
```

### DQN (Deep Q-Network)

| Config | Speed | Description | Best For |
|--------|-------|-------------|----------|
| `dqn_basic` | ⚡⚡ | Standard DQN with replay buffer | Discrete actions |
| `dqn_double` | ⚡ | Larger network and buffer | Best performance |

**Example:**
```bash
python train.py --algorithm dqn_basic
```

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

#### 1. `easy_to_hard` (Recommended)
Gradually reduces the agent's observation size:
- **Stage 1** (100k steps): 7x7 view - Learning basics
- **Stage 2** (200k steps): 5x5 view - Intermediate difficulty  
- **Stage 3** (200k steps): 3x3 view - Advanced, limited vision

```bash
python train.py --algorithm ppo_basic --curriculum easy_to_hard
```

#### 2. `grid_size`
Staged training with checkpoints (same difficulty):
- **Stage 1** (150k steps): Navigation learning
- **Stage 2** (200k steps): Refinement
- **Stage 3** (150k steps): Mastery

```bash
python train.py --algorithm ppo_basic --curriculum grid_size
```

#### 3. `none` (Default)
Standard single-stage training:
- **Stage 1** (500k steps): Complete training in one go

```bash
python train.py --algorithm ppo_basic --curriculum none
# or simply:
python train.py --algorithm ppo_basic
```

### Curriculum Benefits
- ✅ Faster initial learning
- ✅ Better final performance on hard tasks
- ✅ More stable training
- ✅ Gradual skill development

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

### Visual Demonstration

```bash
# Visualize a trained model
python demo.py runs/ppo_basic_*/models/best_model.zip

# Demo with custom settings
python demo.py runs/ppo_basic_*/models/final_model.zip --episodes 10 --delay 0.2
```

---

## 🎮 Interactive Application

Run the GUI application with natural language interface:

```bash
# Interactive mode (default)
python touristbot_app.py

# Single episode mode
python touristbot_app.py --mode single --text "Quiero ir al museo"

# Use specific model
python touristbot_app.py --model runs/ppo_basic_*/models/best_model.zip

# No visualization (metrics only)
python touristbot_app.py --mode single --text "Busca un restaurante" --no-viz
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
│   └── demo.py                    # Visualize trained models
│
├── ⚙️ CONFIGURATION & ENVIRONMENT
│   ├── training_configs.py        # All algorithm configurations
│   ├── touristbot_env.py          # RL environment (20x20 city, 5x5 partial view)
│   └── utils.py                   # Utilities (wrappers, callbacks)
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

## 📚 Examples

### Example 1: Train and Test PPO

```bash
# Train with PPO
python train.py --algorithm ppo_basic

# Monitor training (in another terminal)
tensorboard --logdir runs/

# Test the model
python train.py --test runs/ppo_basic_*/models/final_model.zip --episodes 10

# Visualize
python demo.py runs/ppo_basic_*/models/best_model.zip

# Export graphs for report
python export_tensorboard_graphs.py runs/ppo_basic_*/tensorboard/
```

### Example 2: Compare Algorithms

```bash
# Train different algorithms
python train.py --algorithm ppo_basic --name exp_ppo
python train.py --algorithm a2c_basic --name exp_a2c
python train.py --algorithm dqn_basic --name exp_dqn

# Monitor all runs simultaneously
tensorboard --logdir runs/

# Compare results
python compare_models.py \
    runs/exp_ppo/models/final_model.zip \
    runs/exp_a2c/models/final_model.zip \
    runs/exp_dqn/models/final_model.zip \
    --episodes 20 \
    --output comparison_results.txt

# Export comparison graphs
python export_tensorboard_graphs.py \
    runs/exp_ppo/tensorboard/ \
    runs/exp_a2c/tensorboard/ \
    runs/exp_dqn/tensorboard/

# View results
cat comparison_results.txt
```

### Example 3: Curriculum Learning

```bash
# Train with curriculum
python train.py --algorithm ppo_basic --curriculum easy_to_hard

# Compare stage models
python compare_models.py \
    runs/ppo_basic_easy_to_hard_*/models/stage_1/best_model.zip \
    runs/ppo_basic_easy_to_hard_*/models/stage_2/best_model.zip \
    runs/ppo_basic_easy_to_hard_*/models/stage_3/best_model.zip

# Test final model
python demo.py runs/ppo_basic_easy_to_hard_*/models/final_model.zip
```


### Example 4: Custom Configuration

Edit `training_configs.py` to add your own configuration:

```python
# Add to training_configs.py
PPO_CUSTOM = {
    "algorithm": "PPO",
    "learning_rate": 5e-4,      # Your custom value
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    # ... other parameters
}

# Register it
ALGORITHMS = {
    # ... existing configs
    "ppo_custom": PPO_CUSTOM,
}
```

Then use it:
```bash
python train.py --algorithm ppo_custom
```



---

## 🎯 Features

### ✨ Training System
- **7 Algorithm Variants**: PPO (3), A2C (2), DQN (2)
- **3 Curriculum Strategies**: Progressive learning approaches
- **Easy CLI**: Simple command-line interface
- **Organized Output**: All runs saved with timestamps
- **TensorBoard Integration**: Real-time training monitoring

### 🧠 Environment
- **20x20 Grid City**: Discrete navigation space
- **Partial Observation**: 5x5 view (configurable)
- **5 Place Types**: Restaurant, Museum, Shop, Cinema, Park
- **Random Goal Selection**: Each episode randomly selects a goal type for better generalization
- **Reward Shaping**: Distance-based + time penalty
- **Episode Limit**: 150 steps (configurable)

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

**Option 1: Monitor a single run**
```bash
# Start training in one terminal
python train.py --algorithm ppo_basic

# In another terminal, start TensorBoard
tensorboard --logdir runs/ppo_basic_none_<timestamp>/tensorboard/
```

**Option 2: Monitor all runs (recommended)**
```bash
# This allows comparing multiple experiments
tensorboard --logdir runs/
```

**Option 3: Monitor specific experiments**
```bash
# Compare specific algorithms
tensorboard --logdir runs/ --path_prefix="ppo|a2c"
```

Then open your browser to: **http://localhost:6006**

### TensorBoard Dashboard Overview

TensorBoard organizes metrics into several tabs:

#### 1. **SCALARS Tab** (Most Important)
Shows metric evolution over time. Key graphs to monitor:

**Reward Metrics:**
- `rollout/ep_rew_mean` - Average episode reward
  - **Should increase** ↗️ over training
  - Indicates agent is learning to reach goals
  - Target: Positive values (>0)

- `rollout/ep_len_mean` - Average episode length
  - **Should decrease** ↘️ over training
  - Shorter = more efficient navigation
  - Target: <50 steps for trained agent

**Training Metrics:**
- `train/learning_rate` - Current learning rate
  - Should remain stable (or decay if scheduled)
  
- `train/entropy_loss` - Exploration level
  - **Should decrease** ↘️ gradually
  - Too high = random behavior
  - Too low = no exploration

- `train/value_loss` - Value function accuracy
  - **Should stabilize** and decrease
  - Spikes indicate instability

- `train/policy_gradient_loss` - Policy update size
  - Should stabilize over time
  - Large fluctuations = unstable training

**PPO-Specific Metrics:**
- `train/clip_fraction` - Fraction of clipped updates
  - Should be moderate (0.1-0.3)
  - Too high = learning too fast
  - Too low = learning too slow

- `train/approx_kl` - KL divergence
  - Measures policy change per update
  - Should remain small (<0.05)

**Evaluation Metrics:**
- `eval/mean_reward` - Reward on evaluation episodes
  - Best indicator of real performance
  - Updated every eval_freq steps

- `eval/mean_ep_length` - Steps on evaluation episodes
  - Should decrease over time

#### 2. **GRAPHS Tab**
Shows the neural network architecture:
- Policy network structure
- Value network structure  
- Layer connections and sizes

#### 3. **DISTRIBUTIONS Tab**
Shows weight and bias distributions:
- Useful for detecting vanishing/exploding gradients
- Histograms should evolve smoothly

#### 4. **HISTOGRAMS Tab**
3D visualization of parameter evolution over time

### Key Metrics Reference Table

| Metric | Good Trend | Target Value | Warning Signs |
|--------|-----------|--------------|---------------|
| `ep_rew_mean` | ↗️ Increasing | >5.0 | Decreasing or flat |
| `ep_len_mean` | ↘️ Decreasing | <50 | Increasing or >100 |
| `entropy_loss` | ↘️ Decreasing | 0.5-2.0 | Stuck at high value |
| `value_loss` | ↘️ Decreasing | <1.0 | Large spikes |
| `approx_kl` | Stable | <0.05 | >0.1 consistently |
| `clip_fraction` | Stable | 0.1-0.3 | >0.5 or <0.05 |

### Interpreting Training Progress

**Healthy Training:**
```
Steps: 0-50k    - ep_rew_mean rises from -15 to 0
Steps: 50k-150k - ep_rew_mean rises from 0 to 5
Steps: 150k+    - ep_rew_mean stabilizes around 5-8
                - ep_len_mean drops to 30-50 steps
```

**Signs of Good Learning:**
- ✅ Reward steadily increases
- ✅ Episode length decreases
- ✅ Value loss decreases and stabilizes
- ✅ Entropy gradually decreases
- ✅ No large spikes or crashes

**Warning Signs:**
- ⚠️ Reward stops improving (plateau)
- ⚠️ Episode length not decreasing
- ⚠️ Large spikes in value_loss
- ⚠️ approx_kl > 0.1 (policy changing too fast)
- ⚠️ Entropy crashes to near 0 (no exploration)

### Comparing Multiple Runs

TensorBoard can overlay multiple experiments:

1. Train multiple configurations:
```bash
python train.py --algorithm ppo_basic --name exp1
python train.py --algorithm ppo_aggressive --name exp2
python train.py --algorithm a2c_basic --name exp3
```

2. Start TensorBoard with all runs:
```bash
tensorboard --logdir runs/
```

3. In the browser:
   - Use the checkboxes to select/deselect runs
   - Click on smoothing slider to reduce noise
   - Use the download button to export data

### Saving Graphs for Reports

**Option 1: Screenshot**
- Navigate to the desired graph
- Use browser screenshot or snipping tool

**Option 2: Download Data**
- Click the download icon (⬇️) on any graph
- Exports as CSV for further analysis

**Option 3: Export Images Programmatically**
```bash
# Export all scalars as images
tensorboard --logdir runs/ --export_dir ./tensorboard_exports/
```

**Option 4: Use TensorBoard's Image Export**
```python
# Add to your analysis script
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# Load TensorBoard logs
ea = event_accumulator.EventAccumulator('runs/ppo_basic_*/tensorboard/')
ea.Reload()

# Get specific metric
steps = [e.step for e in ea.Scalars('rollout/ep_rew_mean')]
values = [e.value for e in ea.Scalars('rollout/ep_rew_mean')]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(steps, values)
plt.xlabel('Training Steps')
plt.ylabel('Episode Reward')
plt.title('Learning Curve - PPO Basic')
plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
```

### TensorBoard Advanced Features

**Smoothing:**
- Adjust the "Smoothing" slider (0-1) to reduce noise
- Recommended: 0.6-0.9 for cleaner visualizations

**Horizontal Axis:**
- Switch between STEP, RELATIVE, or WALL time
- STEP is most useful for comparing runs

**Logarithmic Scale:**
- Click "Show data download links" → "Toggle Y-axis log scale"
- Useful for metrics with large ranges

**Regex Filtering:**
- Use regex in the search box to filter metrics
- Example: `rollout/.*` shows only rollout metrics

### Common TensorBoard Issues

**Issue: TensorBoard shows no data**
- Wait a few seconds, data updates periodically
- Refresh the browser (F5)
- Check that training is running and saving logs

**Issue: Port 6006 already in use**
```bash
# Use a different port
tensorboard --logdir runs/ --port 6007
```

**Issue: TensorBoard won't start**
```bash
# Kill existing TensorBoard processes
pkill -f tensorboard

# Restart
tensorboard --logdir runs/
```

**Issue: Want to access TensorBoard remotely**
```bash
# Bind to all interfaces
tensorboard --logdir runs/ --host 0.0.0.0
```

### Example TensorBoard Workflow

1. **Start training:**
```bash
python train.py --algorithm ppo_basic
```

2. **In another terminal, start TensorBoard:**
```bash
tensorboard --logdir runs/
```

3. **Open browser to http://localhost:6006**

4. **Monitor these metrics every 10-15 minutes:**
   - Is `ep_rew_mean` increasing?
   - Is `ep_len_mean` decreasing?
   - Are there any spikes in `value_loss`?

5. **If training looks good after 100k steps:**
   - Continue to completion
   
6. **If training plateaus:**
   - Stop training (Ctrl+C)
   - Try curriculum learning or different algorithm
   - Check troubleshooting section

7. **After training completes:**
   - Take screenshots of key graphs
   - Export data for your report
   - Compare with other runs

### Generating Report Graphics

For academic reports or presentations:

```bash
# 1. Train multiple algorithms
python train.py --algorithm ppo_basic --name report_ppo
python train.py --algorithm a2c_basic --name report_a2c

# 2. Start TensorBoard
tensorboard --logdir runs/

# 3. In TensorBoard interface:
#    - Select both runs (checkboxes)
#    - Smooth graphs (slider to 0.7)
#    - Take screenshots of:
#      * ep_rew_mean (learning curve)
#      * ep_len_mean (efficiency improvement)
#      * value_loss (stability)
#      * entropy_loss (exploration)

# 4. Save screenshots as:
#    - learning_curve.png
#    - episode_length.png
#    - value_loss.png
#    - entropy_evolution.png
```

---

## 🐛 Troubleshooting

### NumPy Version Issues

**Problem**: `AttributeError: _ARRAY_API not found` or similar numpy errors

**Solution**:
```bash
pip install "numpy<2.0.0" --force-reinstall
```

### Training Too Slow

**Solutions**:
- Use faster algorithm: `python train.py --algorithm a2c_fast`
- Reduce timesteps in `training_configs.py`
- Reduce number of parallel environments
- Use smaller network architecture

### Agent Not Learning

**Solutions**:
- Try aggressive config: `python train.py --algorithm ppo_aggressive`
- Use curriculum learning: `python train.py --algorithm ppo_basic --curriculum easy_to_hard`
- Check TensorBoard for gradient issues
- Increase exploration (higher `ent_coef`)

### Training Unstable

**Solutions**:
- Use conservative config: `python train.py --algorithm ppo_conservative`
- Reduce learning rate
- Increase batch size
- Check for exploding gradients in TensorBoard

### Out of Memory

**Solutions**:
- Reduce parallel environments (`n_envs` in config)
- Use smaller network architecture
- Reduce batch size or buffer size (DQN)
- Use `a2c_fast` with smaller network

### Poor Performance on Hard Scenarios

**Solutions**:
- Use curriculum learning: `--curriculum easy_to_hard`
- Train for more timesteps
- Use conservative config for stability
- Check if model overfits to easy scenarios

### Model Files Not Found

**Solutions**:
- Check path with: `ls -la runs/*/models/`
- Use wildcards carefully: `runs/ppo_basic_*/models/best_model.zip`
- Ensure training completed successfully
- Check for best_model.zip in stage_* directories

---

## 📝 Configuration Details

### Environment Parameters (touristbot_env.py)
- `GRID_SIZE`: 20 (city grid size)
- `CELL_SIZE`: 30 pixels (visualization)
- `view_size`: 5 (partial observation size)
- `max_steps`: 150 (episode limit)
- `use_partial_obs`: True/False (full vs partial observation)

### Training Hyperparameters

See `training_configs.py` for all configurations. Key parameters:

**PPO:**
- `learning_rate`: 1e-4 to 1e-3
- `n_steps`: 1024 to 4096 (steps per update)
- `batch_size`: 128 to 512
- `n_epochs`: 5 to 15
- `clip_range`: 0.1 to 0.3

**A2C:**
- `learning_rate`: 7e-4 to 1e-3
- `n_steps`: 3 to 5 (steps per update)
- `n_envs`: 16 (parallel environments)

**DQN:**
- `learning_rate`: 1e-4
- `buffer_size`: 100k to 150k
- `batch_size`: 128 to 256
- `exploration_fraction`: 0.2 to 0.3

---

## 🏆 Tips & Best Practices

### ✅ DO

- **Start simple**: Begin with `ppo_basic` to establish baseline
- **Monitor always**: Use TensorBoard to track training
- **Test regularly**: Validate models during training
- **Compare multiple runs**: Different algorithms work better for different scenarios
- **Use curriculum**: If standard training plateaus
- **Save reports**: Keep comparison results for documentation
- **Version control**: Keep track of configuration changes

### ❌ DON'T

- Train without monitoring (always use TensorBoard!)
- Compare models with too few episodes (<10)
- Modify configs without backup
- Delete the `runs/` directory (your training history!)
- Ignore convergence issues
- Use outdated models without re-evaluation

### 💡 Optimization Tips

| Problem | Solution |
|---------|----------|
| Not learning | Try `ppo_aggressive` or curriculum learning |
| Unstable training | Use `ppo_conservative` |
| Too slow | Try `a2c_fast` or reduce timesteps |
| Plateauing | Apply curriculum learning |
| Out of memory | Reduce `n_envs` or batch size |
| Overfitting | Increase training timesteps or diversity |

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

### Output Directory

Each training run creates:
```
runs/<algorithm>_<curriculum>_<timestamp>/
├── config.txt              # Full configuration
├── models/
│   ├── stage_1/
│   │   ├── best_model.zip        # ⭐ Best performing (use this!)
│   │   └── checkpoint_*.zip      # Regular saves
│   ├── stage_2/ (if curriculum)
│   ├── stage_3/ (if curriculum)
│   └── final_model.zip           # Model at end of training
├── logs/
│   └── evaluations.npz           # Evaluation metrics
└── tensorboard/                  # Training curves
```


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

**Happy Training! 🚀🤖**
- Curriculum learning disponible

✅ **Interfaz Gráfica Interactiva**
- Visualización en tiempo real
- Campo de texto para entrada natural
- Botones interactivos (Exit, etc.)

✅ **Modelos Pre-entrenados**
- Listo para usar sin entrenar
- Múltiples checkpoints disponibles

## 📝 Notas

- Los archivos `demo*.py` y `test_*.py` son legacy y pueden ignorarse
- Usa solo `touristbot_app.py` para la aplicación principal
- El modelo zero-shot se carga automáticamente la primera vez (puede tardar unos segundos)

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
