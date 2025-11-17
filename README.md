# 🌆 TouristBot - Agente de Navegación con RL

Agente que aprende a navegar por una ciudad 2D para llegar a lugares específicos (restaurantes, museos) usando Reinforcement Learning.

```

## 📁 Estructura del Proyecto

```
proyecto/
├── touristbot_env.py       # Entorno (Grid 10x10, vista parcial 5x5)
├── train_ppo_basic.py      # Entrenamiento básico PPO
├── train_advanced.py       # Curriculum + Comparación + Tuning
├── analyze_results.py      # Análisis y visualización
├── demo.py                 # Demo interactiva
├── utils.py                # Utilidades (wrappers, callbacks)
```

## 🎮 Uso Rápido

### 1. Entrenar modelo básico

```bash
python train_ppo_basic.py --train
```

### 2. Probar modelo entrenado

```bash
python demo.py --model models/ppo_basic/ppo_touristbot_final.zip
```

### 3. Ver progreso en TensorBoard

```bash
tensorboard --logdir ./tensorboard/ppo_basic/
```

## 🎓 Técnicas Avanzadas

### Curriculum Learning
```bash
python train_advanced.py --mode curriculum
```
Entrena progresivamente reduciendo tiempo disponible (150→100→75 pasos).

### Comparar Algoritmos (PPO vs SAC vs DQN)
```bash
python train_advanced.py --mode compare --timesteps 100000
```

### Hyperparameter Tuning (Optuna)
```bash
python train_advanced.py --mode tune --trials 50
```

## 📊 Análisis de Resultados

```bash
# Curvas de aprendizaje
python analyze_results.py --plot-learning logs/ppo_basic/

# Visualizar política
python analyze_results.py --visualize-policy models/ppo_basic/best_model.zip

# Reporte completo
python analyze_results.py --full-report models/ppo_basic/best_model.zip logs/ppo_basic/
```

## 🎯 Características del Entorno

- **Grid**: 20x20 celdas (ciudad expandida)
- **Estructura**: Ciudad con calles estilo Manhattan
  - Calles transitables cada 4 celdas (horizontal y vertical)
  - Edificios bloqueados entre calles
  - 204 celdas transitables, 196 bloqueadas
- **Observación**: Vista parcial 5x5 (28 valores)
  - Codificación: 0=edificio, 1=calle, 2=restaurant, 3=museum, 4=agente
- **Acciones**: 4 direccionales (↑↓←→)
  - Solo se puede mover por calles, los edificios bloquean el movimiento
- **Reward shaping**: Potencial basado en distancia + exploration bonus
- **Tiempo máximo**: 200 pasos (aumentado por el tamaño del grid)
- **Compatible**: Gymnasium, Stable-Baselines3

### 🏙️ Visualizar la Estructura de Ciudad

```bash
python demo_city.py
```
Este script muestra el mapa de calles y edificios tanto en texto como visualmente.

## 📈 Configuración

Editar `CONFIG` en `train_ppo_basic.py`:

```python
CONFIG = {
    "use_partial_obs": True,    # Vista parcial
    "view_size": 5,              # Tamaño vista
    "n_envs": 4,                 # Entornos paralelos
    "total_timesteps": 200000,   # Timesteps
    "learning_rate": 3e-4,       # Learning rate
}
```

## 🏆 Resultados Esperados

- **Tasa de éxito**: 70-95%
- **Pasos promedio**: 10-25
- **Tiempo entrenamiento**: 10-30 min (CPU)

---

**Autoras**: Sara Regalado | Zaloa Fernandez | Universidad de Deusto 2025-2026
