# 🌆 TouristBot - Navegación con RL y Lenguaje Natural

Agente que navega por una ciudad 2D hacia lugares específicos (restaurantes, museos, tiendas, cines) usando Reinforcement Learning y procesamiento de lenguaje natural (Zero-Shot Classification).

## 🚀 Inicio Rápido

### Ejecutar la aplicación (modo interactivo)

```bash
python touristbot_app.py
```

Esto inicia la interfaz gráfica donde puedes:
- Presionar **'T'** para escribir tu destino en lenguaje natural
- El agente navegará automáticamente usando el modelo RL entrenado
- **ESC** o botón **EXIT** para salir

### Ejemplos de entrada:
- "Quiero comer algo"
- "Necesito ir a una tienda"
- "Llévame al museo"
- "Busca un cine"

## 📁 Estructura del Proyecto

```
TouristBot_RL/
├── touristbot_app.py       # 🎯 APLICACIÓN PRINCIPAL
├── touristbot_env.py       # Entorno RL (ciudad 20x20, vista parcial 5x5)
├── train_ppo_basic.py      # Entrenamiento PPO básico
├── train_advanced.py       # Curriculum learning y comparación
├── analyze_results.py      # Análisis y visualización
├── utils.py                # Utilidades (wrappers, callbacks)
├── requirements.txt        # Dependencias
└── models/                 # Modelos entrenados
    └── ppo_basic/
        └── best_model.zip  # Mejor modelo
```

## 🎮 Modos de Uso

### 1. Modo Interactivo (por defecto)

```bash
python touristbot_app.py
```

### 2. Episodio único con texto

```bash
python touristbot_app.py --mode single --text "Quiero ir al museo"
```

### 3. Usar modelo específico

```bash
python touristbot_app.py --model models/ppo_basic/best_model.zip
```

### 4. Sin visualización (solo métricas)

```bash
python touristbot_app.py --mode single --text "Busca un restaurante" --no-viz
```

## 🧠 Cómo Funciona

### 1. **Procesamiento de Lenguaje Natural (Zero-Shot)**
   - Usa un modelo BERT en español para clasificar la intención del usuario
   - Mapea texto libre a categorías: restaurante, museo, tienda, cine
   - No requiere entrenamiento adicional

### 2. **Navegación con RL**
   - Agente PPO entrenado para navegar eficientemente
   - Vista parcial 5x5 para simular visión limitada
   - Recompensa basada en distancia + penalización por tiempo

### 3. **Interfaz Interactiva**
   - Visualización en tiempo real con OpenCV
   - Campo de texto para entrada en lenguaje natural
   - Botón EXIT y navegación con teclado

## 🎓 Entrenamiento (Opcional)

Si quieres entrenar tu propio modelo:

### Entrenamiento básico
```bash
python train_ppo_basic.py --train
```

### Curriculum Learning
```bash
python train_advanced.py --mode curriculum
```

### Comparar algoritmos (PPO vs SAC vs DQN)
```bash
python train_advanced.py --mode compare --timesteps 100000
```

### Optimización de hiperparámetros
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

## 🔧 Configuración

### Requisitos
```bash
pip install -r requirements.txt
```

Principales dependencias:
- `stable-baselines3`: Algoritmos RL
- `gymnasium`: API de entornos
- `transformers`: Modelos NLP (Zero-Shot)
- `opencv-python`: Visualización
- `torch`: Backend para NLP

### Variables de entorno (touristbot_env.py)
- `GRID_SIZE`: Tamaño del grid (20x20)
- `CELL_SIZE`: Tamaño de cada celda en píxeles (30)
- `view_size`: Tamaño de vista parcial (5x5)
- `max_steps`: Pasos máximos por episodio (150)

## 🎯 Características Principales

✅ **Procesamiento de Lenguaje Natural**
- Zero-shot classification con BERT en español
- Sin necesidad de datos de entrenamiento adicionales
- Mapeo automático de intenciones a lugares

✅ **Reinforcement Learning**
- Algoritmo PPO optimizado
- Vista parcial para mayor realismo
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
