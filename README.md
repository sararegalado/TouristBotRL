# 🌆 TouristBot - Reinforcement Learning Navigation Agent

Un agente de navegación turística que aprende a moverse por una ciudad 2D y completar tareas siguiendo instrucciones en lenguaje natural.

## 📋 Descripción del Proyecto

TouristBot es un proyecto de Reinforcement Learning que combina navegación espacial con procesamiento de lenguaje natural. El agente (un turista) debe aprender a:

- 🗺️ Navegar por una ciudad grid 2D
- 🎯 Alcanzar objetivos específicos (restaurantes, museos, cafés, etc.)
- 📝 Interpretar instrucciones en lenguaje natural
- 🧠 Generalizar a nuevas instrucciones (zero-shot learning)

## 🚀 Versión Actual: v1.0 (Básica)

Esta es la primera iteración con funcionalidades mínimas:

### Características implementadas ✅
- Grid 10x10
- 1 agente (turista)
- 2 tipos de lugares: restaurante y museo
- 4 acciones básicas: arriba, abajo, izquierda, derecha
- Sistema de recompensas básico
- Visualización con OpenCV
- Compatible con Gymnasium

### Estado del entorno
```python
observation = [agent_x, agent_y, goal_x, goal_y, goal_type_id]
```

### Acciones
- `0`: Arriba (↑)
- `1`: Abajo (↓)
- `2`: Izquierda (←)
- `3`: Derecha (→)

## 📦 Instalación

### Requisitos
```bash
pip install gymnasium
pip install numpy
pip install opencv-python
```

### Uso básico
```python
from touristbot_env import TouristBotEnv

# Crear entorno
env = TouristBotEnv(goal_type="restaurant")

# Reset
observation, info = env.reset()

# Ejecutar paso
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)

# Renderizar
env.render()
```

## 🧪 Probar el Entorno

### Opción 1: Script directo
```bash
cd /Users/sara/Documents/deusto_2025_2026/Aprendizaje\ por\ refuerzo/proyecto
python touristbot_env.py
```

### Opción 2: Notebook interactivo
```bash
jupyter notebook test_touristbot.ipynb
```

## 📁 Estructura del Proyecto

```
proyecto/
├── touristbot_env.py           # Entorno principal
├── test_touristbot.ipynb       # Notebook de pruebas
├── Snake_env/                  # Entorno base (referencia)
│   └── snakeenv.py
└── README.md                   # Este archivo
```

## 🎯 Roadmap - Próximas Versiones

### v1.1 - Más lugares y atributos
- [ ] Añadir cafés, parking, tiendas, hoteles
- [ ] Atributos semánticos: precio, atmósfera, ocupación
- [ ] Grid más grande (20x20)

### v1.2 - Instrucciones en lenguaje natural
- [ ] Instrucciones como "busca un restaurante barato"
- [ ] Embeddings de instrucciones (Sentence-BERT)
- [ ] Espacio de observación multimodal

### v1.3 - Zero-shot classification
- [ ] Integrar clasificador zero-shot (Hugging Face)
- [ ] Mapear texto → estructura semántica
- [ ] Reward shaping basado en atributos

### v2.0 - Entrenamiento con RL
- [ ] Implementar PPO con Stable Baselines3
- [ ] Política condicionada por instrucciones
- [ ] Curriculum learning
- [ ] Evaluación zero-shot

### v2.1 - Vista parcial y realismo
- [ ] Vista parcial del agente (7x7)
- [ ] Observación visual con CNN
- [ ] Generación procedural de ciudades
- [ ] Diferentes layouts (Barrio Gótico, Zona Moderna, etc.)

### v3.0 - Features avanzadas
- [ ] Múltiples objetivos secuenciales
- [ ] Personas en movimiento (lugares concurridos)
- [ ] Inventario (dinero, tickets)
- [ ] Demo interactiva con Streamlit

## 📊 Sistema de Recompensas (v1.0)

```python
+10.0  # Alcanzar el objetivo
+0.5   # Acercarse al objetivo
-0.5   # Alejarse del objetivo
-0.1   # Cada paso (penalización de eficiencia)
-5.0   # Exceder máximo de pasos
```

## 🧠 Arquitectura Futura (v2.0+)

```
┌─────────────────┐
│  Instrucción    │ → Sentence-BERT → [384-dim embedding]
└─────────────────┘                            ↓
                                          ┌─────────┐
┌─────────────────┐                       │         │
│  Vista Grid     │ → CNN → [256-dim] →  │  Fusion │ → Policy (PPO)
└─────────────────┘                       │   MLP   │
                                          └─────────┘
```

## 🔬 Comparación con Snake Environment

| Característica | Snake | TouristBot v1.0 |
|---------------|-------|-----------------|
| Grid size | 50x50 | 10x10 |
| Objetivo | Comer manzanas | Llegar a lugares |
| Acciones | 4 direcciones | 4 direcciones |
| Observación | Posición + historial | Posición + objetivo |
| Crecimiento | Sí (snake crece) | No |
| Auto-colisión | Sí (pierde) | No |
| Complejidad | Media | Baja (v1.0) |

## 🤝 Contribuciones

Este es un proyecto académico para el curso de Aprendizaje por Refuerzo.

### Autor
- Sara Regalado
- Universidad de Deusto
- 2025-2026

## 📄 Licencia

MIT License - Uso académico

## 📚 Referencias

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [MiniGrid Environment](https://github.com/Farama-Foundation/Minigrid)

---

**Versión**: 1.0.0  
**Última actualización**: 12 de noviembre de 2025  
**Estado**: 🟢 Funcional (básico)
