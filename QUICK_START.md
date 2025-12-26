# 🌆 TouristBot - Guía Rápida

## 🚀 Ejecución

### Ejecutar la aplicación:
```bash
python touristbot_app.py
```

### Controles:
- **T**: Activar entrada de texto
- **Escribe tu instrucción** en lenguaje natural
- **ENTER**: Confirmar y comenzar navegación
- El agente navegará automáticamente hasta completar el objetivo
- **Escribe nueva instrucción** para cambiar destino
- **ESC** o botón **EXIT**: Salir

### Ejemplos de texto:
- "Quiero comer algo"
- "Llévame al museo"
- "Busca una tienda"
- "Vamos al cine"

## 📝 Otros comandos

### Episodio único con texto específico:
```bash
python touristbot_app.py --mode single --text "Quiero ir al restaurante"
```

### Usar modelo específico:
```bash
python touristbot_app.py --model models/ppo_basic/best_model.zip
```

### Sin visualización (solo resultados):
```bash
python touristbot_app.py --mode single --text "Museo" --no-viz
```

## 🛠️ Instalación

```bash
pip install -r requirements.txt
```

## 📂 Estructura

- `touristbot_app.py` - **Aplicación principal** ⭐
- `touristbot_env.py` - Entorno de RL
- `config.py` - Configuración
- `train_ppo_basic.py` - Entrenar modelo (opcional)
- `legacy/` - Scripts antiguos (ignorar)

## 🎯 Tecnologías

- **RL**: PPO (Stable-Baselines3)
- **NLP**: Zero-Shot BERT en español
- **Visualización**: OpenCV
