"""
TouristBot - Aplicación Principal
Navegación con RL usando entrada de lenguaje natural (Zero-Shot Classification)
"""

import argparse
import os
from pathlib import Path
from stable_baselines3 import PPO
from touristbot_env import TouristBotEnv
import time


class TouristBotApp:
    """Aplicación principal de TouristBot con NLP"""
    
    def __init__(self, model_path=None):
        """
        Inicializar aplicación
        
        Args:
            model_path: Ruta al modelo PPO entrenado. Si es None, busca el mejor modelo disponible.
        """
        self.model_path = self._find_model(model_path)
        self.model = None
        self.env = None
        
        print("="*70)
        print(" " * 25 + "TOURISTBOT")
        print(" " * 15 + "Navegación con Lenguaje Natural")
        print("="*70)
        
    def _find_model(self, model_path):
        """Encuentra el mejor modelo disponible"""
        if model_path and os.path.exists(model_path):
            return model_path
        
        # Buscar modelos en orden de preferencia
        model_dirs = [
            "models/ppo_basic",
            "models/curriculum",
            "models"
        ]
        
        model_priorities = [
            "best_model.zip",
            "ppo_touristbot_final.zip",
            "ppo_touristbot_160000_steps.zip",
            "ppo_touristbot_80000_steps.zip"
        ]
        
        for model_dir in model_dirs:
            if not os.path.exists(model_dir):
                continue
            for model_name in model_priorities:
                full_path = os.path.join(model_dir, model_name)
                if os.path.exists(full_path):
                    print(f"\n✓ Modelo encontrado: {full_path}")
                    return full_path
        
        raise FileNotFoundError(
            "No se encontró ningún modelo entrenado. "
            "Ejecuta train_ppo_basic.py primero para entrenar un modelo."
        )
    
    def load_model(self):
        """Cargar modelo PPO"""
        print(f"\nCargando modelo desde: {self.model_path}")
        try:
            self.model = PPO.load(self.model_path)
            print("✓ Modelo cargado correctamente")
        except Exception as e:
            print(f"✗ Error al cargar modelo: {e}")
            raise
    
    def create_env(self):
        """Crear entorno con NLP habilitado"""
        self.env = TouristBotEnv(
            goal_type="restaurant",  # Inicial, se cambiará con NLP
            use_nlp=True,
            render_mode="human",
            use_partial_obs=True,
            view_size=5
        )
        print("✓ Entorno creado con NLP habilitado")
    
    def run_interactive(self):
        """Ejecutar modo interactivo con GUI"""
        if self.model is None:
            self.load_model()
        
        if self.env is None:
            self.create_env()
        
        print("\n" + "="*70)
        print("MODO INTERACTIVO")
        print("="*70)
        print("\nInstrucciones:")
        print("  1. Presiona 'T' en la ventana para activar entrada de texto")
        print("  2. Escribe tu destino en lenguaje natural")
        print("  3. Presiona ENTER para confirmar")
        print("  4. El agente navegará hacia el destino indicado")
        print("  5. Escribe nuevas instrucciones para cambiar destino")
        print("  6. Presiona ESC o haz clic en EXIT para salir")
        print("\nEjemplos de entrada:")
        print("  • 'Quiero comer algo'")
        print("  • 'Necesito ir a una tienda'")
        print("  • 'Llévame al museo'")
        print("  • 'Busca un cine'")
        print("="*70)
        
        # Reset inicial
        observation, info = self.env.reset()
        
        print(f"\n✓ Entorno inicializado")
        print(f"✓ Posición del agente: {self.env.agent_pos}")
        print("\n⏸️  El agente está esperando instrucciones...")
        print("Presiona 'T' en la ventana para dar tu primera instrucción")
        
        running = True
        step_count = 0
        episode_count = 0
        
        try:
            while running:
                # Renderizar
                self.env.render()
                
                # Verificar si se solicitó salir
                if self.env.exit_requested:
                    print("\n✓ Saliendo de la aplicación...")
                    running = False
                    continue
                
                # Verificar si hay nueva instrucción
                if self.env.new_instruction_received:
                    print(f"\n🚀 Iniciando navegación hacia: {self.env.goal_type.upper()}")
                    self.env.new_instruction_received = False
                    self.env.navigating = True
                    step_count = 0
                    # Actualizar observación después del reset interno
                    observation = self.env._get_observation()
                
                # Solo navegar si está en modo navegación y no escribiendo
                if self.env.navigating and not self.env.text_input_active:
                    # Usar modelo para predecir acción
                    action, _states = self.model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = self.env.step(action)
                    step_count += 1
                    
                    if terminated:
                        episode_count += 1
                        print(f"\n🎉 ¡Objetivo alcanzado en {step_count} pasos! (Episodio #{episode_count})")
                        print("Presiona 'T' para dar nueva instrucción, o ESC/EXIT para salir")
                        
                        # Detener navegación - esperar nueva instrucción
                        self.env.navigating = False
                        step_count = 0
                    
                    elif truncated:
                        episode_count += 1
                        print(f"\n⏱️ Tiempo agotado después de {step_count} pasos (Episodio #{episode_count})")
                        print("Presiona 'T' para dar nueva instrucción")
                        
                        # Detener navegación - esperar nueva instrucción
                        self.env.navigating = False
                        step_count = 0
                    
                    time.sleep(0.05)  # Ralentizar para visualización
                else:
                    # Pequeña espera cuando está quieto
                    time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrumpido por el usuario")
        
        finally:
            print("\n" + "="*70)
            print(f"ESTADÍSTICAS FINALES:")
            print(f"  • Episodios completados: {episode_count}")
            print("="*70)
            self.env.close()
            print("\n✓ Entorno cerrado. ¡Hasta luego!")
    
    def run_single_episode(self, text_input=None, visualize=True):
        """
        Ejecutar un solo episodio con entrada de texto opcional
        
        Args:
            text_input: Texto en lenguaje natural para establecer objetivo
            visualize: Si True, muestra la ventana de visualización
        """
        if self.model is None:
            self.load_model()
        
        if self.env is None:
            self.create_env()
        
        # Configurar modo de renderizado
        if not visualize:
            self.env.render_mode = None
        
        # Reset con entrada de texto si se proporciona
        reset_options = {}
        if text_input:
            reset_options["nlp_input"] = text_input
        
        observation, info = self.env.reset(options=reset_options)
        
        print(f"\n{'='*70}")
        print("NUEVO EPISODIO")
        print(f"{'='*70}")
        if text_input:
            print(f"Entrada del usuario: '{text_input}'")
        print(f"Objetivo: {info['goal_type'].upper()}")
        print(f"Posición objetivo: {info['goal_position']}")
        print(f"Posición agente: {self.env.agent_pos}")
        print(f"{'='*70}\n")
        
        done = False
        step_count = 0
        total_reward = 0
        
        while not done:
            if visualize:
                self.env.render()
            
            action, _states = self.model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            
            if visualize:
                time.sleep(0.05)
        
        # Resultados
        print(f"\n{'='*70}")
        if terminated:
            print("✓ EPISODIO COMPLETADO - ¡Objetivo alcanzado!")
        else:
            print("✗ EPISODIO TRUNCADO - Tiempo agotado")
        print(f"{'='*70}")
        print(f"Pasos: {step_count}")
        print(f"Recompensa total: {total_reward:.2f}")
        print(f"{'='*70}\n")
        
        if visualize:
            time.sleep(2)
            self.env.close()
        
        return {
            "success": terminated,
            "steps": step_count,
            "reward": total_reward,
            "goal_type": info['goal_type']
        }


def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="TouristBot - Navegación con RL y Lenguaje Natural"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ruta al modelo PPO (por defecto: busca el mejor disponible)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "single"],
        default="interactive",
        help="Modo de ejecución: interactive (GUI interactiva) o single (un episodio)"
    )
    
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Texto en lenguaje natural para modo 'single'"
    )
    
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Desactivar visualización (solo para modo 'single')"
    )
    
    args = parser.parse_args()
    
    # Crear aplicación
    app = TouristBotApp(model_path=args.model)
    
    if args.mode == "interactive":
        app.run_interactive()
    else:
        # Modo single
        if not args.text:
            print("\n⚠️ Modo 'single' requiere --text. Usando ejemplo por defecto.")
            args.text = "Quiero ir a un restaurante"
        
        app.run_single_episode(
            text_input=args.text,
            visualize=not args.no_viz
        )


if __name__ == "__main__":
    main()
