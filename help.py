#!/usr/bin/env python3
"""
Script de ayuda para mostrar información del proyecto
"""

import os

def print_header():
    print("\n" + "="*70)
    print(" " * 20 + "🌆 TOURISTBOT")
    print(" " * 10 + "Navegación con RL y Lenguaje Natural")
    print("="*70 + "\n")

def print_quick_help():
    print("📖 AYUDA RÁPIDA\n")
    print("Para ejecutar la aplicación:")
    print("  $ python touristbot_app.py")
    print()
    print("Controles en la ventana:")
    print("  • Presiona 'T' para escribir tu destino")
    print("  • Escribe en lenguaje natural (ej: 'quiero comer')")
    print("  • Presiona ENTER para confirmar")
    print("  • ESC o botón EXIT para salir")
    print()
    print("Más opciones:")
    print("  $ python touristbot_app.py --help")
    print()
    print("Documentación completa:")
    print("  • README.md - Documentación detallada")
    print("  • QUICK_START.md - Guía rápida")
    print()

def check_models():
    model_path = "models/ppo_basic/best_model.zip"
    if os.path.exists(model_path):
        print(f"✅ Modelo encontrado: {model_path}")
    else:
        print("⚠️  No se encontró modelo entrenado")
        print("   Ejecuta: python train_ppo_basic.py --train")
    print()

def main():
    print_header()
    check_models()
    print_quick_help()
    print("="*70)
    print("Para más información, consulta README.md o QUICK_START.md")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
