# tourist_city_env.py — versión modificada de tu SnakeEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque
from typing import Dict, List, Tuple, Optional

# ======================
# CONFIGURACIÓN
# ======================
TABLE_SIZE = 500
CELL_SIZE = 25  # Cada celda = 25x25 px → grid de 20x20
GRID_SIZE = TABLE_SIZE // CELL_SIZE  # 20

# Tipos de lugares con sus colores (BGR para OpenCV)
PLACE_COLORS = {
    "restaurant": (0, 165, 255),   # naranja (rustic: marrón claro → (42, 82, 190))
    "museum": (0, 0, 255),         # rojo
    "cafe": (0, 255, 255),         # amarillo
    "parking": (128, 128, 128),    # gris
    "hotel": (255, 0, 0),          # azul
    "empty": (0, 0, 0),            # negro
    "agent": (255, 255, 255),      # blanco
}

# Atributos semánticos (simulados; en real usarías zero-shot)
PLACE_ATTRIBUTES = {
    "restaurant_0": {"type": "restaurant", "pos": None, "tags": ["rustic", "cheap"], "price": 2},
    "restaurant_1": {"type": "restaurant", "pos": None, "tags": ["elegant", "expensive"], "price": 5},
    "museum_0": {"type": "museum", "pos": None, "tags": ["classic"], "price": 0},
    "cafe_0": {"type": "cafe", "pos": None, "tags": ["cozy", "outdoor"], "price": 1},
    "parking_0": {"type": "parking", "pos": None, "tags": ["street"], "capacity": 10, "occupied": 3},
}


class TouristCityEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, instruction: str = "go to the rustic restaurant"):
        super().__init__()
        self.instruction = instruction
        
        # Acciones: 0=adelante, 1=giro izq, 2=giro der, 3=interactuar, 4=esperar
        self.action_space = spaces.Discrete(5)
        
        # Observación: vista parcial 5x5 (RGB-like) + embedding de instrucción (simulado 384-dim)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(5, 5, 3), dtype=np.uint8),
            "instruction": spaces.Box(low=-5.0, high=5.0, shape=(384,), dtype=np.float32),
        })

        self.img = np.zeros((TABLE_SIZE, TABLE_SIZE, 3), dtype=np.uint8)
        self.agent_pos = [1, 1]  # (fila, col)
        self.agent_dir = 0  # 0: ↑, 1: →, 2: ↓, 3: ←
        self.grid = None  # grid[20][20]: id del objeto o -1
        self.objects = {}  # nombre → {type, pos, attrs}
        self.step_count = 0
        self.max_steps = 100

        # Para simular embedding (en real usarías SentenceTransformer)
        self._dummy_embeddings = {
            "go to the rustic restaurant": np.random.randn(384).astype(np.float32) * 0.1,
            "find a cheap place to park": np.random.randn(384).astype(np.float32) * 0.1,
            "visit the free museum": np.random.randn(384).astype(np.float32) * 0.1,
        }

    def _generate_city(self):
        """Genera una ciudad 20x20 con lugares estratégicos"""
        self.grid = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=int)  # -1 = vacío
        self.objects = {}

        # Posición inicial del agente
        self.agent_pos = [1, 1]
        self.grid[1, 1] = -2  # agente

        # Colocar lugares (evitando colisiones)
        locations = list(PLACE_ATTRIBUTES.keys())
        random.shuffle(locations)

        for name in locations:
            while True:
                r, c = random.randint(2, GRID_SIZE - 3), random.randint(2, GRID_SIZE - 3)
                if self.grid[r, c] == -1:
                    self.grid[r, c] = len(self.objects)
                    self.objects[name] = PLACE_ATTRIBUTES[name].copy()
                    self.objects[name]["pos"] = [r, c]
                    break

    def _get_partial_view(self) -> np.ndarray:
        """Vista parcial 5x5 centrada en el agente, orientada según dirección"""
        view = np.zeros((5, 5, 3), dtype=np.uint8)
        r0, c0 = self.agent_pos

        # Mapa de offset según dirección (↑, →, ↓, ←)
        offsets = {
            0: [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
                (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
                ( 0, -2), ( 0, -1), ( 0, 0), ( 0, 1), ( 0, 2),
                ( 1, -2), ( 1, -1), ( 1, 0), ( 1, 1), ( 1, 2),
                ( 2, -2), ( 2, -1), ( 2, 0), ( 2, 1), ( 2, 2)],
            1: [(-2,  2), (-1,  2), ( 0,  2), ( 1,  2), ( 2,  2),
                (-2,  1), (-1,  1), ( 0,  1), ( 1,  1), ( 2,  1),
                (-2,  0), (-1,  0), ( 0,  0), ( 1,  0), ( 2,  0),
                (-2, -1), (-1, -1), ( 0, -1), ( 1, -1), ( 2, -1),
                (-2, -2), (-1, -2), ( 0, -2), ( 1, -2), ( 2, -2)],
            # ↓ y ← se pueden añadir...
        }

        for idx, (dr, dc) in enumerate(offsets.get(self.agent_dir, offsets[0])):
            r, c = r0 + dr, c0 + dc
            i, j = idx // 5, idx % 5
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                obj_id = self.grid[r, c]
                if obj_id == -2:
                    color = PLACE_COLORS["agent"]
                elif obj_id == -1:
                    color = PLACE_COLORS["empty"]
                else:
                    name = list(self.objects.keys())[obj_id]
                    obj_type = self.objects[name]["type"]
                    color = PLACE_COLORS.get(obj_type, (255, 255, 255))
                view[i, j] = color
            else:
                view[i, j] = (50, 50, 50)  # pared (gris oscuro)

        return view

    def step(self, action: int):
        self.step_count += 1
        
        # Guardar posición anterior
        old_r, old_c = self.agent_pos

        # 1. Ejecutar acción
        if action == 0:  # adelante
            dr = [-1, 0, 1, 0][self.agent_dir]
            dc = [0, 1, 0, -1][self.agent_dir]
            new_r, new_c = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if 0 <= new_r < GRID_SIZE and 0 <= new_c < GRID_SIZE:
                # Actualizar grid: limpiar posición anterior
                if self.grid[old_r, old_c] == -2:
                    self.grid[old_r, old_c] = -1
                # Mover agente
                self.agent_pos = [new_r, new_c]
                # Actualizar grid: nueva posición (sin sobrescribir objetos)
                if self.grid[new_r, new_c] == -1:
                    self.grid[new_r, new_c] = -2
        elif action == 1:  # giro izq
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == 2:  # giro der
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == 3:  # interactuar
            r, c = self.agent_pos
            obj_id = self.grid[r, c]
            if obj_id >= 0:
                obj_name = list(self.objects.keys())[obj_id]
                obj = self.objects[obj_name]
                # Aquí iría la lógica de reward shaping semántico ✅
                # Ej: si instrucción es "rustic restaurant" y este lo es → +5
                pass

        # 2. Observación
        image = self._get_partial_view()
        instr_emb = self._dummy_embeddings.get(self.instruction, np.zeros(384, dtype=np.float32))
        observation = {"image": image, "instruction": instr_emb}

        # 3. Recompensa y finalización
        reward = -0.1  # small step penalty
        terminated = False
        truncated = self.step_count >= self.max_steps

        # Ejemplo de reward shaping semántico (simulado)
        if action == 3:  # interactuar
            r, c = self.agent_pos
            obj_id = self.grid[r, c]
            if obj_id >= 0:
                obj_name = list(self.objects.keys())[obj_id]
                obj = self.objects[obj_name]
                # Aquí conectarías con tu zero-shot classifier
                if "rustic" in obj.get("tags", []) and "restaurant" in self.instruction:
                    reward += 5.0
                if obj.get("price", 0) <= 2 and "cheap" in self.instruction:
                    reward += 3.0
                if obj["type"] == "parking" and "park" in self.instruction:
                    reward += 4.0

        info = {"step": self.step_count}
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        if options and "instruction" in options:
            self.instruction = options["instruction"]
        self._generate_city()
        image = self._get_partial_view()
        instr_emb = self._dummy_embeddings.get(self.instruction, np.zeros(384, dtype=np.float32))
        observation = {"image": image, "instruction": instr_emb}
        return observation, {}

    def render(self, mode="human"):
        self.img = np.zeros((TABLE_SIZE, TABLE_SIZE, 3), dtype=np.uint8)

        # Dibujar grid
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                obj_id = self.grid[r, c]
                
                # Determinar color base de la celda
                if obj_id == -1:
                    color = PLACE_COLORS["empty"]
                elif obj_id == -2:
                    color = PLACE_COLORS["agent"]
                else:
                    name = list(self.objects.keys())[obj_id]
                    obj_type = self.objects[name]["type"]
                    color = PLACE_COLORS.get(obj_type, (255, 255, 255))
                
                # Dibujar celda
                cv2.rectangle(
                    self.img,
                    (c * CELL_SIZE, r * CELL_SIZE),
                    ((c + 1) * CELL_SIZE, (r + 1) * CELL_SIZE),
                    color,
                    -1
                )
        
        # Dibujar agente encima (siempre visible)
        agent_r, agent_c = self.agent_pos
        cv2.rectangle(
            self.img,
            (agent_c * CELL_SIZE, agent_r * CELL_SIZE),
            ((agent_c + 1) * CELL_SIZE, (agent_r + 1) * CELL_SIZE),
            PLACE_COLORS["agent"],
            -1
        )

        # Dibujar instrucción en pantalla
        cv2.putText(
            self.img,
            f"Instruction: {self.instruction}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.imshow("Tourist City", self.img)
        cv2.waitKey(1)
        time.sleep(0.1)

    def close(self):
        cv2.destroyAllWindows()