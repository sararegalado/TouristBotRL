import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

# Configuration
GRID_SIZE = 20  # Grid 20x20
CELL_SIZE = 30  # Size of each cell = 30x30 px
TABLE_SIZE = GRID_SIZE * CELL_SIZE  # 600x600 px

# RGB colors for visualization
COLORS = {
    "background": (0, 0, 0),        # Black
    "agent": (255, 255, 255),       # White (agent)
    "restaurant": (0, 165, 255),    # Orange
    "museum": (0, 0, 255),          # Red
    "visited": (50, 50, 50),        # Dark grey
    "street": (80, 80, 80),         # Bright grey (streets)
    "building": (40, 40, 40),       # Dark grey (Buildings)
}

# Types of different places
PLACE_TYPES = ["restaurant", "museum"]

# Define the environment class
class TouristBotEnv(gym.Env):
    metadata = {
        'render_modes': ['human'],
        'render_fps': 10
    }

    def __init__(self, goal_type="restaurant", render_mode=None, use_partial_obs=True, view_size=5):
        super(TouristBotEnv, self).__init__()
        
        self.render_mode = render_mode
        self.use_partial_obs = use_partial_obs  # Vista parcial vs completa
        self.view_size = view_size  # Tamaño de la vista parcial (5x5 por defecto)
        
        # Espacio de acciones: 4 movimientos direccionales
        # 0: arriba, 1: abajo, 2: izquierda, 3: derecha
        self.action_space = spaces.Discrete(4)
        
        # Espacio de observación depende del modo
        if use_partial_obs:
            # Vista parcial: grid view_size x view_size + info del objetivo
            # Cada celda codifica: 0=edificio, 1=calle, 2=restaurant, 3=museum, 4=agente
            # + información adicional: [distancia_x, distancia_y, tipo_objetivo]
            obs_size = view_size * view_size + 3
            self.observation_space = spaces.Box(
                low=-GRID_SIZE,
                high=GRID_SIZE,
                shape=(obs_size,),
                dtype=np.float32
            )
        else:
            # Observación completa: [agent_x, agent_y, goal_x, goal_y, goal_type]
            self.observation_space = spaces.Box(
                low=0,
                high=GRID_SIZE,
                shape=(5,),
                dtype=np.float32
            )
        
        # Estado del entorno
        self.agent_pos = [0, 0]  # Posición del agente [x, y]
        self.goal_type = goal_type  # Tipo de lugar objetivo
        self.goal_pos = [0, 0]  # Posición del objetivo
        self.places = {}  # Diccionario de lugares {tipo: posición}
        
        # Estructura de la ciudad (matriz que indica qué es cada celda)
        # 0 = edificio (bloqueado), 1 = calle (transitable)
        self.city_map = self._generate_city_map()
        
        # Visualización
        self.img = np.zeros((TABLE_SIZE, TABLE_SIZE, 3), dtype='uint8')
        
        # Métricas
        self.steps = 0
        self.max_steps = 200  # Aumentado por el tamaño del grid
        self.total_reward = 0
        self.visited_cells = set()  # Para reward shaping
        
        print("🌆 TouristBot Environment v2.0 inicializado")
        print(f"   Grid: {GRID_SIZE}x{GRID_SIZE}")
        print(f"   Vista: {'Parcial ' + str(view_size) + 'x' + str(view_size) if use_partial_obs else 'Completa'}")
        print(f"   Estructura: Ciudad con calles")
        print(f"   Objetivo inicial: {goal_type}")

    def reset(self, *, seed=None, options=None):
        """Reinicia el entorno al estado inicial"""
        # IMPORTANTE: Llamar a super().reset() primero para manejar la seed correctamente
        super().reset(seed=seed)
        
        # Permitir cambiar el objetivo desde options
        if options and "goal_type" in options:
            self.goal_type = options["goal_type"]
        
        # Regenerar el mapa de la ciudad con la nueva seed
        self.city_map = self._generate_city_map()
        
        # Posición inicial del agente en una calle (buscar una posición válida)
        self.agent_pos = self._find_random_street_position()
        
        # Generar lugares de forma aleatoria pero en calles
        self._generate_places()
        
        # Establecer el objetivo según el tipo (hacer copia para evitar referencias)
        self.goal_pos = self.places[self.goal_type].copy()
        
        # Resetear métricas
        self.steps = 0
        self.total_reward = 0
        self.visited_cells = set()  # Limpiar células visitadas
        self.visited_cells.add(tuple(self.agent_pos))  # Marcar posición inicial
        
        # Crear observación inicial
        observation = self._get_observation()
        info = {
            "goal_type": self.goal_type,
            "goal_position": self.goal_pos
        }
        
        return observation, info

    def step(self, action):
        """Ejecuta una acción y retorna el nuevo estado"""
        self.steps += 1
        
        # Guardar posición anterior
        prev_pos = self.agent_pos.copy()
        
        # Ejecutar acción
        self._take_action(action)
        
        # Calcular distancia al objetivo
        prev_distance = self._manhattan_distance(prev_pos, self.goal_pos)
        current_distance = self._manhattan_distance(self.agent_pos, self.goal_pos)
        
        # ===================================
        # SISTEMA DE RECOMPENSAS MEJORADO
        # ===================================
        reward = 0
        terminated = False
        
        # 1. RECOMPENSA GRANDE: Llegar al objetivo
        if self.agent_pos == self.goal_pos:
            reward = +100.0
            terminated = True
            print(f"🎉 ¡Objetivo alcanzado en {self.steps} pasos!")
        
        else:
            # 2. REWARD SHAPING: Potencial basado en distancia
            # Usar distancia normalizada para que la señal sea más fuerte
            max_distance = GRID_SIZE * 2  # Máxima distancia Manhattan posible
            potential_prev = -prev_distance / max_distance
            potential_current = -current_distance / max_distance
            shaping_reward = potential_current - potential_prev
            reward += shaping_reward * 10  # Escalar para que sea significativo
            
            # 3. EXPLORACIÓN: Bonificación por visitar nuevas celdas
            cell_tuple = tuple(self.agent_pos)
            if cell_tuple not in self.visited_cells:
                reward += 0.5
                self.visited_cells.add(cell_tuple)
            
            # 4. EFICIENCIA: Pequeña penalización por paso
            reward -= 0.1
        
        # 5. TRUNCAR: Penalización fuerte si excede máximo de pasos
        truncated = self.steps >= self.max_steps
        if truncated:
            reward -= 10.0
            print(f"⏰ Tiempo agotado después de {self.steps} pasos")
        
        # Actualizar reward total
        self.total_reward += reward
        
        # Observación y información
        observation = self._get_observation()
        info = {
            "steps": self.steps,
            "distance_to_goal": current_distance,
            "total_reward": self.total_reward
        }
        
        return observation, reward, terminated, truncated, info

    def render(self):
        """Renderiza el entorno visualmente"""
        if self.render_mode is None:
            return None
            
        # Limpiar imagen
        self.img = np.zeros((TABLE_SIZE, TABLE_SIZE, 3), dtype='uint8')
        
        # Dibujar ciudad: calles y edificios
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.city_map[y, x] == 1:  # Calle
                    color = COLORS["street"]
                else:  # Edificio
                    color = COLORS["building"]
                
                cv2.rectangle(
                    self.img,
                    (x * CELL_SIZE, y * CELL_SIZE),
                    ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE),
                    color,
                    -1
                )
        
        # Dibujar grid
        for i in range(GRID_SIZE + 1):
            # Líneas verticales
            cv2.line(self.img, (i * CELL_SIZE, 0), (i * CELL_SIZE, TABLE_SIZE), (60, 60, 60), 1)
            # Líneas horizontales
            cv2.line(self.img, (0, i * CELL_SIZE), (TABLE_SIZE, i * CELL_SIZE), (60, 60, 60), 1)
        
        # Dibujar lugares
        for place_type, pos in self.places.items():
            color = COLORS[place_type]
            x, y = pos
            margin = max(2, CELL_SIZE // 6)
            cv2.rectangle(
                self.img,
                (x * CELL_SIZE + margin, y * CELL_SIZE + margin),
                ((x + 1) * CELL_SIZE - margin, (y + 1) * CELL_SIZE - margin),
                color,
                -1
            )
            # Añadir etiqueta (solo primera letra si la celda es muy pequeña)
            if CELL_SIZE >= 40:
                label = place_type[:4].upper()
                font_scale = 0.4
            else:
                label = place_type[0].upper()
                font_scale = 0.3
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_x = x * CELL_SIZE + (CELL_SIZE - text_size[0]) // 2
            text_y = y * CELL_SIZE + (CELL_SIZE + text_size[1]) // 2
            
            cv2.putText(
                self.img,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1
            )
        
        # Dibujar agente (turista) - círculo blanco
        agent_x, agent_y = self.agent_pos
        center = (agent_x * CELL_SIZE + CELL_SIZE // 2, agent_y * CELL_SIZE + CELL_SIZE // 2)
        radius = max(8, CELL_SIZE // 3)  # Radio adaptativo según tamaño de celda
        cv2.circle(self.img, center, radius, COLORS["agent"], -1)
        cv2.circle(self.img, center, radius, (100, 100, 100), 2)  # Borde
        
        # Añadir información en pantalla
        info_text = [
            f"Goal: {self.goal_type.upper()}",
            f"Steps: {self.steps}/{self.max_steps}",
            f"Reward: {self.total_reward:.1f}",
            f"Pos: ({agent_x}, {agent_y})"
        ]
        
        y_offset = 20
        for text in info_text:
            cv2.putText(
                self.img,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            y_offset += 20
        
        # Mostrar ventana solo si render_mode es 'human'
        if self.render_mode == 'human':
            cv2.imshow('TouristBot', self.img)
            cv2.waitKey(1)
        
        return self.img if self.render_mode == 'rgb_array' else None

    def close(self):
        """Cierra las ventanas de visualización"""
        cv2.destroyAllWindows()

    # ==================
    # MÉTODOS PRIVADOS
    # ==================
    
    def _generate_city_map(self):
        """
        Genera un mapa de ciudad con calles y edificios
        0 = edificio (bloqueado), 1 = calle (transitable)
        
        Patrón: cuadrícula con calles cada 4 celdas (estilo Manhattan)
        """
        city_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        
        # Crear calles horizontales cada 4 filas
        for y in range(0, GRID_SIZE, 4):
            city_map[y, :] = 1
        
        # Crear calles verticales cada 4 columnas
        for x in range(0, GRID_SIZE, 4):
            city_map[:, x] = 1
        
        # Los bordes también son calles para asegurar conectividad
        city_map[0, :] = 1
        city_map[-1, :] = 1
        city_map[:, 0] = 1
        city_map[:, -1] = 1
        
        return city_map
    
    def _find_random_street_position(self):
        """Encuentra una posición aleatoria que sea una calle"""
        # Obtener todas las posiciones que son calles
        street_positions = np.argwhere(self.city_map == 1)
        
        # Seleccionar una aleatoria
        if len(street_positions) > 0:
            idx = self.np_random.integers(0, len(street_positions))
            y, x = street_positions[idx]
            return [int(x), int(y)]
        else:
            # Fallback: esquina superior izquierda
            return [0, 0]
    
    def _find_random_building_position(self):
        """Encuentra una posición aleatoria que sea un edificio (no calle)"""
        # Obtener todas las posiciones que son edificios
        building_positions = np.argwhere(self.city_map == 0)
        
        # Seleccionar una aleatoria
        if len(building_positions) > 0:
            idx = self.np_random.integers(0, len(building_positions))
            y, x = building_positions[idx]
            return [int(x), int(y)]
        else:
            # Fallback: posición 1,1
            return [1, 1]
    
    def _is_adjacent_to_street(self, pos):
        """Verifica si una posición tiene al menos una calle adyacente"""
        x, y = pos
        # Verificar las 4 direcciones adyacentes
        adjacent_positions = [
            (x, y-1),  # arriba
            (x, y+1),  # abajo
            (x-1, y),  # izquierda
            (x+1, y)   # derecha
        ]
        
        for adj_x, adj_y in adjacent_positions:
            # Verificar límites
            if 0 <= adj_x < GRID_SIZE and 0 <= adj_y < GRID_SIZE:
                # Verificar si es una calle
                if self.city_map[adj_y, adj_x] == 1:
                    return True
        return False
    
    def _generate_places(self):
        """Genera restaurantes y museos en ubicaciones aleatorias DENTRO DE EDIFICIOS con acceso desde calles"""
        self.places = {}
        
        # Obtener todas las posiciones de edificios que tienen al menos una calle adyacente
        building_positions = np.argwhere(self.city_map == 0)
        accessible_buildings = []
        
        for pos in building_positions:
            y, x = pos
            pos_list = [int(x), int(y)]
            if self._is_adjacent_to_street(pos_list):
                accessible_buildings.append(pos_list)
        
        if len(accessible_buildings) < 2:
            # Fallback: usar edificios aunque no tengan acceso directo
            print("⚠️ Advertencia: Pocos edificios accesibles, usando posiciones alternativas")
            for pos in building_positions[:2]:
                y, x = pos
                accessible_buildings.append([int(x), int(y)])
        
        # Seleccionar posiciones aleatorias para los lugares (asegurar que sean diferentes)
        indices = self.np_random.choice(len(accessible_buildings), size=min(2, len(accessible_buildings)), replace=False)
        
        # Restaurant en edificio
        self.places["restaurant"] = accessible_buildings[indices[0]]
        
        # Museum en edificio (diferente al restaurant)
        if len(indices) > 1:
            self.places["museum"] = accessible_buildings[indices[1]]
        else:
            # Si solo hay un edificio accesible, poner el museum en otro edificio
            self.places["museum"] = self._find_random_building_position()

    def _take_action(self, action):
        """
        Ejecuta la acción de movimiento
        0: arriba, 1: abajo, 2: izquierda, 3: derecha
        Se puede mover a calles o a edificios con lugares de interés (restaurant/museum)
        """
        x, y = self.agent_pos
        new_x, new_y = x, y
        
        if action == 0:  # Arriba
            new_y = max(0, y - 1)
        elif action == 1:  # Abajo
            new_y = min(GRID_SIZE - 1, y + 1)
        elif action == 2:  # Izquierda
            new_x = max(0, x - 1)
        elif action == 3:  # Derecha
            new_x = min(GRID_SIZE - 1, x + 1)
        
        new_pos = [new_x, new_y]
        
        # Permitir movimiento si:
        # 1. Es una calle
        # 2. Es un edificio con un lugar de interés (restaurant o museum)
        if self.city_map[new_y, new_x] == 1:  # Es calle
            self.agent_pos = new_pos
        elif new_pos == self.places.get("restaurant") or new_pos == self.places.get("museum"):
            # Es un lugar de interés, permitir acceso
            self.agent_pos = new_pos
        # Si es un edificio normal (sin lugares de interés), no se mueve

    def _get_observation(self):
        """
        Crea el vector de observación según el modo configurado
        
        Modo completo: [agent_x, agent_y, goal_x, goal_y, goal_type_encoded]
        Modo parcial: [grid_5x5_flattened (25 valores), dist_x, dist_y, goal_type_encoded]
        """
        if not self.use_partial_obs:
            # Observación completa (legacy)
            goal_type_encoded = PLACE_TYPES.index(self.goal_type)
            observation = np.array([
                self.agent_pos[0],
                self.agent_pos[1],
                self.goal_pos[0],
                self.goal_pos[1],
                goal_type_encoded
            ], dtype=np.float32)
        else:
            # Vista parcial: grid centrado en el agente
            observation = self._get_partial_view()
        
        return observation
    
    def _get_partial_view(self):
        """
        Genera una vista parcial centrada en el agente (5x5 por defecto)
        
        Cada celda del grid codifica:
        - 0: edificio (bloqueado)
        - 1: calle (transitable)
        - 2: restaurant
        - 3: museum  
        - 4: agente (siempre en el centro)
        
        Retorna: array de tamaño (view_size*view_size + 3,)
        - Primeros view_size*view_size valores: grid aplanado
        - Últimos 3 valores: [distancia_x, distancia_y, tipo_objetivo]
        """
        half_view = self.view_size // 2
        grid_view = np.zeros((self.view_size, self.view_size), dtype=np.float32)
        
        agent_x, agent_y = self.agent_pos
        
        # Llenar el grid con lo que el agente ve
        for dy in range(-half_view, half_view + 1):
            for dx in range(-half_view, half_view + 1):
                # Posición absoluta en el grid
                abs_x = agent_x + dx
                abs_y = agent_y + dy
                
                # Índice en la vista local (centrada)
                local_x = dx + half_view
                local_y = dy + half_view
                
                # Si está fuera del grid, marcar como edificio (bloqueado)
                if abs_x < 0 or abs_x >= GRID_SIZE or abs_y < 0 or abs_y >= GRID_SIZE:
                    grid_view[local_y, local_x] = 0
                    continue
                
                # Verificar si hay un lugar en esta posición
                pos_check = [abs_x, abs_y]
                
                # Agente siempre en el centro
                if dx == 0 and dy == 0:
                    grid_view[local_y, local_x] = 4
                # Verificar restaurant
                elif pos_check == self.places.get("restaurant"):
                    grid_view[local_y, local_x] = 2
                # Verificar museum
                elif pos_check == self.places.get("museum"):
                    grid_view[local_y, local_x] = 3
                # Mostrar si es calle o edificio
                else:
                    grid_view[local_y, local_x] = float(self.city_map[abs_y, abs_x])
        
        # Aplanar el grid (5x5 -> 25 valores)
        grid_flat = grid_view.flatten()
        
        # Añadir información del objetivo
        dist_x = self.goal_pos[0] - self.agent_pos[0]  # Diferencia en x
        dist_y = self.goal_pos[1] - self.agent_pos[1]  # Diferencia en y
        goal_type_encoded = float(PLACE_TYPES.index(self.goal_type))
        
        # Combinar: grid + info objetivo
        observation = np.concatenate([
            grid_flat,
            np.array([dist_x, dist_y, goal_type_encoded], dtype=np.float32)
        ])
        
        return observation

    def _manhattan_distance(self, pos1, pos2):
        """Calcula la distancia de Manhattan entre dos posiciones"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# ==================
# FUNCIÓN DE PRUEBA
# ==================
def test_environment():
    """Función para probar el entorno con acciones aleatorias"""
    print("="*60)
    print("PROBANDO TOURISTBOT ENVIRONMENT")
    print("="*60)
    
    # Crear entorno
    env = TouristBotEnv(goal_type="restaurant")
    
    # Reset
    observation, info = env.reset()
    print(f"\n📍 Estado inicial:")
    print(f"   Agente en: {env.agent_pos}")
    print(f"   Objetivo: {info['goal_type']} en {info['goal_position']}")
    print(f"   Observación: {observation}")
    
    # Ejecutar algunos pasos
    print(f"\n🎮 Ejecutando acciones aleatorias...")
    
    for episode in range(3):
        observation, info = env.reset(options={"goal_type": random.choice(PLACE_TYPES)})
        print(f"\n--- Episodio {episode + 1} ---")
        print(f"Objetivo: {env.goal_type}")
        
        done = False
        while not done:
            env.render()
            action = env.action_space.sample()  # Acción aleatoria
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            time.sleep(0.1)  # Pausa para visualización
        
        print(f"Resultado: {'✅ Éxito' if terminated else '❌ Tiempo agotado'}")
        print(f"Reward total: {env.total_reward:.2f}")
        time.sleep(1)
    
    env.close()
    print("\n✅ Test completado")


if __name__ == "__main__":
    test_environment()
