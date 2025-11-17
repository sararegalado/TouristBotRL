import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

# Configuration
GRID_SIZE = 20
CELL_SIZE = 30
TABLE_SIZE = GRID_SIZE * CELL_SIZE

# RGB colors for visualization
COLORS = {
    "background": (0, 0, 0),
    "agent": (255, 255, 255),
    "restaurant": (0, 165, 255),
    "museum": (0, 0, 255),
    "shop": (255, 255, 0),
    "cinema": (255, 0, 255),
    "park": (0, 255, 0),
    "visited": (50, 50, 50),
    "street": (80, 80, 80),
    "building": (40, 40, 40),
}

# Place types available in the city
PLACE_TYPES = ["restaurant", "museum", "shop", "cinema", "park"]

class TouristBotEnv(gym.Env):
    metadata = {
        'render_modes': ['human'],
        'render_fps': 10
    }

    def __init__(self, goal_type="restaurant", render_mode=None, use_partial_obs=True, view_size=5):
        super(TouristBotEnv, self).__init__()
        
        self.render_mode = render_mode
        self.use_partial_obs = use_partial_obs
        self.view_size = view_size
        
        # Action space: 4 directional movements (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Observation space depends on mode
        if use_partial_obs:
            # Partial view: grid + goal info (encoding: 0=building, 1=street, 2-6=places, 7=agent)
            obs_size = view_size * view_size + 3
            self.observation_space = spaces.Box(
                low=-GRID_SIZE,
                high=GRID_SIZE,
                shape=(obs_size,),
                dtype=np.float32
            )
        else:
            # Full observation: [agent_x, agent_y, goal_x, goal_y, goal_type]
            self.observation_space = spaces.Box(
                low=0,
                high=GRID_SIZE,
                shape=(5,),
                dtype=np.float32
            )
        
        # Environment state
        self.agent_pos = [0, 0]
        self.goal_type = goal_type
        self.goal_pos = [0, 0]
        self.places = {}
        self.park_area = []
        
        # City structure (0=building blocked, 1=street walkable)
        self.city_map = self._generate_city_map()
        
        # Visualization
        self.img = np.zeros((TABLE_SIZE, TABLE_SIZE, 3), dtype='uint8')
        
        # Metrics
        self.steps = 0
        self.max_steps = 200
        self.total_reward = 0
        self.visited_cells = set()
        
        print("TouristBot Environment v2.0 initialized")
        print(f"   Grid: {GRID_SIZE}x{GRID_SIZE}")
        print(f"   View: {'Partial ' + str(view_size) + 'x' + str(view_size) if use_partial_obs else 'Full'}")
        print(f"   Structure: City with streets")
        print(f"   Initial goal: {goal_type}")

    def reset(self, *, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Allow changing goal from options
        if options and "goal_type" in options:
            self.goal_type = options["goal_type"]
        
        # Regenerate city map with new seed
        self.city_map = self._generate_city_map()
        
        # Initial agent position on a street
        self.agent_pos = self._find_random_street_position()
        
        # Generate places randomly in buildings
        self._generate_places()
        
        # Set goal position
        self.goal_pos = self.places[self.goal_type].copy()
        
        # Reset metrics
        self.steps = 0
        self.total_reward = 0
        self.visited_cells = set()
        self.visited_cells.add(tuple(self.agent_pos))
        
        # Create initial observation
        observation = self._get_observation()
        info = {
            "goal_type": self.goal_type,
            "goal_position": self.goal_pos
        }
        
        return observation, info

    def step(self, action):
        """Execute action and return new state"""
        self.steps += 1
        
        prev_pos = self.agent_pos.copy()
        self._take_action(action)
        
        # Calculate distance to goal
        prev_distance = self._manhattan_distance(prev_pos, self.goal_pos)
        current_distance = self._manhattan_distance(self.agent_pos, self.goal_pos)
        
        # Reward system
        reward = 0
        terminated = False
        
        # Large reward: reach goal
        if self.agent_pos == self.goal_pos:
            reward = +100.0
            terminated = True
            print(f"Goal reached in {self.steps} steps!")
        
        else:
            # Reward shaping: distance-based potential
            max_distance = GRID_SIZE * 2
            potential_prev = -prev_distance / max_distance
            potential_current = -current_distance / max_distance
            shaping_reward = potential_current - potential_prev
            reward += shaping_reward * 10
            
            # Exploration bonus: visit new cells
            cell_tuple = tuple(self.agent_pos)
            if cell_tuple not in self.visited_cells:
                reward += 0.5
                self.visited_cells.add(cell_tuple)
            
            # Efficiency penalty: small step cost
            reward -= 0.1
        
        # Truncate: penalty if max steps exceeded
        truncated = self.steps >= self.max_steps
        if truncated:
            reward -= 10.0
            print(f"Time out after {self.steps} steps")
        
        self.total_reward += reward
        
        # Get observation and info
        observation = self._get_observation()
        info = {
            "steps": self.steps,
            "distance_to_goal": current_distance,
            "total_reward": self.total_reward
        }
        
        return observation, reward, terminated, truncated, info

    def render(self):
        """Render environment visually"""
        if self.render_mode is None:
            return None
            
        # Clear image
        self.img = np.zeros((TABLE_SIZE, TABLE_SIZE, 3), dtype='uint8')
        
        # Dibujar ciudad: calles, edificios y parque
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                pos = [x, y]
                
                # Verificar si es parte del parque
                if pos in self.park_area:
                    color = COLORS["park"]
                elif self.city_map[y, x] == 1:  # Calle
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
        
        # Draw grid lines
        for i in range(GRID_SIZE + 1):
            cv2.line(self.img, (i * CELL_SIZE, 0), (i * CELL_SIZE, TABLE_SIZE), (60, 60, 60), 1)
            cv2.line(self.img, (0, i * CELL_SIZE), (TABLE_SIZE, i * CELL_SIZE), (60, 60, 60), 1)
        
        # Draw places of interest
        for place_type, pos in self.places.items():
            # Park is already drawn as green area, just add icon in center
            if place_type == "park":
                x, y = pos
                label = "P"
                font_scale = 1.0 if CELL_SIZE >= 40 else 0.8
                
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                text_x = x * CELL_SIZE + (CELL_SIZE - text_size[0]) // 2
                text_y = y * CELL_SIZE + (CELL_SIZE + text_size[1]) // 2
                
                # White background for contrast
                cv2.circle(
                    self.img,
                    (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2),
                    CELL_SIZE // 3,
                    (255, 255, 255),
                    -1
                )
                
                cv2.putText(
                    self.img,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 150, 0),
                    2
                )
            else:
                # Other places: draw as colored rectangles
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
                # Add label
                if CELL_SIZE >= 40:
                    if place_type == "restaurant":
                        label = "R"
                    elif place_type == "museum":
                        label = "M"
                    elif place_type == "shop":
                        label = "S"
                    elif place_type == "cinema":
                        label = "C"
                    else:
                        label = place_type[0].upper()
                    font_scale = 0.5
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
        
        # Draw agent as white circle
        agent_x, agent_y = self.agent_pos
        center = (agent_x * CELL_SIZE + CELL_SIZE // 2, agent_y * CELL_SIZE + CELL_SIZE // 2)
        radius = max(8, CELL_SIZE // 3)
        cv2.circle(self.img, center, radius, COLORS["agent"], -1)
        cv2.circle(self.img, center, radius, (100, 100, 100), 2)
        
        # Add on-screen info
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
        
        # Show window only if render_mode is 'human'
        if self.render_mode == 'human':
            cv2.imshow('TouristBot', self.img)
            cv2.waitKey(1)
        
        return self.img if self.render_mode == 'rgb_array' else None

    def close(self):
        """Close visualization windows"""
        cv2.destroyAllWindows()

    # Private methods
    
    def _generate_city_map(self):
        """Generate city map with streets and buildings (0=building, 1=street)"""
        city_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        
        # Create horizontal streets every 4 rows
        for y in range(0, GRID_SIZE, 4):
            city_map[y, :] = 1
        
        # Create vertical streets every 4 columns
        for x in range(0, GRID_SIZE, 4):
            city_map[:, x] = 1
        
        # Borders are also streets to ensure connectivity
        city_map[0, :] = 1
        city_map[-1, :] = 1
        city_map[:, 0] = 1
        city_map[:, -1] = 1
        
        return city_map
    
    # Find random position on a street
    def _find_random_street_position(self):
        street_positions = np.argwhere(self.city_map == 1)
        
        if len(street_positions) > 0:
            idx = self.np_random.integers(0, len(street_positions))
            y, x = street_positions[idx]
            return [int(x), int(y)]
        else:
            return [0, 0]
    
    # Find random position in a building (not on street)
    def _find_random_building_position(self):
        building_positions = np.argwhere(self.city_map == 0)
        
        if len(building_positions) > 0:
            idx = self.np_random.integers(0, len(building_positions))
            y, x = building_positions[idx]
            return [int(x), int(y)]
        else:
            return [1, 1]
    
    # Check if position has at leas one adjacent street
    def _is_adjacent_to_street(self, pos):
        x, y = pos
        adjacent_positions = [
            (x, y-1),
            (x, y+1),
            (x-1, y),
            (x+1, y)
        ]
        
        for adj_x, adj_y in adjacent_positions:
            if 0 <= adj_x < GRID_SIZE and 0 <= adj_y < GRID_SIZE:
                if self.city_map[adj_y, adj_x] == 1:
                    return True
        return False
    
    # Generate touristic places in random positions inside buildings with access to streets
    def _generate_places(self):
        self.places = {}
        
        # Get positions of all buildings that have at least one adjacent street
        building_positions = np.argwhere(self.city_map == 0)
        accessible_buildings = []
        
        for pos in building_positions:
            y, x = pos
            pos_list = [int(x), int(y)]
            if self._is_adjacent_to_street(pos_list):
                accessible_buildings.append(pos_list)
        
        if len(accessible_buildings) < 4:
            print("Warning: Few accessible buildings, using alternative positions")
            for pos in building_positions[:4]:
                y, x = pos
                accessible_buildings.append([int(x), int(y)])
        
        # 1. Park: find 3x3 building block for park
        park_placed = False
        for attempt in range(50):
            start_x = self.np_random.integers(1, GRID_SIZE - 4)
            start_y = self.np_random.integers(1, GRID_SIZE - 4)
            
            # Check if valid 3x3 building block
            is_valid_block = True
            for dy in range(3):
                for dx in range(3):
                    check_x = start_x + dx
                    check_y = start_y + dy
                    if self.city_map[check_y, check_x] != 0:
                        is_valid_block = False
                        break
                if not is_valid_block:
                    break
            
            if is_valid_block:
                # Check for at least one adjacent street on perimeter
                has_street_access = False
                for dy in range(-1, 4):
                    for dx in range(-1, 4):
                        if dy == -1 or dy == 3 or dx == -1 or dx == 3:
                            check_x = start_x + dx
                            check_y = start_y + dy
                            if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                                if self.city_map[check_y, check_x] == 1:
                                    has_street_access = True
                                    break
                    if has_street_access:
                        break
                
                if has_street_access:
                    # Place park at center of 3x3 block
                    park_center = [start_x + 1, start_y + 1]
                    self.places["park"] = park_center
                    
                    # Mark all park cells
                    self.park_area = []
                    for dy in range(3):
                        for dx in range(3):
                            self.park_area.append([start_x + dx, start_y + dy])
                    
                    # Remove park positions from accessible buildings
                    accessible_buildings = [pos for pos in accessible_buildings 
                                           if pos not in self.park_area]
                    
                    park_placed = True
                    break
        
        if not park_placed:
            print("Could not place 3x3 park, using single building")
            self.places["park"] = accessible_buildings[0] if accessible_buildings else [5, 5]
            self.park_area = [self.places["park"]]
            accessible_buildings = accessible_buildings[1:] if len(accessible_buildings) > 1 else []
        
        # 2. Other places: select random positions
        remaining_places = ["restaurant", "museum", "shop", "cinema"]
        num_places_needed = min(len(remaining_places), len(accessible_buildings))
        
        if num_places_needed < len(remaining_places):
            print(f"Warning: Only {num_places_needed} buildings available")
        
        # Select random buildings
        if accessible_buildings:
            indices = self.np_random.choice(
                len(accessible_buildings), 
                size=num_places_needed, 
                replace=False
            )
            
            for i, place_type in enumerate(remaining_places[:num_places_needed]):
                self.places[place_type] = accessible_buildings[indices[i]]
        
        # Ensure at least restaurant and museum exist for compatibility
        if "restaurant" not in self.places:
            self.places["restaurant"] = self._find_random_building_position()
        if "museum" not in self.places:
            self.places["museum"] = self._find_random_building_position()

    # Execute action (0=up, 1=down, 2=left, 3=right)
    def _take_action(self, action):

        x, y = self.agent_pos
        new_x, new_y = x, y
        
        if action == 0:
            new_y = max(0, y - 1)
        elif action == 1:
            new_y = min(GRID_SIZE - 1, y + 1)
        elif action == 2:
            new_x = max(0, x - 1)
        elif action == 3:
            new_x = min(GRID_SIZE - 1, x + 1)
        
        new_pos = [new_x, new_y]
        
        # Allow movement if: street, park, or place of interest
        if self.city_map[new_y, new_x] == 1:
            self.agent_pos = new_pos
        elif new_pos in self.park_area:
            self.agent_pos = new_pos
        elif (new_pos == self.places.get("restaurant") or 
              new_pos == self.places.get("museum") or
              new_pos == self.places.get("shop") or
              new_pos == self.places.get("cinema")):
            self.agent_pos = new_pos

    # Create observation vector
    def _get_observation(self):
        if not self.use_partial_obs:
            # Full observation
            goal_type_encoded = PLACE_TYPES.index(self.goal_type)
            observation = np.array([
                self.agent_pos[0],
                self.agent_pos[1],
                self.goal_pos[0],
                self.goal_pos[1],
                goal_type_encoded
            ], dtype=np.float32)
        else:
            # Partial view: grid centered on agent
            observation = self._get_partial_view()
        
        return observation
    
    # Generate partial view centered on agent
    def _get_partial_view(self):
        half_view = self.view_size // 2
        grid_view = np.zeros((self.view_size, self.view_size), dtype=np.float32)
        
        agent_x, agent_y = self.agent_pos
        
        for dy in range(-half_view, half_view + 1):
            for dx in range(-half_view, half_view + 1):
                abs_x = agent_x + dx
                abs_y = agent_y + dy
                
                local_x = dx + half_view
                local_y = dy + half_view
                
                # Out of bounds = building
                if abs_x < 0 or abs_x >= GRID_SIZE or abs_y < 0 or abs_y >= GRID_SIZE:
                    grid_view[local_y, local_x] = 0
                    continue
                
                pos_check = [abs_x, abs_y]
                
                # Agent always at center
                if dx == 0 and dy == 0:
                    grid_view[local_y, local_x] = 4
                # Check for places
                elif pos_check == self.places.get("restaurant"):
                    grid_view[local_y, local_x] = 2
                elif pos_check == self.places.get("museum"):
                    grid_view[local_y, local_x] = 3
                elif pos_check == self.places.get("shop"):
                    grid_view[local_y, local_x] = 5
                elif pos_check == self.places.get("cinema"):
                    grid_view[local_y, local_x] = 6
                elif pos_check in self.park_area:
                    grid_view[local_y, local_x] = 7
                # Street or building
                else:
                    grid_view[local_y, local_x] = float(self.city_map[abs_y, abs_x])
        
        grid_flat = grid_view.flatten()
        
        dist_x = self.goal_pos[0] - self.agent_pos[0]
        dist_y = self.goal_pos[1] - self.agent_pos[1]
        goal_type_encoded = float(PLACE_TYPES.index(self.goal_type))
        observation = np.concatenate([
            grid_flat,
            np.array([dist_x, dist_y, goal_type_encoded], dtype=np.float32)
        ])
        
        return observation

    # Calculate manhattan distance between 2 positions
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# Test function
def test_environment():
    print("="*60)
    print("TESTING TOURISTBOT ENVIRONMENT")
    print("="*60)
    
    env = TouristBotEnv(goal_type="restaurant")
    
    observation, info = env.reset()
    print(f"\nInitial state:")
    print(f"   Agent at: {env.agent_pos}")
    print(f"   Goal: {info['goal_type']} at {info['goal_position']}")
    print(f"   Observation: {observation}")
    
    print(f"\nRunning random actions...")
    
    for episode in range(3):
        observation, info = env.reset(options={"goal_type": random.choice(PLACE_TYPES)})
        print(f"\n--- Episode {episode + 1} ---")
        print(f"Goal: {env.goal_type}")
        
        done = False
        while not done:
            env.render()
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            time.sleep(0.1)
        
        print(f"Result: {'Success' if terminated else 'Timeout'}")
        print(f"Total reward: {env.total_reward:.2f}")
        time.sleep(1)
    
    env.close()
    print("\nTest completed")


if __name__ == "__main__":
    test_environment()
