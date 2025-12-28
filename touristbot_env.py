import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque
from transformers import pipeline

# 20 x 20 cell city
GRID_SIZE = 20
# 30 pixel cell size
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

    def __init__(self, goal_type="restaurant", render_mode=None, use_partial_obs=True, view_size=5, use_nlp=False):
        super(TouristBotEnv, self).__init__()
        
        self.render_mode = render_mode
        self.use_partial_obs = use_partial_obs
        self.view_size = view_size
        self.use_nlp = use_nlp
        
        # Initialize zero-shot classifier if NLP is enabled
        if self.use_nlp:
            print("Loading zero-shot classification model...")
            self.nlp_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            print("Model loaded successfully!")
        
        # Action space: 4 directional movements (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Observation space depends on mode
        if use_partial_obs:
            obs_size = view_size * view_size + 3
            self.observation_space = spaces.Box(
                low=-GRID_SIZE,
                high=GRID_SIZE,
                shape=(obs_size,),
                dtype=np.float32
            )
        else:
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
        
        # City structure
        self.city_map = self._generate_city_map()
        
        # Visualization
        self.img = np.zeros((TABLE_SIZE, TABLE_SIZE, 3), dtype='uint8')
        
        # NLP input window
        self.nlp_input = ""
        self.waiting_for_input = True if use_nlp else False
        self.text_input_active = False
        self.current_text_input = ""
        
        # Exit button state
        self.exit_button_hovered = False
        self.exit_requested = False
        
        # Navigation control - only moves when an instruction is given
        self.navigating = False  
        self.new_instruction_received = False  
        
        # Metrics
        self.steps = 0
        self.max_steps = 200
        self.total_reward = 0
        self.visited_cells = set()
        
        print("TouristBot Environment v2.1 initialized")
        print(f"   Grid: {GRID_SIZE}x{GRID_SIZE}")
        print(f"   View: {'Partial ' + str(view_size) + 'x' + str(view_size) if use_partial_obs else 'Full'}")
        print(f"   NLP: {'Enabled' if use_nlp else 'Disabled'}")
        print(f"   Initial goal: {goal_type}")

    # Function to determine goal with zero-shot classification
    def classify_intent(self, text):
        if not self.use_nlp:
            print("NLP is not enabled. Using default goal.")
            return self.goal_type
        
        print(f"\nClassifying intent: '{text}'")
        
        # Prepare candidate labels
        candidate_labels = list(PLACE_TYPES)
        
        # Run zero-shot classification
        result = self.nlp_classifier(
            text,
            candidate_labels,
            multi_label=False
        )
        
        # Get best prediction
        predicted_goal = result['labels'][0]
        confidence = result['scores'][0]
        
        print(f"Predicted goal: {predicted_goal} (confidence: {confidence:.2f})")
        print(f"All scores: {dict(zip(result['labels'], result['scores']))}")
        
        return predicted_goal

    # Process text and set new goal. Navigates from actual position
    def set_goal_from_text(self, text):
        predicted_goal = self.classify_intent(text)
        
        self.goal_type = predicted_goal
        
        if predicted_goal in self.places:
            self.goal_pos = self.places[predicted_goal].copy()
            print(f"Goal set to: {self.goal_type} at position {self.goal_pos}")
        else:
            print(f"Warning: {predicted_goal} not found in current map")
        
        # Verify that we are in a valid street
        agent_x, agent_y = self.agent_pos
        current_cell = self.city_map[agent_y, agent_x]
        
        # Verify if we are in a park
        is_in_park = False
        if self.park_area:
            for park_cell in self.park_area:
                if len(park_cell) >= 2 and agent_x == park_cell[0] and agent_y == park_cell[1]:
                    is_in_park = True
                    break
        
        # Verifiy if we are in another objective
        at_other_place = False
        for place_name, place_pos in self.places.items():
            if place_name != "park" and agent_x == place_pos[0] and agent_y == place_pos[1]:
                at_other_place = True
                break
        
        # DEBUG
        print(f"DEBUG: Agent at ({agent_x}, {agent_y}), cell={current_cell}, in_park={is_in_park}, at_place={at_other_place}")
        
        # Reposition if we aren't in a clean street
        on_clean_street = current_cell == 1 and not is_in_park and not at_other_place
        
        if not on_clean_street:
            print(f"Relocating agent to clean street...")
            
            # Verify if clean street
            def is_clean_street(check_x, check_y):
                if not (0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE):
                    return False
                
                if self.city_map[check_y, check_x] != 1:
                    return False
                
                if self.park_area:
                    for park_cell in self.park_area:
                        if len(park_cell) >= 2 and check_x == park_cell[0] and check_y == park_cell[1]:
                            return False
                
                return True
            
            # Find clean street close to the goal
            best_street = None
            best_distance = float('inf')
            
            max_search_radius = 8 if is_in_park else 5
            
            for radius in range(1, max_search_radius + 1):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if abs(dx) == radius or abs(dy) == radius:
                            check_x = agent_x + dx
                            check_y = agent_y + dy
                            
                            if is_clean_street(check_x, check_y):
                                dist_to_goal = self._manhattan_distance([check_x, check_y], self.goal_pos)
                                if dist_to_goal < best_distance:
                                    best_distance = dist_to_goal
                                    best_street = [check_x, check_y]
                
                if best_street is not None:
                    print(f"✓ Found clean street at radius {radius}: {best_street}")
                    break
            
            # else look through all the map
            if best_street is None:
                print(f"Expanding search to entire map...")
                street_positions = np.argwhere(self.city_map == 1)
                
                for street_pos in street_positions:
                    sy, sx = int(street_pos[0]), int(street_pos[1])
                    
                    if is_clean_street(sx, sy):
                        dist_to_goal = self._manhattan_distance([sx, sy], self.goal_pos)
                        if dist_to_goal < best_distance:
                            best_distance = dist_to_goal
                            best_street = [sx, sy]
            
            # Apply reposition
            if best_street:
                self.agent_pos = best_street
                print(f"Agent relocated to: {self.agent_pos} (distance to goal: {best_distance})")
            else:
                # Fallback
                print(f"Using random street fallback...")
                self.agent_pos = self._find_random_street_position()
                print(f"Agent at: {self.agent_pos}")
        else:
            print(f"Agent already on clean street at {self.agent_pos}")
        
        # Reset steps and state
        self.steps = 0
        self.total_reward = 0
        self.visited_cells = set()
        self.visited_cells.add(tuple(self.agent_pos))
        
        distance = self._manhattan_distance(self.agent_pos, self.goal_pos)
        print(f"Navigating from: {self.agent_pos} to {self.goal_pos} (distance: {distance})")
        
        return predicted_goal

    # Reset environment to initial. state
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # Check if using NLP input
        if options and "nlp_input" in options and self.use_nlp:
            text_input = options["nlp_input"]
            predicted_goal = self.classify_intent(text_input)
            self.goal_type = predicted_goal
        elif options and "goal_type" in options:
            self.goal_type = options["goal_type"]
        else:
            # RANDOMIZE GOAL: Choose random goal type for better generalization
            self.goal_type = random.choice(PLACE_TYPES)
        
        # Regenerate city map
        self.city_map = self._generate_city_map()
        
        # Initial agent position on a street
        self.agent_pos = self._find_random_street_position()
        
        # Generate places
        self._generate_places()
        
        # Set goal position
        if self.goal_type in self.places:
            self.goal_pos = self.places[self.goal_type].copy()
        else:
            print(f"Warning: Goal type '{self.goal_type}' not found. Using restaurant.")
            self.goal_type = "restaurant"
            self.goal_pos = self.places[self.goal_type].copy()
        
        # Reset metrics
        self.steps = 0
        self.total_reward = 0
        self.visited_cells = set()
        self.visited_cells.add(tuple(self.agent_pos))
        
        observation = self._get_observation()
        info = {
            "goal_type": self.goal_type,
            "goal_position": self.goal_pos
        }
        
        return observation, info
    
    # Execute action and return new state
    def step(self, action):
        self.steps += 1
        
        prev_pos = self.agent_pos.copy()
        self._take_action(action)
        
        prev_distance = self._manhattan_distance(prev_pos, self.goal_pos)
        current_distance = self._manhattan_distance(self.agent_pos, self.goal_pos)
        
        reward = 0
        terminated = False
        
        # Check if goal is reached
        goal_reached = False
        
        # For park: accept any cell in the park area
        if self.goal_type == "park":
            agent_x, agent_y = self.agent_pos
            if self.park_area:
                for park_cell in self.park_area:
                    if len(park_cell) >= 2 and agent_x == park_cell[0] and agent_y == park_cell[1]:
                        goal_reached = True
                        break
        else:
            # For other places: exact position match
            goal_reached = (self.agent_pos == self.goal_pos)
        
        if goal_reached:
            reward = +100.0
            terminated = True
            print(f"Goal '{self.goal_type}' reached in {self.steps} steps")
        else:
            max_distance = GRID_SIZE * 2
            potential_prev = -prev_distance / max_distance
            potential_current = -current_distance / max_distance
            shaping_reward = potential_current - potential_prev
            reward += shaping_reward * 10
            
            cell_tuple = tuple(self.agent_pos)
            if cell_tuple not in self.visited_cells:
                reward += 0.5
                self.visited_cells.add(cell_tuple)
            
            reward -= 0.1
        
        truncated = self.steps >= self.max_steps
        if truncated:
            reward -= 10.0
            print(f"Time out after {self.steps} steps")
        
        self.total_reward += reward
        
        observation = self._get_observation()
        info = {
            "steps": self.steps,
            "distance_to_goal": current_distance,
            "total_reward": self.total_reward
        }
        
        return observation, reward, terminated, truncated, info

    # Render env
    def render(self):
        if self.render_mode is None:
            return None
        
        # Create expanded image with space for text input at bottom
        input_height = 80 if self.use_nlp else 0
        total_height = TABLE_SIZE + input_height
        self.img = np.zeros((total_height, TABLE_SIZE, 3), dtype='uint8')
        
        # Draw city
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                pos = [x, y]
                
                if pos in self.park_area:
                    color = COLORS["park"]
                elif self.city_map[y, x] == 1:
                    color = COLORS["street"]
                else:
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
        
        # Draw places
        for place_type, pos in self.places.items():
            if place_type == "park":
                x, y = pos
                label = "P"
                font_scale = 1.0 if CELL_SIZE >= 40 else 0.8
                
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                text_x = x * CELL_SIZE + (CELL_SIZE - text_size[0]) // 2
                text_y = y * CELL_SIZE + (CELL_SIZE + text_size[1]) // 2
                
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
                
                if CELL_SIZE >= 40:
                    label_map = {
                        "restaurant": "R",
                        "museum": "M",
                        "shop": "S",
                        "cinema": "C"
                    }
                    label = label_map.get(place_type, place_type[0].upper())
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
        
        # Draw agent
        agent_x, agent_y = self.agent_pos
        center = (agent_x * CELL_SIZE + CELL_SIZE // 2, agent_y * CELL_SIZE + CELL_SIZE // 2)
        radius = max(8, CELL_SIZE // 3)
        cv2.circle(self.img, center, radius, COLORS["agent"], -1)
        cv2.circle(self.img, center, radius, (100, 100, 100), 2)
        
        # Add info text
        info_text = [
            f"Goal: {self.goal_type.upper()}",
            f"Steps: {self.steps}/{self.max_steps}",
            f"Reward: {self.total_reward:.1f}",
            f"Pos: ({agent_x}, {agent_y})"
        ]
        
        if self.use_nlp:
            if self.navigating:
                status = "Status: NAVIGATING"
                status_color = (100, 255, 100) 
            else:
                status = "Status: WAITING"
                status_color = (255, 200, 100) 
            info_text.append(status)
        
        y_offset = 20
        for i, text in enumerate(info_text):
            if self.use_nlp and "Status:" in text:
                color = status_color
            else:
                color = (255, 255, 255)
            
            cv2.putText(
                self.img,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color, 
                1
            )
            y_offset += 20
        
        # Draw text input box if NLP is enabled
        if self.use_nlp:
            input_box_y = TABLE_SIZE + 5
            
            # Draw background box
            cv2.rectangle(
                self.img,
                (5, input_box_y),
                (TABLE_SIZE - 5, TABLE_SIZE + 75),
                (40, 40, 40),
                -1
            )
            cv2.rectangle(
                self.img,
                (5, input_box_y),
                (TABLE_SIZE - 5, TABLE_SIZE + 75),
                (100, 100, 100),
                2
            )
            
            if self.text_input_active:
                prompt_text = "Type your instruction (press Enter):"
            elif self.navigating:
                prompt_text = "Navigating... Press 'T' for new instruction"
            else:
                prompt_text = "Press 'T' to give an instruction"
            
            cv2.putText(
                self.img,
                prompt_text,
                (10, input_box_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1
            )
            
            # Draw input text
            display_text = self.current_text_input if self.text_input_active else self.nlp_input
            if self.text_input_active:
                display_text += "_"  # Cursor
            
            # Truncate if too long
            max_chars = 55
            if len(display_text) > max_chars:
                display_text = display_text[:max_chars] + "..."
            
            cv2.putText(
                self.img,
                display_text,
                (10, input_box_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255) if self.text_input_active else (150, 150, 150),
                1
            )
            
            # Show instructions
            cv2.putText(
                self.img,
                "ESC: cancel | ENTER: confirm | BACKSPACE: delete",
                (10, input_box_y + 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (150, 150, 150),
                1
            )
        
        # Draw Exit button (top-right corner)
        button_width = 80
        button_height = 30
        button_x = TABLE_SIZE - button_width - 10
        button_y = 10
        
        # Button color changes on hover
        button_color = (50, 50, 200) if self.exit_button_hovered else (80, 80, 80)
        text_color = (255, 255, 255)
        
        # Draw button background
        cv2.rectangle(
            self.img,
            (button_x, button_y),
            (button_x + button_width, button_y + button_height),
            button_color,
            -1
        )
        # Draw button border
        cv2.rectangle(
            self.img,
            (button_x, button_y),
            (button_x + button_width, button_y + button_height),
            (150, 150, 150),
            2
        )
        # Draw button text
        cv2.putText(
            self.img,
            "EXIT",
            (button_x + 20, button_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1
        )
        
        if self.render_mode == 'human':
            cv2.imshow('TouristBot Environment', self.img)
            self._handle_keyboard_input()
            if self.use_nlp:
                self._handle_mouse_input()
        
        return self.img if self.render_mode == 'rgb_array' else None

    def _generate_city_map(self):
        city_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        
        for y in range(0, GRID_SIZE, 4):
            city_map[y, :] = 1
        
        for x in range(0, GRID_SIZE, 4):
            city_map[:, x] = 1
        
        city_map[0, :] = 1
        city_map[-1, :] = 1
        city_map[:, 0] = 1
        city_map[:, -1] = 1
        
        return city_map
    
    def _find_random_street_position(self):
        street_positions = np.argwhere(self.city_map == 1)
        
        if len(street_positions) > 0:
            idx = self.np_random.integers(0, len(street_positions))
            y, x = street_positions[idx]
            return [int(x), int(y)]
        else:
            return [0, 0]
    
    def _find_random_building_position(self):
        building_positions = np.argwhere(self.city_map == 0)
        
        if len(building_positions) > 0:
            idx = self.np_random.integers(0, len(building_positions))
            y, x = building_positions[idx]
            return [int(x), int(y)]
        else:
            return [1, 1]
    
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
    
    def _generate_places(self):
        self.places = {}
        
        building_positions = np.argwhere(self.city_map == 0)
        accessible_buildings = []
        
        for pos in building_positions:
            y, x = pos
            pos_list = [int(x), int(y)]
            if self._is_adjacent_to_street(pos_list):
                accessible_buildings.append(pos_list)
        
        if len(accessible_buildings) < 4:
            for pos in building_positions[:4]:
                y, x = pos
                accessible_buildings.append([int(x), int(y)])
        
        park_placed = False
        for attempt in range(50):
            start_x = self.np_random.integers(1, GRID_SIZE - 4)
            start_y = self.np_random.integers(1, GRID_SIZE - 4)
            
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
                    park_center = [start_x + 1, start_y + 1]
                    self.places["park"] = park_center
                    
                    self.park_area = []
                    for dy in range(3):
                        for dx in range(3):
                            self.park_area.append([start_x + dx, start_y + dy])
                    
                    accessible_buildings = [pos for pos in accessible_buildings 
                                           if pos not in self.park_area]
                    
                    park_placed = True
                    break
        
        if not park_placed:
            self.places["park"] = accessible_buildings[0] if accessible_buildings else [5, 5]
            self.park_area = [self.places["park"]]
            accessible_buildings = accessible_buildings[1:] if len(accessible_buildings) > 1 else []
        
        remaining_places = ["restaurant", "museum", "shop", "cinema"]
        num_places_needed = min(len(remaining_places), len(accessible_buildings))
        
        if accessible_buildings:
            indices = self.np_random.choice(
                len(accessible_buildings), 
                size=num_places_needed, 
                replace=False
            )
            
            for i, place_type in enumerate(remaining_places[:num_places_needed]):
                self.places[place_type] = accessible_buildings[indices[i]]
        
        if "restaurant" not in self.places:
            self.places["restaurant"] = self._find_random_building_position()
        if "museum" not in self.places:
            self.places["museum"] = self._find_random_building_position()

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
        
        if self.city_map[new_y, new_x] == 1:
            self.agent_pos = new_pos
        elif new_pos in self.park_area:
            self.agent_pos = new_pos
        elif (new_pos == self.places.get("restaurant") or 
              new_pos == self.places.get("museum") or
              new_pos == self.places.get("shop") or
              new_pos == self.places.get("cinema")):
            self.agent_pos = new_pos

    def _get_observation(self):
        if not self.use_partial_obs:
            goal_type_encoded = PLACE_TYPES.index(self.goal_type)
            observation = np.array([
                self.agent_pos[0],
                self.agent_pos[1],
                self.goal_pos[0],
                self.goal_pos[1],
                goal_type_encoded
            ], dtype=np.float32)
        else:
            observation = self._get_partial_view()
        
        return observation
    
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
                
                if abs_x < 0 or abs_x >= GRID_SIZE or abs_y < 0 or abs_y >= GRID_SIZE:
                    grid_view[local_y, local_x] = 0
                    continue
                
                pos_check = [abs_x, abs_y]
                
                if dx == 0 and dy == 0:
                    grid_view[local_y, local_x] = 4
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

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _handle_keyboard_input(self):
        # Wait longer if text input is active to capture keystrokes better
        wait_time = 50 if self.text_input_active else 1
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == 255:  # No key pressed
            return
        
        # ESC key to exit (only when not in text input mode)
        if key == 27 and not self.text_input_active:  # ESC key
            print("\n[Exit requested by ESC key]")
            self.exit_requested = True
            return
        
        # 'T' key to activate text input
        if key == ord('t') or key == ord('T'):
            if not self.text_input_active:
                self.text_input_active = True
                self.current_text_input = ""
                print("\n[Text input activated - type your destination]")
                return  # Don't process the 't' key as input
        
        # If text input is active, handle typing
        if self.text_input_active:
            if key == 13 or key == 10:  # Enter key (13 on Windows, 10 on Mac/Linux)
                if self.current_text_input.strip():
                    print(f"\n[Processing: '{self.current_text_input}']")
                    self.nlp_input = self.current_text_input
                    predicted_goal = self.set_goal_from_text(self.current_text_input)
                    print(f"[New goal set: {predicted_goal}]")
                    self.text_input_active = False
                    self.current_text_input = ""
                    # Activar navegación cuando hay nueva instrucción
                    self.new_instruction_received = True
                    self.navigating = True
                else:
                    print("\n[Empty input - cancelled]")
                    self.text_input_active = False
            
            elif key == 27:  # ESC key
                print("\n[Text input cancelled]")
                self.text_input_active = False
                self.current_text_input = ""
            
            elif key == 8 or key == 127:  # Backspace (8 on Windows, 127 on Mac)
                if len(self.current_text_input) > 0:
                    self.current_text_input = self.current_text_input[:-1]
            
            elif 32 <= key <= 126:  # Printable ASCII characters
                self.current_text_input += chr(key)
                print(f"Input: {self.current_text_input}")  # Debug: mostrar lo que se escribe
    
    def _handle_mouse_input(self):
        # Mouse callback for exit button
        def mouse_callback(event, x, y, flags, param):
            button_width = 80
            button_height = 30
            button_x = TABLE_SIZE - button_width - 10
            button_y = 10
            
            # Check if mouse is over exit button
            if button_x <= x <= button_x + button_width and button_y <= y <= button_y + button_height:
                self.exit_button_hovered = True
                
                # Check for click
                if event == cv2.EVENT_LBUTTONDOWN:
                    print("\n[Exit button clicked]")
                    self.exit_requested = True
            else:
                self.exit_button_hovered = False
        
        cv2.setMouseCallback('TouristBot Environment', mouse_callback)

    def close(self):
        cv2.destroyAllWindows()
