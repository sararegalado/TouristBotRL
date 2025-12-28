import argparse
import os
from pathlib import Path
from stable_baselines3 import PPO, A2C, DQN
from touristbot_env import TouristBotEnv
import time
import glob

# Main application
class TouristBotApp:
    
    def __init__(self, model_path):
        self.model_path = self._find_model(model_path)
        self.model = None
        self.env = None
        
        print("="*70)
        print(" " * 25 + "TOURISTBOT APPLICATION")
        print("="*70)
    
    # Find model
    def _find_model(self, model_path):
        if model_path and os.path.exists(model_path):
            return model_path
        
        # If the path doesn't exist, use basic PPO
        pattern = "runs/ppo_basic*/models/final_model.zip"
        matches = glob.glob(pattern)
        
        if matches:
            # Return the first match found
            return matches[0]
        else:
            raise FileNotFoundError(f"No model found matching pattern: {pattern}")
    
    
    def _detect_algorithm(self, model_path):
        model_path_lower = model_path.lower()
        if 'a2c' in model_path_lower:
            return A2C
        elif 'dqn' in model_path_lower:
            return DQN
        else:
            return PPO 
        
    
    def load_model(self):
        print(f"\nLoading model from: {self.model_path}")
        try:
            # Detect algorithm
            algorithm_class = self._detect_algorithm(self.model_path)
            algorithm_name = algorithm_class.__name__
            
            print(f"Detected algorithm: {algorithm_name}")
            
            self.model = algorithm_class.load(self.model_path)
            print(f"✓ Model {algorithm_name} loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    

    def create_env(self):
        self.env = TouristBotEnv(
            goal_type="restaurant",  # Initial, will be changed with NLP
            use_nlp=True,
            render_mode="human",
            use_partial_obs=True,
            view_size=5
        )
        print("Environment created")
    
    def run_interactive(self):
        if self.model is None:
            self.load_model()
        
        if self.env is None:
            self.create_env()
        
        
        # Initial reset
        observation, info = self.env.reset()
        
        running = True
        step_count = 0
        episode_count = 0
        
        try:
            while running:
                self.env.render()
                
                # Verify if exit requested
                if self.env.exit_requested:
                    print("\nExiting application...")
                    running = False
                    continue
                
                # Verify if new instruction
                if self.env.new_instruction_received:
                    print(f"\nNavigating to: {self.env.goal_type.upper()}")
                    self.env.new_instruction_received = False
                    self.env.navigating = True
                    step_count = 0

                    # Reset episode state without regenerating map
                    self.env.steps = 0
                    self.env.total_reward = 0
                    self.env.visited_cells = set()
                    self.env.visited_cells.add(tuple(self.env.agent_pos))

                    # Update observation
                    observation = self.env._get_observation()
                
                # Only navigate if navigation mode and no writing
                if self.env.navigating and not self.env.text_input_active:
                    # Predict action
                    action, _states = self.model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = self.env.step(action)
                    step_count += 1
                    
                    if terminated:
                        episode_count += 1
                        print(f"\n✓ Goal reached in {step_count} steps! (Episode #{episode_count})")
                        print("Press 'T' to give a new instruction, or ESC/EXIT to exit")
                        
                        self.env.navigating = False
                        step_count = 0
                    
                    elif truncated:
                        episode_count += 1
                        print(f"\nTime exhausted after {step_count} steps (Episode #{episode_count})")
                        print("Press 'T' to give a new instruction")
                        
                        # Stop navigation - wait for new instruction
                        self.env.navigating = False
                        step_count = 0
                    
                    time.sleep(0.05)  # Slow down for visualization
                else:
                    # Small wait when idle
                    time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            print("\n" + "="*70)
            print(f"FINAL STATISTICS:")
            print(f"  • Episodes completed: {episode_count}")
            print("="*70)
            self.env.close()
            print("\nEnvironment closed")


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(
        description="TouristBot - Navigation with RL and NLP"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to PPO model (default: uses ppo basic model)"
    )
    
    args = parser.parse_args()
    
    # Create application
    app = TouristBotApp(model_path=args.model)
    app.run_interactive()

if __name__ == "__main__":
    main()
