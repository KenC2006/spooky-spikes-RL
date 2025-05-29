import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
import numpy as np
import math
import random
from typing import Optional, Any, Dict


class PlatformerPyEnvironment(py_environment.PyEnvironment):
    """
    3D Platformer TensorFlow-Agents Environment
    
    The agent controls a cube that must avoid incoming beams by jumping or ducking.
    Uses TF-Agents framework for compatibility with TensorFlow RL algorithms.
    """
    
    def __init__(self, max_steps: int = 10000, render_mode: str = 'none'):
        super().__init__()
        
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.current_step = 0
        self._last_action_change = 0  # Track when actions change
        self._last_action = 0  # Track last action
        
        # Game constants
        self.PLATFORM_WIDTH = 10
        self.PLATFORM_DEPTH = 10
        
        # Player constants
        self.START_X = 3
        self.START_Z = 0.0
        self.START_Y = 1.5
        self.CUBE_WIDTH = 1.5
        self.CUBE_DEPTH = 1.5
        self.CUBE_HEIGHT_NORMAL = 3.0
        self.CUBE_HEIGHT_DUCK = 1.5
        
        # Physics
        self.gravity = -0.03
        self.jump_force = 0.5
        
        # Beam mechanics
        self.INITIAL_SPAWN_INTERVAL = 90
        self.min_spawn_interval = 30
        self.spawn_acceleration_rate = 0.02
        self.MAX_SPAWN_ACCELERATION = 0.4
        
        # Speed mechanics
        self.INITIAL_BAR_SPEED = 0.35
        self.MAX_BAR_SPEED = 0.6
        self.bar_speed_increase = 0.00005
        
        # Define action and observation specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action'
        )
        
        # Observation: [
        #   player_y, velocity_y, is_jumping, is_ducking,
        #   beam1_distance, beam1_is_high,
        #   beam2_distance, beam2_is_high,
        #   current_speed, beam1_danger,
        #   is_player_small, beam_in_range,
        #   beam1_critical, beam2_critical,
        #   time_in_state
        # ]
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(15,), dtype=np.float32, minimum=-50.0, maximum=50.0, name='observation'
        )
        
        # Initialize game state
        self._reset_game_state()
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset_game_state(self):
        """Reset all game variables to initial state"""
        # Player state
        self.player_x = self.START_X
        self.player_z = self.START_Z
        self.player_y = self.START_Y
        self.velocity_y = 0.0
        self.jumping = False
        self.ducking = False
        self._last_action_change = 0
        self._last_action = 0
        
        # Beams
        self.beams = []
        self.beam_timer = 0
        self.beam_spawn_interval = self.INITIAL_SPAWN_INTERVAL
        self.current_spawn_acceleration = self.spawn_acceleration_rate
        
        # Speed
        self.current_bar_speed = self.INITIAL_BAR_SPEED
        
        # Game state
        self.score = 0
        self.survival_time = 0
        self.episode_ended = False
    
    def _reset(self):
        """Reset environment for new episode"""
        self._reset_game_state()
        self.current_step = 0
        return ts.restart(self._get_observation())
    
    def _step(self, action):
        """Execute one step in the environment"""
        if self.episode_ended:
            return self.reset()
        
        self.current_step += 1
        self.survival_time += 1
        
        # Execute action
        self._execute_action(action)
        
        # Update game physics
        self._update_physics()
        
        # Update beams
        self._update_beams()
        
        # Check collisions
        collision = self._check_collisions()
        
        # Check if player fell off platform
        fell_off = self._check_bounds()
        
        # Calculate reward
        reward = self._calculate_reward(collision, fell_off)
        
        # Check if episode should end
        if collision or fell_off or self.current_step >= self.max_steps:
            self.episode_ended = True
            return ts.termination(self._get_observation(), reward)
        else:
            return ts.transition(self._get_observation(), reward)
    
    def _execute_action(self, action):
        """Execute the given action with proper ducking logic"""
        # Track action changes
        if action != self._last_action:
            self._last_action_change = self.current_step
            self._last_action = action
            
        # Actions: 0 = do nothing, 1 = jump, 2 = duck
        if action == 1 and not self.jumping and not self.ducking and self.player_y <= self.START_Y + 0.1:  # Jump only if on ground and not ducking
            self.velocity_y = self.jump_force
            self.jumping = True
            self.ducking = False
        elif action == 2:  # Duck - removed the "not jumping" condition to allow maintaining duck
            self.ducking = True
            self.jumping = False
            self.velocity_y = min(0, self.velocity_y)  # Cancel any upward velocity
        elif action == 0:  # Do nothing - IMMEDIATELY stop ducking
            self.ducking = False
            # Don't stop jumping mid-air, let physics handle it
            if self.player_y <= self.START_Y + 0.1:  # Only reset jumping if on ground
                self.jumping = False
        
    def _update_physics(self):
        """Update player physics"""
        # Apply gravity
        self.velocity_y += self.gravity
        self.player_y += self.velocity_y
        
        # Ground collision
        if self.player_y <= self.START_Y:
            self.player_y = self.START_Y
            self.velocity_y = 0
            self.jumping = False
    
    def _update_beams(self):
        """Update beam spawning and movement"""
        # Update timer
        self.beam_timer += 1
        
        # Gradually speed up spawning
        self.current_spawn_acceleration = min(self.MAX_SPAWN_ACCELERATION, self.current_spawn_acceleration)
        self.beam_spawn_interval = max(self.min_spawn_interval, 
                                     self.beam_spawn_interval - self.current_spawn_acceleration)
        
        # Gradually increase bar speed
        self.current_bar_speed = min(self.MAX_BAR_SPEED, 
                                   self.current_bar_speed + self.bar_speed_increase)
        
        # Spawn new beam
        if self.beam_timer >= self.beam_spawn_interval:
            self.beam_timer = 0
            height = random.choice([1.5, 3.0])
            self.beams.append({
                "x": self.PLATFORM_WIDTH / 2 + 10,
                "z": 0,
                "width": 0.5,
                "height": height,
                "y_base": 1.5 if height == 1.5 else 3.0,
            })
        
        # Move beams
        for beam in self.beams:
            beam["x"] -= self.current_bar_speed
        
        # Remove off-screen beams
        self.beams = [b for b in self.beams if b["x"] > -self.PLATFORM_WIDTH / 2 - 2]
    
    def _check_collisions(self):
        """Check for collisions between player and beams"""
        player_hw = self.CUBE_WIDTH / 2
        player_hd = self.CUBE_DEPTH / 2
        current_height = self.CUBE_HEIGHT_DUCK if self.ducking else self.CUBE_HEIGHT_NORMAL
        player_y_base = self.player_y
        player_y_top = player_y_base + current_height
        
        # Add buffer width for collision detection
        BUFFER_WIDTH = 0.3  # Extra width to ensure proper ducking duration
        
        for beam in self.beams:
            beam_hw = beam["width"] / 2 + BUFFER_WIDTH  # Add buffer to beam width
            beam_hd = self.PLATFORM_DEPTH / 2
            beam_y_base = beam["y_base"]
            beam_y_top = beam_y_base + beam["height"]
            
            # 3D AABB collision check with buffered width
            if (abs(self.player_x - beam["x"]) < player_hw + beam_hw and
                abs(self.player_z - beam["z"]) < player_hd + beam_hd and
                player_y_base < beam_y_top and
                player_y_top > beam_y_base):
                return True
        
        return False
    
    def _check_bounds(self):
        """Check if player fell off the platform"""
        half_w = self.PLATFORM_WIDTH / 2
        half_d = self.PLATFORM_DEPTH / 2
        
        return (self.player_x < -half_w or self.player_x > half_w or
                self.player_z < -half_d or self.player_z > half_d)
    
    def _calculate_reward(self, collision: bool, fell_off: bool):
        """Calculate reward with proper incentives for ducking and jumping"""
        reward = 0.0
        
        if collision or fell_off:
            # Severe penalty for collisions
            reward = -800.0  # Even higher base penalty
            
            # Extra penalty for failing on a high beam while not ducking
            if collision and any(b["height"] == 3.0 and abs(b["x"] - self.player_x) < self.CUBE_WIDTH * 2 for b in self.beams):
                if not self.ducking or self._last_action != 2:
                    reward -= 500.0  # Much higher penalty for failing to maintain duck
            
            # Extra penalty for failing on a low beam while not jumping
            if collision and any(b["height"] == 1.5 and abs(b["x"] - self.player_x) < self.CUBE_WIDTH * 2 for b in self.beams):
                if not self.jumping:
                    reward -= 500.0  # Equal penalty for failing to jump
                elif self.ducking:  # Extra penalty for ducking at low beam
                    reward -= 200.0
            
            return reward
        
        # Base survival reward
        reward = 5.0
        
        # Progressive reward based on survival time
        reward += min(20.0, self.survival_time * 0.02)
        
        # Speed bonus
        speed_bonus = (self.current_bar_speed - self.INITIAL_BAR_SPEED) * 30.0
        reward += max(0, speed_bonus)
        
        # Calculate player bounds
        player_left = self.player_x - self.CUBE_WIDTH / 2
        player_right = self.player_x + self.CUBE_WIDTH / 2
        current_height = self.CUBE_HEIGHT_DUCK if self.ducking else self.CUBE_HEIGHT_NORMAL
        
        # Process each beam's interaction with the player
        for beam in self.beams:
            beam_left = beam["x"] - beam["width"] / 2
            beam_right = beam["x"] + beam["width"] / 2
            
            # Calculate overlap zones with extended buffer for ducking
            extended_width = self.CUBE_WIDTH * 2.5  # Much larger interaction zone
            approach_zone = beam_left - extended_width > player_right
            passing_zone = (beam_left - extended_width <= player_right and 
                          beam_right + extended_width >= player_left)
            past_zone = beam_right + extended_width * 0.6 < player_left  # Even more reduced past zone check
            
            # Distance to beam center for scaling
            distance = beam["x"] - self.player_x
            distance_scale = math.exp(-abs(distance) / 3.5)  # Much slower decay
            
            if beam["height"] == 3.0:  # High beam
                if passing_zone:
                    if self.ducking and self._last_action == 2:
                        # Major reward for maintaining duck while beam passes
                        duck_maintain_reward = 220.0 * distance_scale
                        # Extra reward for perfect positioning
                        position_bonus = 100.0 * (1.0 - min(1.0, abs(distance) / (beam["width"] + extended_width)))
                        reward += duck_maintain_reward + position_bonus
                    else:
                        # Severe penalty for not maintaining duck during passage
                        reward -= 250.0 * distance_scale
                
                elif approach_zone and distance < 6.0:
                    if self.ducking and self._last_action == 2:
                        # Reward for early preparation, scaled by distance
                        prep_scale = 1.0 - min(1.0, (distance - 3.0) / 3.0)
                        reward += 100.0 * prep_scale
                    elif distance < 4.0:
                        # Penalty for not preparing to duck when beam is close
                        reward -= 120.0 * distance_scale
                
                elif past_zone and abs(distance) < extended_width * 1.2:  # Extended check distance
                    if self.ducking and self._last_action == 2:
                        # Higher reward for maintaining duck through complete passage
                        reward += 150.0  # Further increased reward for maintaining duck
                    else:
                        # Higher penalty for early duck release
                        reward -= 200.0  # Further increased penalty for early release
            
            else:  # Low beam
                if passing_zone:
                    if self.jumping:
                        # High reward for jumping over low beam
                        jump_reward = 150.0 * distance_scale
                        if self.velocity_y > 0:  # Extra reward for jumping at the right time
                            jump_reward += 50.0 * distance_scale
                        reward += jump_reward
                    elif self.ducking:
                        # Much higher penalty for ducking under low beam
                        reward -= 200.0 * distance_scale  # Severe penalty for wrong action
                    else:
                        # Penalty for doing nothing at low beam
                        reward -= 100.0 * distance_scale
                
                elif approach_zone and distance < 4.0:
                    if self.jumping:
                        # Higher reward for well-timed jump preparation
                        jump_prep_reward = 60.0 * distance_scale
                        if self.velocity_y > 0:  # Extra reward for good jump timing
                            jump_prep_reward += 40.0 * distance_scale
                        reward += jump_prep_reward
                    elif self.ducking:  # Penalize ducking during jump beam approach
                        reward -= 150.0 * distance_scale
                    elif distance < 2.5:  # Penalize not preparing to jump
                        reward -= 50.0 * distance_scale
        
        # Penalties for unnecessary actions when no beams are nearby
        nearby_beams = any(abs(b["x"] - self.player_x) < max(6.0, self.CUBE_WIDTH * 4) for b in self.beams)
        if not nearby_beams:
            if self.ducking:
                # Higher penalty for unnecessary ducking
                reward -= 20.0
            if self.jumping:
                # Higher penalty for unnecessary jumping
                reward -= 15.0
            if not self.ducking and not self.jumping and self.player_y <= self.START_Y + 0.1:
                # Increased reward for maintaining ready position
                reward += 10.0
        
        return reward

    def _get_observation(self):
        """Get current observation state with enhanced beam information"""
        # Get next two beams that are approaching
        upcoming_beams = [b for b in self.beams if b["x"] > self.player_x - self.CUBE_WIDTH/2]
        upcoming_beams.sort(key=lambda b: b["x"])
        
        # Pad with default values if not enough beams
        while len(upcoming_beams) < 2:
            upcoming_beams.append({
                "x": 100.0,  # Far away
                "height": 1.5,
                "y_base": 1.5
            })
        
        # Calculate distances considering player width
        beam1_distance = upcoming_beams[0]["x"] - (self.player_x + self.CUBE_WIDTH/2)
        beam2_distance = upcoming_beams[1]["x"] - (self.player_x + self.CUBE_WIDTH/2)
        
        # Enhanced beam timing information
        beam1_critical = float(0 < beam1_distance < max(1.5, self.CUBE_WIDTH))
        beam2_critical = float(0 < beam2_distance < max(1.5, self.CUBE_WIDTH))
        
        # Is the player tall enough to hit the beam?
        current_height = self.CUBE_HEIGHT_DUCK if self.ducking else self.CUBE_HEIGHT_NORMAL
        player_top = self.player_y + current_height
        
        # Enhanced danger assessment considering player width
        beam1_danger = 1.0 if (beam1_distance < max(3.0, self.CUBE_WIDTH * 2) and 
                            upcoming_beams[0]["y_base"] < player_top and 
                            upcoming_beams[0]["y_base"] + upcoming_beams[0]["height"] > self.player_y) else 0.0
        
        # Convert beam heights to binary indicators (1.5 = 0, 3.0 = 1)
        beam1_is_high = float(upcoming_beams[0]["height"] == 3.0)
        beam2_is_high = float(upcoming_beams[1]["height"] == 3.0)
        
        # Time since last action change
        time_in_state = float(self.current_step - self._last_action_change) / 10.0
        
        observation = np.array([
            self.player_y,                    # Player Y position
            self.velocity_y,                  # Player Y velocity  
            float(self.jumping),              # Is jumping
            float(self.ducking),              # Is ducking
            beam1_distance,                   # Distance to next beam
            beam1_is_high,                    # Next beam is high (needs ducking)
            beam2_distance,                   # Distance to second beam
            beam2_is_high,                    # Second beam is high (needs ducking)
            self.current_bar_speed,           # Current game speed
            beam1_danger,                     # Will hit next beam if no action taken
            float(current_height == self.CUBE_HEIGHT_DUCK),  # Is player currently small
            float(beam1_distance < max(3.0, self.CUBE_WIDTH * 2)),  # Is beam in action range
            beam1_critical,                   # Is in critical passing zone for beam 1
            beam2_critical,                   # Is in critical passing zone for beam 2
            time_in_state,                    # How long in current action state
        ], dtype=np.float32)
        
        return observation
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Score: {self.survival_time}, "
                  f"Player Y: {self.player_y:.2f}, Beams: {len(self.beams)}")
        elif mode == 'rgb_array':
            # Return a simple representation as RGB array
            # This is a simplified version - you could integrate with pygame for full rendering
            return np.zeros((84, 84, 3), dtype=np.uint8)


class PlatformerTensorFlowEnv:
    """
    TensorFlow wrapper for the Platformer environment
    Provides utilities for training with TF-Agents
    """
    
    def __init__(self, max_steps: int = 10000):
        self.py_env = PlatformerPyEnvironment(max_steps=max_steps)
        self.tf_env = tf_py_environment.TFPyEnvironment(self.py_env)
    
    def get_py_environment(self):
        """Get the Python environment"""
        return self.py_env
    
    def get_tf_environment(self):
        """Get the TensorFlow environment"""
        return self.tf_env
    
    def create_training_env(self, num_parallel_envs: int = 1):
        """Create parallel environments for training"""
        if num_parallel_envs == 1:
            return self.tf_env
        else:
            # Create multiple parallel environments
            py_envs = [PlatformerPyEnvironment(max_steps=self.py_env.max_steps) 
                      for _ in range(num_parallel_envs)]
            return tf_py_environment.TFPyEnvironment(py_envs)


def create_platformer_env(max_steps: int = 10000, parallel_envs: int = 1):
    """
    Factory function to create the platformer environment
    
    Args:
        max_steps: Maximum steps per episode
        parallel_envs: Number of parallel environments for training
    
    Returns:
        TensorFlow environment ready for training
    """
    env_wrapper = PlatformerTensorFlowEnv(max_steps=max_steps)
    
    if parallel_envs == 1:
        return env_wrapper.get_tf_environment()
    else:
        return env_wrapper.create_training_env(parallel_envs)


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = create_platformer_env(max_steps=1000)
    
    # Test random actions
    time_step = env.reset()
    print("Initial observation shape:", time_step.observation.shape)
    print("Action spec:", env.action_spec())
    print("Observation spec:", env.observation_spec())
    
    # Run a few random steps
    for i in range(10):
        action = tf.random.uniform([], 0, 3, dtype=tf.int32)
        time_step = env.step(action)
        print(f"Step {i}: Action={action.numpy()}, Reward={time_step.reward.numpy():.2f}, "
              f"Done={time_step.is_last().numpy()}")
        
        if time_step.is_last():
            print("Episode ended, resetting...")
            time_step = env.reset()
            break
    
    print("Environment test completed successfully!")