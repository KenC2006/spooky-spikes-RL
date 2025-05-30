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
    def __init__(self, max_steps: int = 10000, render_mode: str = 'none'):
        super().__init__()
        
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.current_step = 0
        self._last_action_change = 0
        self._last_action = 0
        
        # Game dimensions and physics 
        self.PLATFORM_WIDTH = 10
        self.PLATFORM_DEPTH = 10
        self.START_X = 3
        self.START_Z = 0.0
        self.START_Y = 1.5
        self.CUBE_WIDTH = 1.5
        self.CUBE_DEPTH = 1.5
        self.CUBE_HEIGHT_NORMAL = 3.0
        self.CUBE_HEIGHT_DUCK = 1.5
        self.gravity = -0.03
        self.jump_force = 0.5

        # Beam spawning and movement 
        self.INITIAL_SPAWN_INTERVAL = 90
        self.min_spawn_interval = 30
        self.spawn_acceleration_rate = 0.02
        self.MAX_SPAWN_ACCELERATION = 0.4
        self.INITIAL_BAR_SPEED = 0.3
        self.MAX_BAR_SPEED = 0.6
        self.bar_speed_increase = 0.00005
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action'
        )
        
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(15,), dtype=np.float32, minimum=-50.0, maximum=50.0, name='observation'
        )
        
        self._reset_game_state()
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset_game_state(self):
        self.player_x = self.START_X
        self.player_z = self.START_Z
        self.player_y = self.START_Y
        self.velocity_y = 0.0
        self.jumping = False
        self.ducking = False
        self._last_action_change = 0
        self._last_action = 0
        self.beams = []
        self.beam_timer = 0
        self.beam_spawn_interval = self.INITIAL_SPAWN_INTERVAL
        self.current_spawn_acceleration = self.spawn_acceleration_rate
        self.current_bar_speed = self.INITIAL_BAR_SPEED
        self.score = 0
        self.survival_time = 0
        self.episode_ended = False
    
    def _reset(self):
        self._reset_game_state()
        self.current_step = 0
        return ts.restart(self._get_observation())
    
    def _step(self, action):
        if self.episode_ended:
            return self.reset()
        
        self.current_step += 1
        self.survival_time += 1
        
        self._execute_action(action)
        self._update_physics()
        self._update_beams()
        collision = self._check_collisions()
        
        reward = self._calculate_reward(collision)
        
        if collision or self.current_step >= self.max_steps:
            self.episode_ended = True
            return ts.termination(self._get_observation(), reward)
        else:
            return ts.transition(self._get_observation(), reward)
    
    def _execute_action(self, action):
        if action != self._last_action:
            self._last_action_change = self.current_step
            self._last_action = action
            
        
        if action == 1 and not self.jumping and not self.ducking and self.player_y <= self.START_Y + 0.1: # Jump
            self.velocity_y = self.jump_force
            self.jumping = True
            self.ducking = False
        elif action == 2:  # Duck
            self.ducking = True
            self.jumping = False
            self.velocity_y = min(0, self.velocity_y)
        elif action == 0:  # Idle
            self.ducking = False
            if self.player_y <= self.START_Y + 0.1:
                self.jumping = False
        
    def _update_physics(self):
        # Jump phyics
        self.velocity_y += self.gravity
        self.player_y += self.velocity_y
        
        # Ground collision
        if self.player_y <= self.START_Y:
            self.player_y = self.START_Y
            self.velocity_y = 0
            self.jumping = False
    
    def _update_beams(self):
        self.beam_timer += 1
        
        # Increase bar spawn interval and speed for difficulty 
        self.current_spawn_acceleration = min(self.MAX_SPAWN_ACCELERATION, self.current_spawn_acceleration)
        self.beam_spawn_interval = max(self.min_spawn_interval, 
                                     self.beam_spawn_interval - self.current_spawn_acceleration)
        
        self.current_bar_speed = min(self.MAX_BAR_SPEED, 
                                   self.current_bar_speed + self.bar_speed_increase)
        
        if self.beam_timer >= self.beam_spawn_interval:
            self.beam_timer = 0
            height = random.choice([1.5, 3.0]) # Low beam = 1.5, High beam = 3.0
            self.beams.append({
                "x": self.PLATFORM_WIDTH / 2 + 10,
                "z": 0,
                "width": 0.5,
                "height": height,
                "y_base": 1.5 if height == 1.5 else 3.0,
            })
        
        for beam in self.beams:
            beam["x"] -= self.current_bar_speed
        
        self.beams = [b for b in self.beams if b["x"] > -self.PLATFORM_WIDTH / 2 - 2]
    
    def _check_collisions(self):
        # Calculate player bounds
        player_hw = self.CUBE_WIDTH / 2
        player_hd = self.CUBE_DEPTH / 2
        current_height = self.CUBE_HEIGHT_DUCK if self.ducking else self.CUBE_HEIGHT_NORMAL
        player_y_base = self.player_y
        player_y_top = player_y_base + current_height
        
        BUFFER_WIDTH = 0.3 
        
        # Collision checking
        for beam in self.beams:
            beam_hw = beam["width"] / 2 + BUFFER_WIDTH
            beam_hd = self.PLATFORM_DEPTH / 2
            beam_y_base = beam["y_base"]
            beam_y_top = beam_y_base + beam["height"]
            
            if (abs(self.player_x - beam["x"]) < player_hw + beam_hw and
                abs(self.player_z - beam["z"]) < player_hd + beam_hd and
                player_y_base < beam_y_top and
                player_y_top > beam_y_base):
                return True
        
        return False
    
    def _calculate_reward(self, collision: bool):
        reward = 0.0

        # Reward penalties for hitting the beam
        if collision:
            reward = -800.0

            if any(b["height"] == 3.0 and abs(b["x"] - self.player_x) < self.CUBE_WIDTH * 2 for b in self.beams):
                if not self.ducking or self._last_action != 2:
                    reward -= 500.0

            if any(b["height"] == 1.5 and abs(b["x"] - self.player_x) < self.CUBE_WIDTH * 2 for b in self.beams):
                if not self.jumping:
                    reward -= 500.0
                elif self.ducking:
                    reward -= 200.0

            return reward

        # General survival penalties
        reward = 10.0
        reward += min(20.0, self.survival_time * 0.02)
        speed_bonus = (self.current_bar_speed - self.INITIAL_BAR_SPEED) * 30.0
        reward += max(0, speed_bonus)

        player_left = self.player_x - self.CUBE_WIDTH / 2
        player_right = self.player_x + self.CUBE_WIDTH / 2
        current_height = self.CUBE_HEIGHT_DUCK if self.ducking else self.CUBE_HEIGHT_NORMAL

        # Ai keeps unducking to early on the high beams so rewards for longer duck
        for beam in self.beams:
            beam_left = beam["x"] - beam["width"] / 2
            beam_right = beam["x"] + beam["width"] / 2

            extended_width = self.CUBE_WIDTH * 2.5
            approach_zone = beam_left - extended_width > player_right
            passing_zone = (beam_left - extended_width <= player_right and 
                            beam_right + extended_width >= player_left)
            past_zone = beam_right + extended_width * 0.6 < player_left

            distance = beam["x"] - self.player_x
            distance_scale = math.exp(-abs(distance) / 3.5)

            if beam["height"] == 3.0:
                if passing_zone:
                    if self.ducking and self._last_action == 2:
                        duck_maintain_reward = 220.0 * distance_scale
                        position_bonus = 100.0 * (1.0 - min(1.0, abs(distance) / (beam["width"] + extended_width)))
                        reward += duck_maintain_reward + position_bonus
                    else:
                        reward -= 250.0 * distance_scale

                elif approach_zone and distance < 6.0:
                    if self.ducking and self._last_action == 2:
                        prep_scale = 1.0 - min(1.0, (distance - 3.0) / 3.0)
                        reward += 100.0 * prep_scale
                    elif distance < 4.0:
                        reward -= 120.0 * distance_scale

                elif past_zone and abs(distance) < extended_width * 1.2:
                    if self.ducking and self._last_action == 2:
                        reward += 150.0
                    else:
                        reward -= 200.0

            else:
                if passing_zone:
                    if self.jumping:
                        jump_reward = 150.0 * distance_scale
                        if self.velocity_y > 0:
                            jump_reward += 50.0 * distance_scale
                        reward += jump_reward
                    elif self.ducking:
                        reward -= 200.0 * distance_scale
                    else:
                        reward -= 100.0 * distance_scale

                elif approach_zone and distance < 4.0:
                    if self.jumping:
                        jump_prep_reward = 60.0 * distance_scale
                        if self.velocity_y > 0:
                            jump_prep_reward += 40.0 * distance_scale
                        reward += jump_prep_reward
                    elif self.ducking:
                        reward -= 150.0 * distance_scale
                    elif distance < 2.5:
                        reward -= 50.0 * distance_scale

        nearby_beams = any(abs(b["x"] - self.player_x) < max(6.0, self.CUBE_WIDTH * 4) for b in self.beams)
        if not nearby_beams:
            if self.ducking:
                reward -= 20.0
            if self.jumping:
                reward -= 15.0
            if not self.ducking and not self.jumping and self.player_y <= self.START_Y + 0.1:
                reward += 10.0

        return reward

    def _get_observation(self):
        upcoming_beams = [b for b in self.beams if b["x"] > self.player_x - self.CUBE_WIDTH/2]
        upcoming_beams.sort(key=lambda b: b["x"])
        
        while len(upcoming_beams) < 2:
            upcoming_beams.append({
                "x": 100.0,  
                "height": 1.5,
                "y_base": 1.5
            })
        
        # Calculate distances and danger indicators
        beam1_distance = upcoming_beams[0]["x"] - (self.player_x + self.CUBE_WIDTH/2)
        beam2_distance = upcoming_beams[1]["x"] - (self.player_x + self.CUBE_WIDTH/2)
        
        beam1_critical = float(0 < beam1_distance < max(1.5, self.CUBE_WIDTH))
        beam2_critical = float(0 < beam2_distance < max(1.5, self.CUBE_WIDTH))
        
        current_height = self.CUBE_HEIGHT_DUCK if self.ducking else self.CUBE_HEIGHT_NORMAL
        player_top = self.player_y + current_height
        
        beam1_danger = 1.0 if (beam1_distance < max(3.0, self.CUBE_WIDTH * 2) and 
                            upcoming_beams[0]["y_base"] < player_top and 
                            upcoming_beams[0]["y_base"] + upcoming_beams[0]["height"] > self.player_y) else 0.0
        
        beam1_is_high = float(upcoming_beams[0]["height"] == 3.0)
        beam2_is_high = float(upcoming_beams[1]["height"] == 3.0)
        
        time_in_state = float(self.current_step - self._last_action_change) / 10.0
        
        observation = np.array([
            self.player_y,            
            self.velocity_y,           
            float(self.jumping),      
            float(self.ducking),      
            beam1_distance,         
            beam1_is_high,        
            beam2_distance,      
            beam2_is_high,       
            self.current_bar_speed, 
            beam1_danger,          
            float(current_height == self.CUBE_HEIGHT_DUCK),  
            float(beam1_distance < max(3.0, self.CUBE_WIDTH * 2)),  
            beam1_critical,          
            beam2_critical,         
            time_in_state,          
        ], dtype=np.float32)
        
        return observation


class PlatformerTensorFlowEnv:
    # Wrapper for converting the environment to TensorFlow
    def __init__(self, max_steps: int = 10000):
        self.py_env = PlatformerPyEnvironment(max_steps=max_steps)
        self.tf_env = tf_py_environment.TFPyEnvironment(self.py_env)
    
    def get_py_environment(self):
        return self.py_env
    
    def get_tf_environment(self):
        return self.tf_env
    
    def create_training_env(self, num_parallel_envs: int = 1):
        if num_parallel_envs == 1:
            return self.tf_env
        else:
            py_envs = [PlatformerPyEnvironment(max_steps=self.py_env.max_steps) 
                      for _ in range(num_parallel_envs)]
            return tf_py_environment.TFPyEnvironment(py_envs)


def create_platformer_env(max_steps: int = 10000, parallel_envs: int = 1):
    # Creates single/parallel training environments
    env_wrapper = PlatformerTensorFlowEnv(max_steps=max_steps)
    
    if parallel_envs == 1:
        return env_wrapper.get_tf_environment()
    else:
        return env_wrapper.create_training_env(parallel_envs)