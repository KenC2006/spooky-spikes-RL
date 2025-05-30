import tensorflow as tf
import numpy as np
import pygame
import math
import random
import time
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import tf_py_environment

from platformer_env import PlatformerPyEnvironment, create_platformer_env

class VisualPlatformerEnv(PlatformerPyEnvironment):
    
    def __init__(self, max_steps=10000, render_mode='human', show_ai_info=True):
        super().__init__(max_steps, render_mode)
        self.show_ai_info = show_ai_info
        self.last_action = 0
        self.last_q_values = [0, 0, 0]
        self.confidence = 0
        self.pygame_initialized = False
        self.should_quit = False
        
        if render_mode == 'human':
            self.init_pygame()
    
    def init_pygame(self):
        try:
            pygame.init()
            self.WIDTH, self.HEIGHT = 1000, 700
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Spooky Spikes RL")
            self.clock = pygame.time.Clock()
            self.target_fps = 40
            self.frame_skip = 0 
            self.font = pygame.font.Font(None, 24)
            self.big_font = pygame.font.Font(None, 36)
            self.pygame_initialized = True
            
            # Color scheme was AI'd
            self.RED = (255, 0, 0)
            self.SKY_COLOR = (25, 25, 35) 
            self.PLATFORM_COLOR = (45, 45, 55)  
            self.PLATFORM_GRID_COLOR = (55, 55, 65)  
            self.SIDE_COLOR = (220, 50, 50) 
            self.TOP_COLOR = (255, 70, 70)  
            self.BEAM_COLOR_LOW = (140, 200, 255)  
            self.BEAM_COLOR_HIGH = (255, 165, 0)  
            self.BEAM_EDGE_LOW = (160, 220, 255)  
            self.BEAM_EDGE_HIGH = (255, 185, 20) 
            self.TEXT_COLOR = (200, 200, 220)  
            self.AI_COLOR = (0, 255, 170) 
            self.GRID_SIZE = 1.0 
            
            # Camera settings for the projections
            self.CAMERA_HEIGHT = 10
            self.CAMERA_DISTANCE = 28
            self.FOV = math.radians(60)
            
        except Exception as e:
            print(f"Failed to initialize: {e}")
            self.pygame_initialized = False
    
    def project_3d(self, x, y, z):
        rel_z = z + self.CAMERA_DISTANCE
        if rel_z <= 0.01:
            rel_z = 0.01
        scale = self.FOV / rel_z
        x_offset = -4
        sx = int((x + x_offset) * scale * self.WIDTH * 0.6 + self.WIDTH * 0.3)
        sy = int(self.HEIGHT * 0.5 - (y - self.CAMERA_HEIGHT) * scale * self.HEIGHT * 0.6)
        return sx, sy
    
    def draw_platform(self):
        if not self.pygame_initialized:
            return
            
        px, py, pz = 0, 0, 0
        pw, pd, ph = self.PLATFORM_WIDTH, self.PLATFORM_DEPTH, 1.5
        
        hx = pw / 2
        hz = pd / 2
        corners = {
            "flb": (px - hx, py, pz - hz),
            "frb": (px + hx, py, pz - hz),
            "blb": (px - hx, py, pz + hz),
            "brb": (px + hx, py, pz + hz),
            "flt": (px - hx, py + ph, pz - hz),
            "frt": (px + hx, py + ph, pz - hz),
            "blt": (px - hx, py + ph, pz + hz),
            "brt": (px + hx, py + ph, pz + hz)
        }
        
        p = {k: self.project_3d(*v) for k, v in corners.items()}
        
        faces = [
            (["blb", "brb", "brt", "blt"], self.PLATFORM_COLOR),
            (["frb", "brb", "brt", "frt"], self.PLATFORM_COLOR),
            (["flb", "frb", "frt", "flt"], self.PLATFORM_COLOR),
            (["flb", "blb", "blt", "flt"], self.PLATFORM_COLOR),
            (["flt", "frt", "brt", "blt"], self.PLATFORM_COLOR),
        ]
        
        def avg_z(face): 
            return sum([corners[k][2] for k in face[0]]) / 4
        faces.sort(key=lambda face: avg_z(face), reverse=True)
        
        for keys, color in faces:
            try:
                points = [p[k] for k in keys]
                pygame.draw.polygon(self.screen, color, points)
            except:
                pass

        for x in np.arange(-hx, hx + self.GRID_SIZE, self.GRID_SIZE):
            start = self.project_3d(x, py + 0.01, -hz) 
            end = self.project_3d(x, py + 0.01, hz)
            pygame.draw.line(self.screen, self.PLATFORM_GRID_COLOR, start, end, 1)
            
        for z in np.arange(-hz, hz + self.GRID_SIZE, self.GRID_SIZE):
            start = self.project_3d(-hx, py + 0.01, z)
            end = self.project_3d(hx, py + 0.01, z)
            pygame.draw.line(self.screen, self.PLATFORM_GRID_COLOR, start, end, 1)
    
    def draw_cube(self, x, y, z, width, depth, height):
        if not self.pygame_initialized:
            return
            
        hw = width / 2
        hd = depth / 2
        
        corners = {
            "flb": (x - hw, y, z - hd),
            "frb": (x + hw, y, z - hd),
            "blb": (x - hw, y, z + hd),
            "brb": (x + hw, y, z + hd),
            "flt": (x - hw, y + height, z - hd),
            "frt": (x + hw, y + height, z - hd),
            "blt": (x - hw, y + height, z + hd),
            "brt": (x + hw, y + height, z + hd)
        }
        
        p = {k: self.project_3d(*v) for k, v in corners.items()}
        
        faces = [
            (["blb", "brb", "brt", "blt"], self.SIDE_COLOR),
            (["frb", "brb", "brt", "frt"], self.SIDE_COLOR),
            (["flb", "frb", "frt", "flt"], self.SIDE_COLOR),
            (["flb", "blb", "blt", "flt"], self.SIDE_COLOR),
            (["flt", "frt", "brt", "blt"], self.TOP_COLOR),
        ]
        
        def avg_z(face): 
            return sum([corners[k][2] for k in face[0]]) / 4
        faces.sort(key=lambda face: avg_z(face), reverse=True)
        
        for keys, color in faces:
            try:
                points = [p[k] for k in keys]
                pygame.draw.polygon(self.screen, color, points)
            except:
                pass
    
    def draw_beam(self, beam):
        if not self.pygame_initialized:
            return
            
        x = beam["x"]
        z = beam["z"]
        bw = beam["width"]
        bh = beam["height"]
        bd = self.PLATFORM_DEPTH
        beam_y_base = beam.get("y_base", 1.5)
        beam_y_top = beam_y_base + bh
        
        hw = bw / 2
        hd = bd / 2
        
        corners = {
            "flb": (x - hw, beam_y_base, z - hd),
            "frb": (x + hw, beam_y_base, z - hd),
            "blb": (x - hw, beam_y_base, z + hd),
            "brb": (x + hw, beam_y_base, z + hd),
            "flt": (x - hw, beam_y_top, z - hd),
            "frt": (x + hw, beam_y_top, z - hd),
            "blt": (x - hw, beam_y_top, z + hd),
            "brt": (x + hw, beam_y_top, z + hd)
        }
        
        p = {k: self.project_3d(*v) for k, v in corners.items()}
        
        beam_color = self.BEAM_COLOR_LOW if bh == 1.5 else self.BEAM_COLOR_HIGH
        edge_color = self.BEAM_EDGE_LOW if bh == 1.5 else self.BEAM_EDGE_HIGH
        
        faces = [
            (["blb", "brb", "brt", "blt"], beam_color),
            (["frb", "brb", "brt", "frt"], beam_color),
            (["flb", "frb", "frt", "flt"], beam_color),
            (["flb", "blb", "blt", "flt"], beam_color),
            (["flt", "frt", "brt", "blt"], beam_color),
        ]
        
        def avg_z(face): 
            return sum([corners[k][2] for k in face[0]]) / 4
        faces.sort(key=lambda face: avg_z(face), reverse=True)
        
        for keys, color in faces:
            try:
                points = [p[k] for k in keys]
                pygame.draw.polygon(self.screen, color, points)
                pygame.draw.polygon(self.screen, edge_color, points, 2)
            except:
                pass
    
    def draw_ui(self):
        if not self.pygame_initialized or not self.show_ai_info:
            return
        
        panel_x = self.WIDTH - 300
        panel_y = 10
        panel_width = 280
        panel_height = 200
        
        gradient_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        for i in range(panel_height):
            alpha = int(128 * (1 - i/panel_height)) 
            pygame.draw.line(gradient_surface, (0, 0, 0, alpha), 
                           (0, i), (panel_width, i))
        self.screen.blit(gradient_surface, (panel_x, panel_y))
        
        border_glow = pygame.Surface((panel_width + 4, panel_height + 4), pygame.SRCALPHA)
        pygame.draw.rect(border_glow, (*self.AI_COLOR, 64), 
                        (0, 0, panel_width + 4, panel_height + 4), 4)
        self.screen.blit(border_glow, (panel_x - 2, panel_y - 2))
        pygame.draw.rect(self.screen, self.AI_COLOR, 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        title = self.big_font.render("AI STATUS", True, self.AI_COLOR)
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        y_offset = panel_y + 50
        
        # Current action
        actions = ["IDLE", "JUMP", "DUCK"]
        action_text = f"Action: {actions[self.last_action]}"
        action_surface = self.font.render(action_text, True, self.TEXT_COLOR)
        self.screen.blit(action_surface, (panel_x + 10, y_offset))
        y_offset += 25
        
        self.screen.blit(self.font.render("Q-Values:", True, self.TEXT_COLOR), 
                        (panel_x + 10, y_offset))
        y_offset += 20
        
        for i, (action, q_val) in enumerate(zip(actions, self.last_q_values)):
            color = self.AI_COLOR if i == self.last_action else self.TEXT_COLOR
            q_text = f"  {action}: {q_val:.3f}"
            q_surface = self.font.render(q_text, True, color)
            self.screen.blit(q_surface, (panel_x + 10, y_offset))
            
            bar_width = int(max(0, min(100, q_val * 50))) 
            if bar_width > 0:
                pygame.draw.rect(self.screen, color, 
                               (panel_x + 150, y_offset + 5, bar_width, 5))
            y_offset += 18
        
        # Confidence number is lokey bogus and means nothing
        conf_text = f"Confidence: {self.confidence*100:.1f}%"
        conf_surface = self.font.render(conf_text, True, self.TEXT_COLOR)
        self.screen.blit(conf_surface, (panel_x + 10, y_offset))
        
        stats_y = self.HEIGHT - 100
        stats = [
            f"Survival Time: {self.survival_time}",
            f"Speed: {self.current_bar_speed:.3f}",
            f"Beams: {len(self.beams)}",
            f"Height: {self.player_y:.2f}"
        ]
        
        for i, stat in enumerate(stats):
            stat_surface = pygame.Surface((200, 20), pygame.SRCALPHA)
            pygame.draw.rect(stat_surface, (0, 0, 0, 100), (0, 0, 200, 20))
            self.screen.blit(stat_surface, (5, stats_y + i * 22 - 2))
            
            text_surface = self.font.render(stat, True, self.TEXT_COLOR)
            self.screen.blit(text_surface, (10, stats_y + i * 22))
    
    def handle_events(self):
        """Handle pygame events"""
        if not self.pygame_initialized:
            return True
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_quit = True
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.should_quit = True
                    return False
        return True
    
    def render_frame(self):
        """Render a single frame"""
        if not self.pygame_initialized or self.should_quit:
            return False
        
        if not self.handle_events():
            return False

        # Skipping frames so it runs a bit smoother
        self.frame_skip = (self.frame_skip + 1) % 2
        if self.frame_skip != 0:
            return True
        
        try:
            self.screen.fill(self.SKY_COLOR)
            
            self.draw_platform()
            
            current_height = self.CUBE_HEIGHT_DUCK if self.ducking else self.CUBE_HEIGHT_NORMAL
            self.draw_cube(self.player_x, self.player_y, self.player_z, 
                          self.CUBE_WIDTH, self.CUBE_DEPTH, current_height)
            
            for beam in self.beams:
                self.draw_beam(beam)
            
            self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(self.target_fps)  
            
        except Exception as e:
            print(f"Render error: {e}")
            return False
            
        return True
    
    def update_ai_info(self, action, q_values=None):
        """Update AI information for display with proper confidence"""
        self.last_action = action
        if q_values is not None:
            self.last_q_values = q_values.tolist()
            
            probabilities = tf.nn.softmax(q_values).numpy()
            self.confidence = float(np.max(probabilities))  # fixed range to 0-1 
    
    def close(self):
        """Clean up pygame resources"""
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False


class SimpleTrainer:
    # Neural network params and training settings
    def __init__(self):
        self.train_env = create_platformer_env(max_steps=3000)
        self.batch_size = 512
        
        preprocessing_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(
                512, activation='elu',
                kernel_regularizer=tf.keras.regularizers.l2(0.005),
                name='hidden_1'
            ),
            tf.keras.layers.Dense(
                256, activation='elu',
                kernel_regularizer=tf.keras.regularizers.l2(0.005),
                name='hidden_2'
            ),
            tf.keras.layers.Dense(
                128, activation='elu',
                kernel_regularizer=tf.keras.regularizers.l2(0.005),
                name='hidden_3'
            )
        ])
        
        self.q_net = q_network.QNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            preprocessing_layers=preprocessing_layers,
            activation_fn=tf.keras.activations.elu
        )


        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=2e-4,
            clipnorm=0.5,
            epsilon=1e-5
        )
        
        self.train_step_counter = tf.Variable(0)
        
        self.initial_epsilon = 1.0
        self.min_epsilon = 0.1
        self.total_decay_steps = 20000
        
        preprocessing_layers_target = tf.keras.Sequential([
            tf.keras.layers.Dense(
                512, activation='elu',
                kernel_regularizer=tf.keras.regularizers.l2(0.005),
                name='target_hidden_1'
            ),
            tf.keras.layers.Dense(
                256, activation='elu',
                kernel_regularizer=tf.keras.regularizers.l2(0.005),
                name='target_hidden_2'
            ),
            tf.keras.layers.Dense(
                128, activation='elu',
                kernel_regularizer=tf.keras.regularizers.l2(0.005),
                name='target_hidden_3'
            )
        ])
        
        self.target_q_net = q_network.QNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            preprocessing_layers=preprocessing_layers_target,
            activation_fn=tf.keras.activations.elu
        )

        self.agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=self.q_net,
            optimizer=self.optimizer,
            target_q_network=self.target_q_net,
            td_errors_loss_fn=common.element_wise_huber_loss,
            train_step_counter=self.train_step_counter,
            target_update_period=50,
            target_update_tau=0.01,
            epsilon_greedy=self.initial_epsilon,
            gamma=0.99,
            reward_scale_factor=0.1,
            gradient_clipping=True,
            debug_summaries=True,
            summarize_grads_and_vars=True
        )
        
        self.agent.initialize()
        
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=100000
        )
        
        self.random_policy = random_tf_policy.RandomTFPolicy(
            self.train_env.time_step_spec(),
            self.train_env.action_spec()
        )
    
    def collect_step(self, environment, policy, epsilon=None):
        if epsilon is not None:
            original_epsilon = self.agent._epsilon_greedy
            self.agent._epsilon_greedy = epsilon
            
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        self.replay_buffer.add_batch(traj)
        
        if epsilon is not None:
            self.agent._epsilon_greedy = original_epsilon
    
    def update_epsilon(self, training_step):
        step = min(training_step, self.total_decay_steps)
        progress = step / self.total_decay_steps
        new_epsilon = self.initial_epsilon - ((self.initial_epsilon - self.min_epsilon) * progress)
        self.agent._epsilon_greedy = new_epsilon
        return new_epsilon
    
    def evaluate_agent(self, num_episodes=20):
        total_return = 0.0
        original_epsilon = self.agent._epsilon_greedy
        self.agent._epsilon_greedy = 0.0
        
        try:
            for _ in range(num_episodes):
                time_step = self.train_env.reset()
                episode_return = 0.0
                while not time_step.is_last():
                    action_step = self.agent.policy.action(time_step)
                    time_step = self.train_env.step(action_step.action)
                    episode_return += float(time_step.reward)
                total_return += episode_return
        finally:
            self.agent._epsilon_greedy = original_epsilon
        
        avg_return = total_return / num_episodes
        return avg_return
    
    def train_basic_agent(self):
        print("Training AI")
        
        env = self.train_env
        env.reset()
        
        print("Phase 1: Collecting random experience")
        for i in range(100000):
            self.collect_step(env, self.agent.collect_policy, epsilon=1.0)
            if i % 1000 == 0:
                print(f"Random experience: {i}/100000")
        
        print("Phase 2: Training neural network")
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size,
            num_steps=2
        ).prefetch(3)
        
        iterator = iter(dataset)
        
        for i in range(20000):
            try:
                experience, _ = next(iterator)
                train_step = self.agent.train(experience)
                loss = float(train_step.loss.numpy())
                current_epsilon = self.update_epsilon(i)
                
                if i % 500 == 0:
                    avg_return = float(self.evaluate_agent(num_episodes=20))
                    print(f"Step {i}, Loss: {loss:.4f}, Epsilon: {current_epsilon:.4f}, "
                          f"Avg Return: {avg_return:.2f}")
                
            except Exception as e:
                print(f"Training error at step {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                break
        
        print("Training complete!")
        return self.agent



def watch_trained_ai():
    print("Training AI agent...")
    
    try:
        trainer = SimpleTrainer()
        trained_agent = trainer.train_basic_agent()
        
        print("Trained AI now running")
        visual_env = VisualPlatformerEnv(max_steps=5000, render_mode='human')
        visual_tf_env = tf_py_environment.TFPyEnvironment(visual_env)
        
        if not visual_env.pygame_initialized:
            print("ERROR: failed")
            return False
        
        time_step = visual_tf_env.reset()
        episode_return = 0
        
        while not visual_env.should_quit:
            q_values = trainer.q_net(time_step.observation)[0].numpy()[0]
            action_step = trained_agent.policy.action(time_step)
            
            visual_env.update_ai_info(action_step.action.numpy()[0], q_values)
            time_step = visual_tf_env.step(action_step.action)
            episode_return += time_step.reward.numpy()[0]
            
            if time_step.is_last():
                time_step = visual_tf_env.reset()
            
            if not visual_env.render_frame():
                break
        
        visual_env.close()
        print(f"Episode score: {episode_return:.1f}")
        return True
        
    except Exception as e:
        print(f"Error in demo: {e}")
        return False


def compare_random_vs_trained():
    print("First you'll see random AI, then trained AI")
    
    try:
        print("Showing RANDOM AI")
        visual_env = VisualPlatformerEnv(max_steps=1000, render_mode='human')
        visual_tf_env = tf_py_environment.TFPyEnvironment(visual_env)
        
        if not visual_env.pygame_initialized:
            print("ERROR: failed")
            return False
            
        random_policy = random_tf_policy.RandomTFPolicy(
            visual_tf_env.time_step_spec(), visual_tf_env.action_spec()
        )
        
        time_step = visual_tf_env.reset()
        random_score = 0
        
        while not visual_env.should_quit:
            action_step = random_policy.action(time_step)
            visual_env.update_ai_info(action_step.action.numpy()[0])
            time_step = visual_tf_env.step(action_step.action)
            random_score += time_step.reward.numpy()[0]
            
            if time_step.is_last():
                time_step = visual_tf_env.reset()
            
            if not visual_env.render_frame():
                break
        
        visual_env.close()
        print(f"Random AI score: {random_score:.1f}")
        
        if not visual_env.should_quit:
            input("Press Enter to see the TRAINED AI...")
            return watch_trained_ai()
        return True
    
    except Exception as e:
        print(f"Error in comparison: {e}")
        return False


def test_visual_environment():
    print("Testing 3D rendering and game env...")
    
    try:
        visual_env = VisualPlatformerEnv(max_steps=500, render_mode='human')
        visual_tf_env = tf_py_environment.TFPyEnvironment(visual_env)
        
        if not visual_env.pygame_initialized:
            print("ERROR: failed")
            return False
        
        time_step = visual_tf_env.reset()
        
        while not visual_env.should_quit:
            action = tf.random.uniform([], 0, 3, dtype=tf.int32)
            visual_env.update_ai_info(action.numpy())
            time_step = visual_tf_env.step(action)
            
            if time_step.is_last():
                time_step = visual_tf_env.reset()
            
            if not visual_env.render_frame():
                break
        
        visual_env.close()
        print("Visual test completed")
        return True
        
    except Exception as e:
        print(f"Visual test failed: {e}")
        return False


if __name__ == "__main__":
    print("\nChoose what you want to see:")
    print("1. Watch trained AI play")
    print("2. Compare random vs trained AI")
    print("3. Test runtime environment")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        watch_trained_ai()
    elif choice == "2":
        compare_random_vs_trained()
    elif choice == "3":
        success = test_visual_environment()
        if success:
            print("\nThe environment works. You can now try option 1.")
        else:
            print("\nThere's an issue with the visual environment. Check your pygame installation.")
    else:
        print("Invalid choice")