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

# Import the environment
from platformer_env import PlatformerPyEnvironment, create_platformer_env

class VisualPlatformerEnv(PlatformerPyEnvironment):
    """Enhanced environment with pygame visualization"""
    
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
        """Initialize pygame for visualization"""
        try:
            pygame.init()
            self.WIDTH, self.HEIGHT = 1000, 700
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("AI Learning 3D Platformer")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.big_font = pygame.font.Font(None, 36)
            self.pygame_initialized = True
            
            # Colors
            self.RED = (255, 0, 0)
            self.SKY_COLOR = (180, 220, 255)
            self.PLATFORM_COLOR = (100, 100, 100)
            self.SIDE_COLOR = (200, 50, 50)
            self.TOP_COLOR = (255, 70, 70)
            self.BEAM_COLOR_LOW = (200, 200, 200)
            self.BEAM_COLOR_HIGH = (255, 140, 0)
            self.TEXT_COLOR = (255, 255, 255)
            self.AI_COLOR = (0, 255, 0)
            
            # Camera settings for 3D projection
            self.CAMERA_HEIGHT = 10
            self.CAMERA_DISTANCE = 28
            self.FOV = math.radians(60)
            
        except Exception as e:
            print(f"Failed to initialize pygame: {e}")
            self.pygame_initialized = False
    
    def project_3d(self, x, y, z):
        """Project 3D coordinates to 2D screen coordinates"""
        rel_z = z + self.CAMERA_DISTANCE
        if rel_z <= 0.01:
            rel_z = 0.01
        scale = self.FOV / rel_z
        x_offset = -4
        sx = int((x + x_offset) * scale * self.WIDTH * 0.6 + self.WIDTH * 0.3)
        sy = int(self.HEIGHT * 0.5 - (y - self.CAMERA_HEIGHT) * scale * self.HEIGHT * 0.6)
        return sx, sy
    
    def draw_platform(self):
        """Draw the 3D platform"""
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
    
    def draw_cube(self, x, y, z, width, depth, height):
        """Draw the player cube"""
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
        """Draw a beam obstacle"""
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
            except:
                pass
    
    def draw_ui(self):
        """Draw AI status information"""
        if not self.pygame_initialized or not self.show_ai_info:
            return
        
        # AI Status Panel
        panel_x = self.WIDTH - 300
        panel_y = 10
        panel_width = 280
        panel_height = 200
        
        # Draw panel background
        pygame.draw.rect(self.screen, (0, 0, 0, 128), 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.AI_COLOR, 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title
        title = self.big_font.render("AI STATUS", True, self.AI_COLOR)
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        y_offset = panel_y + 50
        
        # Current action
        actions = ["IDLE", "JUMP", "DUCK"]
        action_text = f"Action: {actions[self.last_action]}"
        action_surface = self.font.render(action_text, True, self.TEXT_COLOR)
        self.screen.blit(action_surface, (panel_x + 10, y_offset))
        y_offset += 25
        
        # Q-values
        self.screen.blit(self.font.render("Q-Values:", True, self.TEXT_COLOR), 
                        (panel_x + 10, y_offset))
        y_offset += 20
        
        for i, (action, q_val) in enumerate(zip(actions, self.last_q_values)):
            color = self.AI_COLOR if i == self.last_action else self.TEXT_COLOR
            q_text = f"  {action}: {q_val:.3f}"
            q_surface = self.font.render(q_text, True, color)
            self.screen.blit(q_surface, (panel_x + 10, y_offset))
            y_offset += 18
        
        # Confidence
        conf_text = f"Confidence: {self.confidence*100:.1f}%"
        conf_surface = self.font.render(conf_text, True, self.TEXT_COLOR)
        self.screen.blit(conf_surface, (panel_x + 10, y_offset))
        
        # Game stats
        stats_y = self.HEIGHT - 100
        stats = [
            f"Survival Time: {self.survival_time}",
            f"Current Speed: {self.current_bar_speed:.3f}",
            f"Beams on Screen: {len(self.beams)}",
            f"Player Y: {self.player_y:.2f}"
        ]
        
        for i, stat in enumerate(stats):
            stat_surface = self.font.render(stat, True, self.TEXT_COLOR)
            self.screen.blit(stat_surface, (10, stats_y + i * 22))
    
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
        
        try:
            # Clear screen
            self.screen.fill(self.SKY_COLOR)
            
            # Draw 3D scene
            self.draw_platform()
            
            # Draw player cube
            current_height = self.CUBE_HEIGHT_DUCK if self.ducking else self.CUBE_HEIGHT_NORMAL
            self.draw_cube(self.player_x, self.player_y, self.player_z, 
                          self.CUBE_WIDTH, self.CUBE_DEPTH, current_height)
            
            # Draw beams
            for beam in self.beams:
                self.draw_beam(beam)
            
            # Draw UI
            self.draw_ui()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(30)
            
        except Exception as e:
            print(f"Render error: {e}")
            return False
            
        return True
    
    def update_ai_info(self, action, q_values=None):
        """Update AI information for display with proper confidence"""
        self.last_action = action
        if q_values is not None:
            self.last_q_values = q_values.tolist()
            
            # FIXED: Proper confidence calculation using softmax
            probabilities = tf.nn.softmax(q_values).numpy()
            self.confidence = float(np.max(probabilities))  # Now 0-1 range
    
    def close(self):
        """Clean up pygame resources"""
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False


class SimpleTrainer:
    """Improved trainer for creating a more stable AI agent"""
    
    def __init__(self):
        self.train_env = create_platformer_env(max_steps=3000)
        self.batch_size = 512  # Even larger batch for better gradient estimates
        
        # Create preprocessing layers with stronger regularization
        preprocessing_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(
                512, activation='elu',
                kernel_regularizer=tf.keras.regularizers.l2(0.005),  # Reduced regularization
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
        
        # Network with preprocessing layers
        self.q_net = q_network.QNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            preprocessing_layers=preprocessing_layers,
            activation_fn=tf.keras.activations.elu
        )

        # Faster initial learning rate with cyclical decay
        initial_learning_rate = 2e-4  # Higher initial learning rate
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_learning_rate,
            clipnorm=0.5,  # Less aggressive clipping
            epsilon=1e-5
        )
        
        self.train_step_counter = tf.Variable(0)
        
        # Faster epsilon decay
        self.initial_epsilon = 1.0
        self.min_epsilon = 0.05  # Lower minimum for more exploitation
        self.total_decay_steps = 20000  # Faster decay
        
        # Create target network
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
            target_update_period=50,  # More frequent target updates
            target_update_tau=0.01,  # Faster target updates
            epsilon_greedy=self.initial_epsilon,
            gamma=0.99,  # Higher discount for better long-term planning
            reward_scale_factor=0.1,  # Scale rewards less
            gradient_clipping=True,
            debug_summaries=True,
            summarize_grads_and_vars=True
        )
        
        self.agent.initialize()
        
        # Replay buffer with prioritized sampling
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=100000  # Smaller buffer for more recent experiences
        )
        
        self.random_policy = random_tf_policy.RandomTFPolicy(
            self.train_env.time_step_spec(),
            self.train_env.action_spec()
        )
    
    def collect_step(self, environment, policy, epsilon=None):
        """Collect a single step of experience with optional epsilon override"""
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
        """Update epsilon using linear decay to 0.1"""
        # Ensure we don't exceed total decay steps
        step = min(training_step, self.total_decay_steps)
        
        # Linear decay from 1.0 to 0.1
        progress = step / self.total_decay_steps
        new_epsilon = 1.0 - (0.9 * progress)  # Will go from 1.0 to 0.1 linearly
        
        self.agent._epsilon_greedy = new_epsilon
        return new_epsilon
    
    def evaluate_agent(self, num_episodes=20):
        """Evaluate agent performance with epsilon=0 (no exploration)"""
        total_return = 0.0
        original_epsilon = self.agent._epsilon_greedy
        self.agent._epsilon_greedy = 0.0  # No exploration during evaluation
        
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
            self.agent._epsilon_greedy = original_epsilon  # Restore original epsilon
        
        avg_return = total_return / num_episodes
        return avg_return
    
    def train_basic_agent(self):
        print("Training AI agent with improved parameters...")
        
        env = self.train_env
        env.reset()
        
        # Collect more initial experience with full exploration
        print("Phase 1: Collecting random experience...")
        for i in range(100000):  # Reduced from 100000 to 50000
            self.collect_step(env, self.agent.collect_policy, epsilon=1.0)
            if i % 1000 == 0:
                print(f"Random experience: {i}/100000")
        
        print("Phase 2: Training neural network...")
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size,
            num_steps=2
        ).prefetch(3)
        
        iterator = iter(dataset)
        
        best_loss = float('inf')
        best_return = float('-inf')
        patience = 20
        patience_counter = 0
        moving_avg_loss = []
        moving_avg_return = []
        
        # Training with more frequent evaluation
        for i in range(20000):
            try:
                experience, _ = next(iterator)
                train_step = self.agent.train(experience)
                loss = float(train_step.loss.numpy())
                
                # Update epsilon
                current_epsilon = self.update_epsilon(i)
                
                # Track moving average of loss
                moving_avg_loss.append(loss)
                if len(moving_avg_loss) > 100:
                    moving_avg_loss.pop(0)
                avg_loss = sum(moving_avg_loss) / len(moving_avg_loss)
                
                if i % 500 == 0:  # Evaluate twice as often
                    # Evaluate agent with more episodes
                    avg_return = float(self.evaluate_agent(num_episodes=20))
                    
                    # Track moving average of returns with larger window
                    moving_avg_return.append(avg_return)
                    if len(moving_avg_return) > 10:  # Doubled window size
                        moving_avg_return.pop(0)
                    smoothed_return = sum(moving_avg_return) / len(moving_avg_return)
                    
                    print(f"Step {i}, Loss: {avg_loss:.4f}, Epsilon: {current_epsilon:.4f}, "
                          f"Avg Return: {smoothed_return:.2f}")
                    
                    # Early stopping based on smoothed performance
                    if smoothed_return > best_return:
                        best_return = smoothed_return
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        if smoothed_return > 200:
                            print("Early stopping triggered with good performance!")
                            break
                        else:
                            print("Resetting patience - continuing training...")
                            patience_counter = 0
                    
                    # Adjust learning rate if needed
                    if avg_loss > best_loss * 1.2:
                        print("WARNING: Training unstable, reducing learning rate...")
                        if isinstance(self.optimizer.learning_rate, tf.keras.optimizers.schedules.ExponentialDecay):
                            current_lr = self.optimizer.learning_rate(self.train_step_counter)
                        else:
                            current_lr = self.optimizer.learning_rate
                        new_lr = tf.keras.optimizers.schedules.ExponentialDecay(
                            current_lr * 0.8,
                            decay_steps=5000,
                            decay_rate=0.95,
                            staircase=True
                        )
                        self.optimizer.learning_rate = new_lr
                        patience_counter = 0
                    
                    best_loss = min(best_loss, avg_loss)
                
            except Exception as e:
                print(f"Training error at step {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                break
        
        print("Training complete!")
        return self.agent



def watch_trained_ai():
    """Watch a trained AI play the game"""
    print("=== AI DEMO ===")
    print("Training AI agent...")
    
    try:
        # Train the agent
        trainer = SimpleTrainer()
        trained_agent = trainer.train_basic_agent()
        
        # Show it playing
        print("Now watch the trained AI play!")
        visual_env = VisualPlatformerEnv(max_steps=5000, render_mode='human')
        visual_tf_env = tf_py_environment.TFPyEnvironment(visual_env)
        
        time_step = visual_tf_env.reset()
        episode_return = 0
        step_count = 0
        
        while True:
            # Get Q-values for display
            q_values = trainer.q_net(time_step.observation)[0].numpy()[0]
            action_step = trained_agent.policy.action(time_step)
            
            # Update display
            visual_env.update_ai_info(action_step.action.numpy()[0], q_values)
            
            time_step = visual_tf_env.step(action_step.action)
            episode_return += time_step.reward.numpy()[0]
            step_count += 1
            
            # Render every other frame for smooth performance
            if step_count % 2 == 0:
                if not visual_env.render_frame():
                    break
        
        # visual_env.close()
        print(f"Episode score: {episode_return:.1f}")
        
    except Exception as e:
        print(f"Error in demo: {e}")


def compare_random_vs_trained():
    """Compare random AI vs trained AI"""
    print("=== COMPARISON MODE ===")
    print("First you'll see random AI, then trained AI")
    
    try:
        # Show random AI first
        print("Showing RANDOM AI (plays randomly)")
        visual_env = VisualPlatformerEnv(max_steps=1000, render_mode='human')
        visual_tf_env = tf_py_environment.TFPyEnvironment(visual_env)
        random_policy = random_tf_policy.RandomTFPolicy(
            visual_tf_env.time_step_spec(), visual_tf_env.action_spec()
        )
        
        time_step = visual_tf_env.reset()
        step_count = 0
        random_score = 0
        
        while not time_step.is_last() and not visual_env.should_quit:
            action_step = random_policy.action(time_step)
            visual_env.update_ai_info(action_step.action.numpy()[0])
            time_step = visual_tf_env.step(action_step.action)
            random_score += time_step.reward.numpy()[0]
            step_count += 1
            
            if step_count % 2 == 0:
                if not visual_env.render_frame():
                    break
        
        visual_env.close()
        print(f"Random AI score: {random_score:.1f}")
        
        if not visual_env.should_quit:
            input("Press Enter to see the TRAINED AI...")
            watch_trained_ai()
    
    except Exception as e:
        print(f"Error in comparison: {e}")


def test_visual_environment():
    """Test the visual environment"""
    print("=== TESTING VISUAL ENVIRONMENT ===")
    print("Testing pygame and 3D rendering...")
    
    try:
        visual_env = VisualPlatformerEnv(max_steps=500, render_mode='human')
        visual_tf_env = tf_py_environment.TFPyEnvironment(visual_env)
        
        if not visual_env.pygame_initialized:
            print("ERROR: Pygame failed to initialize!")
            return False
        
        print("Pygame initialized successfully!")
        print("You should see a 3D platformer with a red cube moving randomly")
        print("Close window or press ESC to exit")
        
        time_step = visual_tf_env.reset()
        step_count = 0
        
        while step_count < 1000 and not visual_env.should_quit:
            action = tf.random.uniform([], 0, 3, dtype=tf.int32)
            visual_env.update_ai_info(action.numpy())
            time_step = visual_tf_env.step(action)
            step_count += 1
            
            if step_count % 2 == 0:
                if not visual_env.render_frame():
                    break
            
            if time_step.is_last():
                time_step = visual_tf_env.reset()
                print(f"Episode ended at step {step_count}, starting new episode")
        
        visual_env.close()
        print("Visual test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Visual test failed: {e}")
        return False


if __name__ == "__main__":
    print("\nChoose what you want to see:")
    print("1. Watch trained AI play")
    print("2. Compare random vs trained AI")
    print("3. Test visual environment")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        watch_trained_ai()
    elif choice == "2":
        compare_random_vs_trained()
    elif choice == "3":
        success = test_visual_environment()
        if success:
            print("\nGreat! The visual environment works. You can now try option 1.")
        else:
            print("\nThere's an issue with the visual environment. Check your pygame installation.")
    else:
        print("Invalid choice!")