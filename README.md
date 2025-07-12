# Spooky Spikes RL

This project is a 3D recreation of the Spooky Spikes minigame from Pummel Party, built in pygame with a custom reinforcement learning environment.

After consistently getting last place in the original minigame, I built a similar environment to train an AI to play optimally. The environment simulates the core mechanics of Spooky Spikes, including jumping, ducking, and dodging moving beams.

## Features

- **3D Rendering in Pygame:**
  - Perspective projection for 3D look using only 2D polygons
  - Depth sorting and visual cues for gameplay
- **Reinforcement Learning Environment:**
  - Built with TensorFlow, Keras, and TF-Agents
  - Reward function that promotes smart timing, sustained ducking, and optimal movement
  - Dynamic difficulty scaling with increasing beam speed and spawn rate
- **AI Agent:**
  - Deep Q-Network (DQN) learns to jump and duck at the right times
  - Observes player state, beam positions, and environment speed
  - Outperforms human players after training
- **Visualization:**
  - Real-time 3D game view
  - AI status panel showing action, Q-values, and confidence

## How It Works

- The environment is defined in `platformer_env.py` as a subclass of `py_environment.PyEnvironment`.
- 3D rendering is achieved by projecting 3D coordinates to 2D using a perspective formula and drawing polygons in the correct order.
- The reward function is designed to:
  - penalize collisions
  - Reward survival, speed, and especially maintaining ducking under high beams
  - Penalize early unducking and random actions

## Setup

1. **Install dependencies:**
   ```bash
   python 3.10 or lower is required
   pip install pygame tensorflow tf-agents numpy
   ```
2. **Run the environment:**
   ```bash
   python trainer.py
   ```
3. **Choose an option:**
   - Watch trained AI play
   - Compare random vs trained AI
   - Test the 3D environment
