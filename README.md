# Spooky Spikes RL

This project is a 3D recreation of the Spooky Spikes minigame from Pummel Party, built in pygame with a custom reinforcement learning environment. The goal is to train an AI agent to master the game and outperform human players.

## Motivation

After consistently getting last place in the original minigame, this project was created to build a similar environment and train an AI to play optimally. The environment simulates the core mechanics of Spooky Spikes, including jumping, ducking, and dodging moving beams.

## Features

- **3D Rendering in Pygame:**
  - Custom perspective projection for a 3D look using only 2D polygons
  - Depth sorting and visual cues for immersive gameplay
- **Reinforcement Learning Environment:**
  - Built with TensorFlow, Keras, and TF-Agents
  - Custom reward function encourages smart timing, sustained ducking, and optimal movement
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
- 3D rendering is achieved by projecting 3D coordinates to 2D using a simple perspective formula and drawing polygons in the correct order.
- The reward function is carefully designed to:
  - Strongly penalize collisions
  - Reward survival, speed, and especially maintaining ducking under high beams
  - Penalize premature unducking and random actions
- The AI is trained using TensorFlow and Keras, leveraging TF-Agents for RL algorithms.

## Setup

1. **Install dependencies:**
   ```bash
   pip install pygame tensorflow tf-agents numpy
   ```
2. **Run the environment or training script:**
   ```bash
   python trainer.py
   ```
3. **Choose an option:**
   - Watch trained AI play
   - Compare random vs trained AI
   - Test the 3D environment

## File Overview

- `platformer_env.py` - Main environment and reward logic
- `trainer.py` - Training loop, visualization, and agent management

## Controls (for manual play, if implemented)

- Jump: [Space]
- Duck: [Down Arrow]
- Idle: [No input]

## Notes

- The AI learns to commit to ducking under high beams and not unduck too early, thanks to the reward structure.
- The 3D effect is achieved without a 3D engine, using only math and pygame's polygon drawing.
- The project is a fun way to explore RL, game AI, and custom rendering techniques.

## License

MIT
