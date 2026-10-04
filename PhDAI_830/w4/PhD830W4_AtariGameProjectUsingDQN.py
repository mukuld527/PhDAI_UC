import gymnasium as gym
import ale_py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import logging
from collections import deque
from dataclasses import dataclass

''' ============================================================================
                    DQN on Atari Game Project

                    
                    Mukul Kumar Dhali
    University of Cumberland, School of Computer and Information Sciences,
    PhDAI 830: Applied Machine Intelligence and Reinforcement Learning,
    Professor Dr. Soamar Homsi​
    September 20, 2026
================================================================================ '''  

# Run below command to install required packages if not already installed
# pip install gymnasium ale_py torch numpy matplotlib "gymnasium[other]"


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(
    level=logging.INFO,                      # INFO level logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dqn_training.log"),  # save logs to file
        logging.StreamHandler()                   # also print to console
    ]
)
logger = logging.getLogger(__name__)


#Part 1: Atari environment setup (PyTorch + Gymnasium)
def make_atari_env(env_id: str = "ALE/Breakout-v5", seed: int = 42):
    gym.register_envs(ale_py)

    # env = gym.make(env_id, render_mode=None)
    # Disable built-in frame skipping
    env = gym.make(env_id, render_mode="human", frameskip=1)

    # Standard Atari preprocessing wrapper
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,        
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=False,  # keep uint8, normalize in network
    )

    # Stack 4 frames to provide temporal information
    env = gym.wrappers.FrameStackObservation(env, 4)

    env.action_space.seed(seed)
    return env

# Part 2: DQN agent components Convolutional Q-network
class AtariDQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        # Input: (4, 84, 84) stacked grayscale frames
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        # x: (batch, 4, 84, 84), normalize from uint8 to [0,1]
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Q-values for each action
# This architecture mirrors the classic DQN convolutional network used for Atari games.
# https://github.com/nilsleut/Deep-Q-Network-Atari-Breakout/blob/main


# Step 1. Replay buffer
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

# Epsilon-greedy policy
def select_action(state, online_net, epsilon, n_actions, device):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    with torch.no_grad():
        state = state.unsqueeze(0).to(device)  # add batch dimension
        q_values = online_net(state)
        return int(q_values.argmax(dim=1).item())
# Epsilon starts high (pure exploration) and decays over time toward a small value to favor exploitation. 
# https://github.com/nilsleut/Deep-Q-Network-Atari-Breakout/blob/main/

# Part 3: Training loop and hyperparameters
# Configuration
# Step 2: Hyperparameters
@dataclass
class DQNConfig:
    env_id: str = "ALE/Breakout-v5"
    seed: int = 42

    # Training
    total_frames: int = 1_000_000
    batch_size: int = 32
    learning_rate: float = 1e-4
    gamma: float = 0.99

    # Replay buffer
    buffer_size: int = 100_000
    min_replay_size: int = 10_000

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_frames: int = 500_000

    # Target network
    target_update_freq: int = 1_000

    # Logging
    log_interval: int = 10_000

# These values are in line with common Atari DQN configurations, scaled down for a course project. 
# Training loop (core skeleton)


def train(config: DQNConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_atari_env(config.env_id, config.seed)
    n_actions = env.action_space.n

    online_net = AtariDQN(n_actions).to(device)
    target_net = AtariDQN(n_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=config.learning_rate)
    replay_buffer = ReplayBuffer(config.buffer_size)

    # Initialize environment
    state, _ = env.reset(seed=config.seed)
    state = torch.tensor(np.array(state), dtype=torch.uint8)  # (4,84,84)

    episode_reward = 0
    rewards_history = []
    losses_history = []

    epsilon = config.epsilon_start
    epsilon_decay = (config.epsilon_start - config.epsilon_end) / config.epsilon_decay_frames

    for frame_idx in range(1, config.total_frames + 1):
        # Select action
        action = select_action(state, online_net, epsilon, n_actions, device)

        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action)

        # 50 ms delay → slow motion , 0.01 → fast , 0.05 → slow  , 0.1 → very slow
        #time.sleep(0.01)

        logger.info(
            f"Frame {frame_idx} | "
            f"Epsilon {epsilon:.3f} | "
            f"Replay buffer size: {len(replay_buffer)}"
        )

        done = terminated or truncated
        next_state_t = torch.tensor(np.array(next_state), dtype=torch.uint8)

        replay_buffer.push(state, action, reward, next_state_t, done)
        state = next_state_t
        episode_reward += reward

        # Handle episode end
        if done:
            rewards_history.append(episode_reward)
            state, _ = env.reset()
            state = torch.tensor(np.array(state), dtype=torch.uint8)
            episode_reward = 0
            logger.info(f"Episode finished with reward: {episode_reward}")

        # Decay epsilon
        if epsilon > config.epsilon_end:
            epsilon -= epsilon_decay
            epsilon = max(epsilon, config.epsilon_end)

        # Start learning once buffer is warm
        if len(replay_buffer) < config.min_replay_size:
            if frame_idx % 1000 == 0:
                 logger.info(f"Warm-up… replay buffer size: {len(replay_buffer)}")
            continue

        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(config.batch_size)
        states = states.to(device)
        next_states = next_states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)

        # Compute current Q-values
        q_values = online_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = target_net(next_states).max(dim=1)[0]
            targets = rewards + config.gamma * next_q_values * (1 - dones)

        loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses_history.append(loss.item())

        # Target network update
        if frame_idx % config.target_update_freq == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Logging
        if frame_idx % config.log_interval == 0:
            mean_reward = np.mean(rewards_history[-100:]) if rewards_history else 0.0
            mean_loss = np.mean(losses_history[-100:]) if losses_history else 0.0
            logging.debug(
                f"Frame {frame_idx} | "
                f"Epsilon {epsilon:.3f} | "
                f"Mean Reward (last 100): {mean_reward:.2f} | "
                f"Mean Loss (last 100): {mean_loss:.4f}"
            )
            logger.debug(f"Loss: {loss.item():.4f}")

    logger.debug(f"Frame {frame_idx}, Action {action}, Reward {reward}, Done {done}")
    env.close()
    return rewards_history, losses_history
# This loop covers: experience collection, replay sampling, TD target computation, target network updates, epsilon decay, and basic logging for rewards and loss

# Part 4: Results, plots, and evaluation
def plot_results(rewards_history, losses_history):
    logger.debug(f"Inside the Plot Results function. Rewards history length: {len(rewards_history)}, Losses history length: {len(losses_history)}")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    logger.debug(f"Plotting Training Loss. Rewards history length: {len(rewards_history)}, Losses history length: {len(losses_history)}")
    plt.subplot(1, 2, 2)
    plt.plot(losses_history)
    plt.title("Training Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")

    logger.debug(f"Saving training curves. Rewards history length: {len(rewards_history)}, Losses history length: {len(losses_history)}")
    plt.tight_layout()
    plt.savefig("dqn_breakout_training_curves.png")
    plt.show()


if __name__ == "__main__":
    config = DQNConfig(
        total_frames=20_000,      # smaller for testing
        min_replay_size=1_000,    # faster warm-up
        log_interval=1_000        # more frequent logs
    )

    rewards_history, losses_history = train(config)
    plot_results(rewards_history, losses_history)
