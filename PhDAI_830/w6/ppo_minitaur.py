import gymnasium as gym
import pybullet_envs_gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import matplotlib.pyplot as plt
import os
import logging

''' ============================================================================
                    PPO for Robotics Task Project

                    
                    Mukul Kumar Dhali
    University of Cumberland, School of Computer and Information Sciences,
    PhDAI 830: Applied Machine Intelligence and Reinforcement Learning,
    Professor Dr. Soamar Homsi​
    October 04, 2026
================================================================================ '''  

# pip install pybullet-envs-gymnasium gym==0.26.2 gym-notices pybullet torch matplotlib numpy==1.26.4


logging.basicConfig(
    level=logging.INFO,                      # INFO level logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ppo_minitaur_training.log"),  # save logs to file
        logging.StreamHandler()                   # also print to console
    ]
)
logger = logging.getLogger(__name__)

# ==========================
# Hyperparameters
# ==========================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ENV_ID = "AntBulletEnv-v0"
SEED = 1

GAMMA = 0.99          # discount factor
LAMBDA = 0.95         # GAE lambda
CLIP_EPS = 0.2        # PPO clip range
LR = 3e-4             # learning rate
BATCH_SIZE = 4096     # steps per rollout
MINI_BATCH_SIZE = 256
PPO_EPOCHS = 10
MAX_STEPS = 500_000   # total environment steps
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# Actor-Critic Network
# ==========================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        hidden_size = 128
        logger.info(f"Initializing ActorCritic with obs_dim={obs_dim}, act_dim={act_dim}, hidden_size={hidden_size}")

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Actor head: mean of Gaussian
        self.mu_head = nn.Linear(hidden_size, act_dim)
        # Log std as a parameter (state-independent)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic head: state value
        self.v_head = nn.Linear(hidden_size, 1)
        #logger.info(f"ActorCritic network initialized. Total parameters: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x):
        x = self.shared(x)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        v = self.v_head(x)
        #logger.info(f"Forward pass: mu={mu.detach().cpu().numpy()}, std={std.detach().cpu().numpy()}, v={v.detach().cpu().numpy()}")
        return mu, std, v

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        mu, std, v = self.forward(obs_t)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        #logger.info(f"Action taken: {action.cpu().numpy()}, Log prob: {log_prob.detach().cpu().numpy()}, Value: {v.detach().cpu().numpy()}")
        return action.cpu().numpy(), log_prob.detach().cpu().numpy(), v.detach().cpu().numpy()

    def evaluate_actions(self, obs, actions):
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        #logger.info(f"Evaluating actions: Log probs: {log_probs.detach().cpu().numpy()}, Entropy: {entropy.detach().cpu().numpy()}, Values: {v.squeeze(-1).detach().cpu().numpy()}")
        return log_probs, entropy, v.squeeze(-1)


# ==========================
# GAE computation
# ==========================
def compute_gae(rewards, values, dones, gamma, lam):
    """
    rewards: np.array [T]
    values: np.array [T+1] (bootstrap value for last state)
    dones: np.array [T]
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    #logger.info(f"Computed GAE: Advantages: {advantages}, Returns: {returns}")
    return advantages, returns


# ==========================
# PPO update
# ==========================
def ppo_update(model, optimizer, obs, actions, old_log_probs, returns, advantages):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
    actions_t = torch.as_tensor(actions, dtype=torch.float32, device=DEVICE)
    old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=DEVICE)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=DEVICE)

    # Normalize advantages
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    num_samples = obs_t.shape[0]
    idxs = np.arange(num_samples)

    policy_losses = []
    value_losses = []
    entropies = []

    for _ in range(PPO_EPOCHS):
        np.random.shuffle(idxs)
        for start in range(0, num_samples, MINI_BATCH_SIZE):
            end = start + MINI_BATCH_SIZE
            mb_idx = idxs[start:end]

            mb_obs = obs_t[mb_idx]
            mb_actions = actions_t[mb_idx]
            mb_old_log_probs = old_log_probs_t[mb_idx]
            mb_returns = returns_t[mb_idx]
            mb_advantages = advantages_t[mb_idx]

            log_probs, entropy, values = model.evaluate_actions(mb_obs, mb_actions)

            # Policy loss (clipped surrogate)
            ratio = torch.exp(log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (MSE)
            value_loss = (mb_returns - values).pow(2).mean()

            # Total loss
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.mean().item())

    #logger.info(f"Training update completed. Policy loss: {np.mean(policy_losses)}, Value loss: {np.mean(value_losses)}, Entropy: {np.mean(entropies)}")
    return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)


# ==========================
# Training loop
# ==========================
def train():
    # env = gym.make(ENV_ID)
    env = gym.make(ENV_ID, render_mode=None) # Set render_mode to None for headless training
    # env.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    #logger.info(f"Environment {ENV_ID} initialized with seed {SEED}. Observation space: {env.observation_space}, Action space: {env.action_space}")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    episode_rewards = []
    rolling_returns = deque(maxlen=100)

    total_steps = 0
    episode = 0

    # For logging losses
    policy_loss_log = []
    value_loss_log = []
    entropy_log = []

    obs, info = env.reset(seed=SEED)
    #logger.info(f"Initial observation: {obs}")
    while total_steps < MAX_STEPS:
        # Rollout storage
        obs_buf = []
        actions_buf = []
        log_probs_buf = []
        rewards_buf = []
        dones_buf = []
        values_buf = []

        steps_collected = 0
        while steps_collected < BATCH_SIZE:
            action, log_prob, value = model.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            obs_buf.append(obs)
            actions_buf.append(action)
            log_probs_buf.append(log_prob)
            rewards_buf.append(reward)
            dones_buf.append(float(done))
            values_buf.append(value)

            obs = next_obs
            steps_collected += 1
            total_steps += 1

            if done:
                episode += 1
                rolling_returns.append(sum(rewards_buf))
                episode_rewards.append(rolling_returns[-1])
                #logger.info(f"Episode {episode} completed. Return: {rolling_returns[-1]}")
                obs, info = env.reset(seed=SEED)

        # Bootstrap value for last state
        last_value = model.act(obs)[2]
        values_buf.append(last_value)

        rewards_np = np.array(rewards_buf, dtype=np.float32)
        dones_np = np.array(dones_buf, dtype=np.float32)
        values_np = np.array(values_buf, dtype=np.float32).squeeze(-1)

        advantages, returns = compute_gae(
            rewards_np, values_np, dones_np, GAMMA, LAMBDA
        )

        obs_arr = np.array(obs_buf, dtype=np.float32)
        actions_arr = np.array(actions_buf, dtype=np.float32)
        old_log_probs_arr = np.array(log_probs_buf, dtype=np.float32)

        p_loss, v_loss, ent = ppo_update(
            model, optimizer, obs_arr, actions_arr, old_log_probs_arr, returns, advantages
        )

        policy_loss_log.append(p_loss)
        value_loss_log.append(v_loss)
        entropy_log.append(ent)

        if episode % 10 == 0 and len(rolling_returns) > 0:
            logger.info(f"Episode {episode} completed. Mean Return: {np.mean(rolling_returns):.2f}")

    env.close()

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Episode Rewards")

    plt.subplot(3, 1, 2)
    plt.plot(policy_loss_log)
    plt.xlabel("Update")
    plt.ylabel("Policy Loss")

    plt.subplot(3, 1, 3)
    plt.plot(value_loss_log, label="Value Loss")
    plt.plot(entropy_log, label="Entropy")
    plt.xlabel("Update")
    plt.legend()

    plt.tight_layout()
    plt.savefig("ppo_minitaur_training.png")
    plt.show()


if __name__ == "__main__":
    train()
