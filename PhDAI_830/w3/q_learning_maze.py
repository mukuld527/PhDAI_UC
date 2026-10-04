import numpy as np
import random
from collections import defaultdict

''' ===================================================
Mukul Kumar Dhali
University of Cumberland, School of Computer and Information Sciences,
PhDAI 830: Applied Machine Intelligence and Reinforcement Learning,
Professor Dr. Soamar Homsi​
September 13, 2026
=================================================== '''  

# -----------------------------
# Part 1: Maze Environment
# -----------------------------

class GridMazeEnv:
    """
    Simple grid-based maze:
    - 0 = free cell
    - 1 = wall / blocked
    - S = start
    - G = goal
    Rewards:
      +1 for reaching goal
      -1 for invalid move (into wall or outside grid)
      -0.01 per step to encourage shorter paths
    Episode ends when goal is reached or max_steps exceeded.
    """

    def __init__(self):
        # The maze layout 
        # 0 = free, 
        # 1 = wall
        # S at (0,0), 
        # G at (3,4)

        self.grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ])

        self.n_rows, self.n_cols = self.grid.shape
        self.start_state = (0, 0)
        self.goal_state = (3, 4)

        # Actions: 0=up, 1=down, 2=left, 3=right
        
        self.actions = [0, 1, 2, 3]
        self.action_map = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }

        self.max_steps = 100
        self.reset()

    def reset(self):
        # Reset environment to start state.
        self.state = self.start_state
        self.steps = 0
        return self.state

    def step(self, action):
        """
        Take an action:
        - If move is invalid (wall or outside), stay in place and give penalty.
        - If reach goal, give positive reward and done=True.
        - Otherwise small negative step cost.
        """
        self.steps += 1
        r, c = self.state
        dr, dc = self.action_map[action]
        nr, nc = r + dr, c + dc

        # Check bounds
        if nr < 0 or nr >= self.n_rows or nc < 0 or nc >= self.n_cols:
            # Invalid move: out of bounds
            reward = -1.0
            next_state = self.state
            done = False
        # Check wall
        elif self.grid[nr, nc] == 1:
            # Invalid move: wall
            reward = -1.0
            next_state = self.state
            done = False
        else:
            # Valid move
            next_state = (nr, nc)
            if next_state == self.goal_state:
                reward = 1.0
                done = True
            else:
                reward = -0.01
                done = False

        self.state = next_state

        # Optional: terminate if too many steps
        if self.steps >= self.max_steps and not done:
            done = True

        return next_state, reward, done

    def get_state_space(self):
        """Return list of all valid states (non-wall cells)."""
        states = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r, c] == 0:
                    states.append((r, c))
        return states

# -----------------------------
# Part 2: Q-Learning Agent
# -----------------------------

class QLearningAgent:
    # Tabular Q-learning agent with epsilon-greedy exploration.
    # Q-table indexed by (state, action).

    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.env = env
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table: dict[(state), action] -> value
        self.Q = defaultdict(lambda: np.zeros(len(self.env.actions)))

    def choose_action(self, state):
        # Epsilon-greedy action selection:
        #  - With probability epsilon: random action
        #  - Otherwise: argmax over Q-values

        if random.random() < self.epsilon:
            return random.choice(self.env.actions)
        else:
            q_values = self.Q[state]
            return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done):
        #  Q-learning update rule:
        #    Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]

        q_sa = self.Q[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        self.Q[state][action] = q_sa + self.alpha * (target - q_sa)

    def decay_epsilon(self):
        # Decay epsilon after each episode.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_policy(self):
        # Return a policy mapping from state -> best action index.

        policy = {}
        for s in self.env.get_state_space():
            policy[s] = int(np.argmax(self.Q[s]))
        return policy

# ----------------------------------------------
# Part 3: Training the agent and Tracking
# ----------------------------------------------

def train_agent(env, agent, num_episodes=500):
    # Train Q-learning agent in the maze.
    # Track episode rewards and optionally convergence trends.

    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        # Optional: print progress every N episodes
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, total reward = {total_reward:.3f}, epsilon = {agent.epsilon:.3f}")

    return episode_rewards

def print_learned_policy(env, agent):
    #   Print the learned greedy policy as arrows on the grid.

    policy = agent.greedy_policy()
    arrow_map = {0: "^", 1: "v", 2: "<", 3: ">"}

    print("\nLearned greedy policy (arrows show best action per state):")
    for r in range(env.n_rows):
        row_str = ""
        for c in range(env.n_cols):
            if (r, c) == env.goal_state:
                row_str += " G "
            elif (r, c) == env.start_state:
                row_str += " S "
            elif env.grid[r, c] == 1:
                row_str += " # "
            else:
                a = policy.get((r, c), None)
                if a is None:
                    row_str += " . "
                else:
                    row_str += f" {arrow_map[a]} "
        print(row_str)

def main():
    # Base environment
    env = GridMazeEnv()

    # Example hyperparameters (This will vary these in Part 4)
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    num_episodes = 500

    agent = QLearningAgent(env,
                           alpha=alpha,
                           gamma=gamma,
                           epsilon=epsilon,
                           epsilon_min=epsilon_min,
                           epsilon_decay=epsilon_decay)

    print("Training agent...")
    rewards = train_agent(env, agent, num_episodes=num_episodes)

    print_learned_policy(env, agent)

    # Simple summary of learning progress
    print("\nEpisode rewards (first 10):", rewards[:10])
    print("Episode rewards (last 10):", rewards[-10:])
    print(f"\nAverage reward over all episodes: {np.mean(rewards):.3f}")

    # Matplotlib plots can be added here (episode reward vs episode index)
    # to show convergence trends.

if __name__ == "__main__":
    main()
