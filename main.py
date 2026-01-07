"""
abstract_surgery_rl.py
Abstract reinforcement learning demo for simulated "approach + symbolic cut" task.
This is a toy research simulation only — NOT medical/surgical guidance.
"""
print("Program is running...")

import os
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ---------------------------
# 1) Create an abstract Gymnasium environment
# ---------------------------
class AbstractSurgeryEnv(gym.Env):
    """
    Simple 2D environment:
      - Observation: [tool_x, tool_y, target_x, target_y]
      - Action (Discrete 5): 0:left,1:right,2:up,3:down,4:symbolic_cut
      - Episode ends when symbolic_cut performed or max_steps
      - Reward: -distance each step; large +reward if symbolic_cut performed when tool within threshold
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=80, reach_threshold=0.05):
        super().__init__()
        self.max_steps = max_steps
        self.step_count = 0
        self.reach_threshold = reach_threshold

        # Define observation space: 4 floats in [0,1]
        low = np.array([0, 0, 0, 0], dtype=np.float32)
        high = np.array([1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Discrete actions: left, right, up, down, symbolic_cut
        self.action_space = spaces.Discrete(5)

        # State variables
        self.tool_pos = np.array([0.5, 0.1], dtype=np.float32)
        self.target_pos = np.array([0.5, 0.9], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tool_pos = np.clip(np.random.rand(2) * 0.6 + 0.2, 0.0, 1.0)
        self.target_pos = np.clip(np.random.rand(2) * 0.4 + 0.5, 0.0, 1.0)
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.tool_pos, self.target_pos]).astype(np.float32)

    def step(self, action):
        self.step_count += 1
        done = False
        info = {}

        # movement delta
        delta = 0.05
        if action == 0:   # left
            self.tool_pos[0] = max(0.0, self.tool_pos[0] - delta)
        elif action == 1: # right
            self.tool_pos[0] = min(1.0, self.tool_pos[0] + delta)
        elif action == 2: # up
            self.tool_pos[1] = min(1.0, self.tool_pos[1] + delta)
        elif action == 3: # down
            self.tool_pos[1] = max(0.0, self.tool_pos[1] - delta)
        elif action == 4: # symbolic cut
            dist = np.linalg.norm(self.tool_pos - self.target_pos)
            if dist <= self.reach_threshold:
                reward = 100.0
                done = True
                info["success"] = True
            else:
                reward = -50.0
                done = True
                info["success"] = False

            return self._get_obs(), reward, done, False, info

        # If not a symbolic cut, compute reward
        dist = np.linalg.norm(self.tool_pos - self.target_pos)
        reward = -dist
        
        if self.step_count >= self.max_steps:
            done = True
            info["timeout"] = True

        return self._get_obs(), reward, done, False, info

    def render(self):
        print(f"Tool: {self.tool_pos}, Target: {self.target_pos}")

    def close(self):
        pass

# ---------------------------
# 2) Train the agent (PPO)
# ---------------------------
def train_and_evaluate():
    env = DummyVecEnv([lambda: AbstractSurgeryEnv(max_steps=80)])
    model = PPO("MlpPolicy", env, verbose=1, n_steps=256)

    timesteps = 20_000_000   # reduce for quick testing
    model.learn(total_timesteps=timesteps)

    os.makedirs("models", exist_ok=True)
    model_path = "models/ppo_abstract_surgery"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # ---------------------------
    # Evaluation
    # ---------------------------
    eval_env = AbstractSurgeryEnv(max_steps=80)
    n_eval_episodes = 50
    successes = 0
    episode_rewards = []

    for ep in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = eval_env.step(int(action))
            total_r += reward
            if done and "success" in info:
                if info["success"]:
                    successes += 1
        episode_rewards.append(total_r)

    success_rate = successes / n_eval_episodes
    print(f"Evaluation success rate: {success_rate:.2f}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")

    # Plot rewards
    plt.figure()
    plt.plot(episode_rewards, marker='.')
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()

# ---------------------------
# Run it!
# ---------------------------
if __name__ == "__main__":
    train_and_evaluate()
