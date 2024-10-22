# Define modularized code structure for RL API import gym
import numpy as np
from collections import deque
import gym
import random
import os
import json
from flask import Flask, request, jsonify


# Environment Setup
class Environment:
    def __init__(self, env_name='LunarLander-v2', seed=42):
        self.env = gym.make(env_name)
        self.env.seed(seed)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()

class Tester:
    def __init__(self, env_manager, agent):
        self.env_manager = env_manager
        self.agent = agent
    
    def run_episode(self, render=False):
        state = self.env_manager.reset()
        total_reward = 0
        done = False
        while not done:
            if render:
                self.env_manager.render()
            action = self.agent.select_action(state)
            next_state, reward, done, _ = self.env_manager.step(action)
            total_reward += reward
            state = next_state
        return total_reward
    
    def evaluate(self, episodes=10, render=False):
        rewards = [self.run_episode(render) for _ in range(episodes)]
        avg_reward = np.mean(rewards)
        print(f"Average Reward: {avg_reward}")
        return avg_reward

# Training Result Management
class TrainingLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logs = []
    
    def log(self, episode, reward, loss=None):
        log_entry = {"episode": episode, "reward": reward, "loss": loss}
        self.logs.append(log_entry)
        print(f"Episode: {episode}, Reward: {reward}, Loss: {loss if loss else 'N/A'}")
    
    def save(self, filename='training_log.json'):
        log_path = os.path.join(self.log_dir, filename)
        with open(log_path, 'w') as f:
            json.dump(self.logs, f)
        print(f"Logs saved to {log_path}")
    
    def load(self, filename='training_log.json'):
        log_path = os.path.join(self.log_dir, filename)
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                self.logs = json.load(f)
            print(f"Logs loaded from {log_path}")
        else:
            print(f"No log file found at {log_path}")

# Server Integration using Flask
class RLServer:
    def __init__(self, env_manager, agent, logger, host='0.0.0.0', port=5000):
        self.env_manager = env_manager
        self.agent = agent
        self.logger = logger
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/train', methods=['POST'])
        def train():
            episodes = int(request.json.get('episodes', 100))
            for episode in range(episodes):
                state = self.env_manager.reset()
                total_reward = 0
                done = False
                while not done:
                    action = self.agent.select_action(state)
                    next_state, reward, done, _ = self.env_manager.step(action)
                    self.agent.update(state, reward, next_state, action)
                    total_reward += reward
                    state = next_state
                self.logger.log(episode, total_reward)
            return jsonify({"message": f"Training completed for {episodes} episodes", "logs": self.logger.logs})

        @self.app.route('/evaluate', methods=['GET'])
        def evaluate():
            episodes = int(request.args.get('episodes', 10))
            avg_reward = self.logger.evaluate(episodes)
            return jsonify({"average_reward": avg_reward})

        @self.app.route('/logs', methods=['GET'])
        def get_logs():
            return jsonify(self.logger.logs)

    def run(self):
        self.app.run(host=self.host, port=self.port)

# Module 8: Fix/Utility Functions
def save_model(agent, path='model.pth'):
    torch.save(agent.q_net.state_dict(), path)
    print(f"Model saved at {path}")

def load_model(agent, path='model.pth'):
    agent.q_net.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")

def load_config(config_path='config.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    else:
        print(f"No configuration file found at {config_path}")
        return {}

def save_config(config, config_path='config.json'):
    with open(config_path, 'w') as f:
        json.dump(config, f)
    print(f"Configuration saved to {config_path}")

# Example usage:
# env_manager = EnvironmentManager()
# agent = DQNAgent(env_manager.state_dim, env_manager.action_dim)
# logger = TrainingLogger()
# server = RLServer(env_manager, agent, logger)
# server.run()

# This code defines the training, server setup, and utility functions for easier deployment and testing.
