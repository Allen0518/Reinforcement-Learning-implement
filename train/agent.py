import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Policy Gradient Agent
class PolicyGradientAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3):
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.action_dim = action_dim
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def update_policy(self, rewards, log_probs, gamma=0.99):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        loss = -torch.sum(torch.stack(log_probs) * discounted_rewards)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Actor-Critic Agent
class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def update(self, state, reward, next_state, log_prob, gamma=0.99):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        
        td_target = reward + gamma * self.critic(next_state)
        td_error = td_target - self.critic(state)
        
        actor_loss = -log_prob * td_error.detach()
        critic_loss = td_error.pow(2)
        
        loss = actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

#  DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.target_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)
        return torch.argmax(q_values).item()
    
    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        q_values = self.q_net(states).gather(1, actions).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0]
        target = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())