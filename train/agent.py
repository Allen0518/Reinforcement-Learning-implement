import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class PolicyGradientNetwork(nn.Module):

    def __init__(self, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(hid)
        return F.softmax(self.fc3(hid), dim=-1)

# Value Network
class ValueNet(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.hidden = nn.Linear(8, hidden_dim)
        self.output = nn.Linear(hidden_dim, 4)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        value = self.output(outs)
        return value
    
from torch.optim.lr_scheduler import StepLR
class PolicyGradientAgent():

    def __init__(self, network):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.002)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.rewards = None
        self.discounted_rewards = None

    def forward(self, state):
        return self.network(state)

    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn_A2C(self, log_probs, rewards, states, value_func):
        with torch.no_grad():
            values = value_func(states).squeeze()
        advantages = rewards - values
        loss = (-log_probs * advantages).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        # Convert state to a FloatTensor
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()  # Convert directly from NumPy array to tensor
        elif isinstance(state, list):
            state = torch.FloatTensor(state)
        else:
            raise ValueError(f"Unexpected state format: {type(state)}")

        # Get action probabilities and sample an action
        action_prob = self.network(state)
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
