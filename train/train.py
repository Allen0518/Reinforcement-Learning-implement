import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from agent import PolicyGradientNetwork, PolicyGradientAgent

def fix(env, seed):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(env_name='LunarLander-v2', num_batch=500, episode_per_batch=5, gamma=0.99, seed=2023):
    # Setup environment and agent
    env = gym.make(env_name)
    fix(env, seed)

    network = PolicyGradientNetwork()
    agent = PolicyGradientAgent(network)

    agent.network.train()  # Switch network into training mode

    avg_total_rewards, avg_final_rewards = [], []

    prg_bar = tqdm(range(num_batch))
    for batch in prg_bar:

        log_probs, rewards = [], []
        total_rewards, final_rewards = [], []

        # Collect trajectory
        for episode in range(episode_per_batch):
            result = env.reset()
            if isinstance(result, tuple):
                state, _ = result  # If reset returns a tuple, extract the state
            else:
                state = result  # If it returns only the state

            total_reward, total_step = 0, 0
            episode_rewards = []  # Store episode-specific rewards

            while True:
                action, log_prob = agent.sample(state)  # Get action and log probability
                next_state, reward, done, truncated, _ = env.step(action)

                log_probs.append(log_prob)  # Store log probability
                episode_rewards.append(reward)  # Store immediate reward
                state = next_state
                total_reward += reward
                total_step += 1

                if done or truncated:
                    final_rewards.append(reward)
                    total_rewards.append(total_reward)
                    break

            # Convert episode rewards to discounted cumulative rewards
            discounted_rewards = []
            cumulative_reward = 0

            # Compute cumulative decaying rewards for the episode
            for r in reversed(episode_rewards):
                cumulative_reward = r + gamma * cumulative_reward
                discounted_rewards.insert(0, cumulative_reward)  # Insert at the front to keep the right order

            rewards.extend(discounted_rewards)  # Extend the rewards list with the episode's discounted rewards

        # Record training process
        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)
        prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

        # Update the agent using cumulative decaying rewards
        rewards = np.array(rewards)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # Normalize the rewards
        agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))

    # Plot the training progress
    plt.plot(avg_total_rewards)
    plt.xlabel('Batch')
    plt.ylabel('Average Total Reward')
    plt.title('Policy Gradient Training Progress')
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a policy gradient agent.")
    parser.add_argument('--env_name', type=str, default='LunarLander-v2', help='Name of the environment')
    parser.add_argument('--num_batch', type=int, default=500, help='Number of batches for training')
    parser.add_argument('--episode_per_batch', type=int, default=5, help='Number of episodes per batch')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')

    args = parser.parse_args()

    train(env_name=args.env_name, num_batch=args.num_batch, episode_per_batch=args.episode_per_batch, gamma=args.gamma, seed=args.seed)
