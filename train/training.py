import torch
from agent import DQNAgent
from env import Environment, TrainingLogger, RLServer  # Assuming these classes are saved in `rl_modules.py`

def main():
    # Initialize environment, agent, and logger
    env_name = 'LunarLander-v2'
    env = Environment(env_name)
    agent = DQNAgent(env.state_dim, env.action_dim)
    logger = TrainingLogger()

    # Optionally load a pre-trained model
    model_path = 'model.pth'
    try:
        agent.q_net.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"No pre-trained model found at {model_path}, starting training from scratch.")

    # Initialize and run the server
    server = RLServer(env, agent, logger, host='0.0.0.0', port=5000)
    server.run()

main()
