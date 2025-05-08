import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from DQN import DQN
import os
from tqdm import tqdm
import numpy as np
import pickle

class DQNAgent:
    def __init__(self, input_dim, action_set, device):
        self.input_dim = input_dim
        self.action_set = action_set
        self.device = device

        # Mapping actions to indices and vice versa for compatibility with tensors
        self.action_to_index = {action: idx for idx, action in enumerate(action_set)}
        self.index_to_action = {idx: action for action, idx in self.action_to_index.items()}

        self.action_size = len(action_set)

        self.model = DQN(input_dim, self.action_size).to(device)
        self.target_model = DQN(input_dim, self.action_size).to(device)
        self.update_target_model()

        self.memory = deque(maxlen=75000)

        # Hyperparameters
        self.gamma = 0.99  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration rate (probability of random action)
        self.epsilon_decay = 0.999  # Decay rate for epsilon
        self.epsilon_min = 0.01  # Minimum epsilon (stopping point for decay)
        self.learning_rate = 0.001  # Learning rate for optimizer
        self.batch_size = 64  # Number of experiences to sample per training step
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update_target_model(self):
        # Copy weights from the main network to the target network
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # Store a transition tuple in the replay memory
        self.memory.append((state, self.action_to_index[action], reward, next_state, done))

    def select_action(self, state):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            action_index = np.random.choice(self.action_size)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action_index = torch.argmax(q_values).item()
        return self.index_to_action[action_index]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath, total_episodes):
        """Saves the agent's model and training state with a loading bar."""
        save_data = {
            "model_state": self.model.state_dict(),
            "target_model_state": self.target_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_episodes": total_episodes,
            "epsilon": self.epsilon,
            "memory": list(self.memory)  # Convert deque to list for saving
        }
        with tqdm(total=100, desc="Saving Progress", unit="step") as pbar:
            with open(filepath, "wb") as f:
                pickle.dump(save_data, f)
                # Simulate progress for the save process
                for i in range(100):
                    pbar.update(1)
        print(f"[INFO] Model and training state saved to {filepath}")

    def load(self, filepath):
        """Loads the agent's model and training state."""
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)
        self.model.load_state_dict(save_data["model_state"])
        self.target_model.load_state_dict(save_data["target_model_state"])
        self.optimizer.load_state_dict(save_data["optimizer_state"])
        self.epsilon = save_data["epsilon"]
        self.memory = deque(save_data["memory"], maxlen=self.memory.maxlen)
        print(f"[INFO] Model and training state loaded from {filepath}")
        return save_data["total_episodes"]