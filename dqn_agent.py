import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from DQN import DQN
import os


class DQNAgent:
    def __init__(self, state_size, action_set, device):
        self.state_size = state_size
        self.action_set = action_set
        self.device = device

        # Mapping actions to indices and vice versa for compatibility with tensors
        self.action_to_index = {action: idx for idx, action in enumerate(action_set)}
        self.index_to_action = {idx: action for action, idx in self.action_to_index.items()}

        self.action_size = len(action_set)

        # Initialize the main Q-network and the target Q-network
        self.model = DQN(4, self.action_size, state_size).to(device)
        self.target_model = DQN(4, self.action_size, state_size).to(device)
        self.update_target_model()

        self.memory = deque(maxlen=75000)

        # Hyperparameters
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration rate (probability of random action)
        self.epsilon_decay = 0.999  # Decay rate for epsilon
        self.epsilon_min = 0.01  # Minimum epsilon (stopping point for decay)
        self.learning_rate = 0.001  # Learning rate for optimizer
        self.batch_size = 32  # Number of experiences to sample per training step
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.frame_stack = 4  # number of frames to stack
        self.state_stack = deque(maxlen=self.frame_stack)

    def update_target_model(self):
        # Copy weights from the main network to the target network
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # Store a transition tuple in the replay memory
        self.memory.append((state, self.action_to_index[action], reward, next_state, done))

    def select_action(self, state):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            action_index = np.random.choice(len(self.action_set))
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
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path="checkpoint"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pth"))
        torch.save(self.target_model.state_dict(), os.path.join(path, "target_model.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pth"))
        torch.save({
            "epsilon": self.epsilon,
            "memory": list(self.memory)  # Save as list for compatibility
        }, os.path.join(path, "meta.pth"))
        print("Agent state saved.")

    def load(self, path="checkpoint"):
        self.model.load_state_dict(torch.load(os.path.join(path, "model.pth")))
        self.target_model.load_state_dict(torch.load(os.path.join(path, "target_model.pth")))
        self.optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer.pth")))

        meta = torch.load(os.path.join(path, "meta.pth"))
        self.epsilon = meta["epsilon"]
        self.memory = deque(meta["memory"], maxlen=self.memory.maxlen)
        print("Agent state loaded.")