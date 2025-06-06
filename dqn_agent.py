import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from DQN import DQN
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

        self.memory = deque(maxlen=250000)
        self.win_memory = deque(maxlen=10000)

        # Hyperparameters
        self.gamma = 0.99  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration rate (probability of random action)
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.epsilon_min = 0.05  # Minimum epsilon (stopping point for decay)
        self.learning_rate = 0.0005  # Learning rate for optimizer
        self.batch_size = 512  # Number of experiences to sample per training step
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        self.rng = np.random.default_rng()

        self.loss_history = []

    def update_target_model(self):
        # Copy weights from the main network to the target network
        self.target_model.load_state_dict(self.model.state_dict())

    def remember_single(self, state, action, reward, next_state, done, to_win_memory=False):
        # Store a transition tuple in the replay memory
        if to_win_memory:
            self.win_memory.append((state, self.action_to_index[action], reward, next_state, done))
        else:
            self.memory.append((state, self.action_to_index[action], reward, next_state, done))

    def remember_batch(self, transition_list, to_win_memory=False):
        for transition in transition_list:
            state, action, reward, next_state, done = transition
            if to_win_memory:
                self.win_memory.append((state, self.action_to_index[action], reward, next_state, done))
            else:
                self.memory.append((state, self.action_to_index[action], reward, next_state, done))

    def select_action(self, state):
        # Choose action using epsilon-greedy policy
        if self.rng.random() < self.epsilon:
            action_index = self.rng.integers(self.action_size)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action_index = torch.argmax(q_values).item()
        return self.index_to_action[action_index]

    def replay(self):
        if len(self.memory) + len(self.win_memory) < self.batch_size:
            return

        win_batch_size = int(self.batch_size * 0.3)
        win_batch_size = min(win_batch_size, len(self.win_memory))
        normal_batch_size = self.batch_size - win_batch_size

        # Sample a batch
        batch = []
        if normal_batch_size > 0:
            batch.extend(random.sample(self.memory, normal_batch_size))
        if win_batch_size > 0:
            batch.extend(random.sample(self.win_memory, win_batch_size))



        states = torch.from_numpy(np.stack([b[0] for b in batch])).to(self.device)
        next_states = torch.from_numpy(np.stack([b[3] for b in batch])).to(self.device)
        rewards = torch.from_numpy(np.array([b[2] for b in batch], dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.array([b[4] for b in batch], dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.array([b[1] for b in batch], dtype=np.int64)).to(self.device)

        # Compute current Q-values from the main model
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: Use main model to select best actions in next state
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and optimize
        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Store loss history
        self.loss_history.append(loss.item())
        if len(self.loss_history) > 100000:
            self.loss_history.pop(0)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def save(self, filepath, total_episodes):
        """Saves the agent's model and training state with a loading bar."""
        save_data = {
            "model_state": self.model.state_dict(),
            "target_model_state": self.target_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_episodes": total_episodes,
            "epsilon": self.epsilon,
            "memory": list(self.memory),  # Convert deque to list for saving
            "win_memory": list(self.win_memory),
        }
        with tqdm(total=100, desc="Saving Progress", unit="step") as pbar:
            with open(filepath, "wb") as f:
                pickle.dump(save_data, f)
                # Simulate progress for the save process
                for _ in range(100):
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
        self.win_memory = deque(save_data["win_memory"], maxlen=self.win_memory.maxlen)
        print(f"[INFO] Model and training state loaded from {filepath}")
        return save_data["total_episodes"]