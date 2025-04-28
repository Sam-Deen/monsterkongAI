import numpy as np
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    def __init__(self, action_space, downsample_size=(42, 42), learning_rate=0.001,
                 discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995,
                 min_exploration_rate=0.01, batch_size=32, memory_size=10000):
        self.action_space = action_space
        self.downsample_size = downsample_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate
        self.batch_size = batch_size

        # Experience replay memory
        self.memory = []
        self.memory_size = memory_size

        # Initialize the DQN
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def build_model(self):
        """Build the neural network for DQN."""
        model = nn.Sequential(
            nn.Linear(np.prod(self.downsample_size), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.action_space))
        )
        return model

    def preprocess(self, image):
        """Shrink size, and flatten."""
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        image = cv2.resize(image, self.downsample_size, interpolation=cv2.INTER_AREA)
        state = image.flatten() / 255.0  # Normalize pixel values
        return torch.FloatTensor(state)

    def act(self, image):
        """Return the key corresponding to the chosen action."""
        state = self.preprocess(image).unsqueeze(0)  # Add batch dimension

        if random.random() < self.epsilon:
            action_idx = random.randint(0, len(self.action_space) - 1)
        else:
            with torch.no_grad():
                q_values = self.model(state)
                action_idx = torch.argmax(q_values).item()

        return self.action_space[action_idx]

    def learn(self, current_image, action_key, reward, next_image, done):
        """Update the DQN based on a transition."""
        action_idx = self.get_action_index(action_key)
        state = self.preprocess(current_image)
        next_state = self.preprocess(next_image)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        self.memory.append((state, action_idx, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

        if len(self.memory) >= self.batch_size:
            self.replay()

        if done.item():
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def replay(self):
        """Train the model using random samples from the replay memory."""
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.cat(rewards)
        next_states = torch.stack(next_states)
        dones = torch.cat(dones)

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1)[0]
            q_targets = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.loss_fn(q_values.squeeze(), q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        """Update the target model weights."""
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action_index(self, action_key):
        """Find action index given a key."""
        if action_key in self.action_space:
            return self.action_space.index(action_key)
        else:
            raise ValueError(f"Invalid action key: {action_key}")