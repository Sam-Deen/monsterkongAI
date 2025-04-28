import numpy as np
import random
import cv2


class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.001, discount_factor=0.9,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01,
                 downsample_size=(42, 42)):

        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate
        self.downsample_size = downsample_size

        self.q_table = {}

    def preprocess(self, image):
        """shrink size, and flatten"""

        # try to turn image into numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if np.max(image) == 0:
            print("Warning: Empty (all black) frame detected!")
            return tuple(image.flatten())

        # shrink image to set size, def 42x42
        target_h, target_w = self.downsample_size
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        # flatten image into 1d array
        state = tuple(image.flatten())
        return state

    def act(self, image):
        """Return the key corresponding to the chosen action."""
        state = self.preprocess(image)

        if random.random() < self.epsilon:
            action_idx = random.randint(0, len(self.action_space) - 1)
        else:
            q_values = self.q_table.get(state, np.zeros(len(self.action_space)))
            action_idx = np.argmax(q_values)

        return self.action_space[action_idx]

    def learn(self, current_image, action_key, reward, next_image, done):
        """Update Q-table based on transition."""
        # Find the action index from the key
        action_idx = self.get_action_index(action_key)

        state = self.preprocess(current_image)
        next_state = self.preprocess(next_image)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.action_space))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.action_space))

        q_predict = self.q_table[state][action_idx]
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action_idx] += self.lr * (q_target - q_predict)

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_action_index(self, action_key):
        """Find action index given a key."""
        if action_key in self.action_space:
            return self.action_space.index(action_key)
        else:
            raise ValueError(f"Invalid action key: {action_key}")
