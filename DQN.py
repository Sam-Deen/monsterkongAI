import torch
import torch.nn as nn

# This class defines a DQN which helps a game-playing AI decide what move to make.
# It takes in images from the game and outputs a score  for each possible move.
class DQN(nn.Module):
    def __init__(self, input_channels, num_actions, input_size):
        super(DQN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Dynamically compute flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, *input_size)
            n_flatten = self.features(dummy_input).view(1, -1).size(1)

        self.net = nn.Sequential(
            self.features,  # Feature extractor
            nn.Flatten(),  # Convert 2D feature maps to 1D vector
            nn.Linear(n_flatten, 512),  # First fully connected layer
            nn.ReLU(),  # Non-linear activation
            nn.Linear(512, num_actions)  # Output layer: Q-values for each action
        )

    def forward(self, x):
        return self.net(x)

