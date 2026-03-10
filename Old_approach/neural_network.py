import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNN(BaseFeaturesExtractor):
    """
    CNN features extractor for small grids like (C, 7, 7) or (C, 11, 11).
    Works with VecTransposeImage -> CHW observations.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Observation_space.shape should be (C, H, W)
        n_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # One gentle downsample (7->3, 11->5). Keeps locality but reduces params.
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # Make output independent of H,W
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # After AdaptiveAvgPool2d(1,1), we have 64 features
        self.mlp = nn.Sequential(
            nn.Linear(64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.cnn(observations)
        return self.mlp(x)
