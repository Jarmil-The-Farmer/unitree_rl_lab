import torch
import torch.nn as nn
from rsl_rl.algorithms.actor_critic import ActorCritic

class VisualActorCritic(ActorCritic):
    def __init__(self, obs_shape, action_shape, **kwargs):
        super().__init__(obs_shape, action_shape, **kwargs)

        c, h, w = 3, 230, 320
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # získej velikost výstupu encoderu
        with torch.no_grad():
            n_flat = self.encoder(torch.zeros(1, c, h, w)).shape[1]

        # nahraď původní MLP embedder encoderem
        self.actor_body = nn.Sequential(
            self.encoder,
            nn.Linear(n_flat, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
        )
        self.critic_body = nn.Sequential(
            self.encoder,
            nn.Linear(n_flat, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
        )

    def forward(self, obs):
        img = obs["head_rgb"]
        # (N, C, H, W)
        features = self.encoder(img)
        return super().forward_features(features)
