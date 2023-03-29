import numpy as np
import torch

# TODO: Make this configurable from the command line
class PolicyNet(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        # Input: 4x84x84, assumes frame stacking, resizing, and gray-scaling have been done.
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, 8, stride=4), # 4x84x84 -> 16x20x20
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, 4, stride=2), # 16x20x20 -> 32x9x9
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.network(torch.rand((4, 84, 84))).size()) # Need this for Actor, Critic functions to work correctly.
        self.device = device

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        logits = self.network(obs)
        return logits, state