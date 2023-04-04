import numpy as np
import torch

def layer_init(
    layer: torch.nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> torch.nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# TODO: Make this configurable from the command line
class PolicyNet(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        # Input: 4x84x84, assumes frame stacking, resizing, and gray-scaling have been done.
        self.network = torch.nn.Sequential(
            layer_init(torch.nn.Conv2d(4, 32, 8, stride=4)), # 4x84x84 -> 32x20x20
            torch.nn.ReLU(inplace=True),
            layer_init(torch.nn.Conv2d(32, 64, 4, stride=2)), # 32x20x20 -> 64x9x9
            torch.nn.ReLU(inplace=True),
            layer_init(torch.nn.Conv2d(64, 64, 3, stride=1)), # 64x9x9 -> 64x7x7
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.network(torch.rand((4, 84, 84))).size()) # Need this for Actor, Critic functions to work correctly.
        self.device = device

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        logits = self.network(obs / 255.0)
        return logits, state