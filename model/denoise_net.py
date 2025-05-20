import torch
import torch.nn as nn

class SimpleDenoiseNet(nn.Module):
    def __init__(self, in_channels, time_dim=64, depth=5):
        super().__init__()
        self.time_embed = nn.Embedding(1000, time_dim)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_channels + time_dim, in_channels, 3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.LeakyReLU()
            ))

    def forward(self, x, t):
        t_embed = self.time_embed(t).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat([x, t_embed], dim=1)
        for block in self.blocks:
            x = x + block(x)
        return x
