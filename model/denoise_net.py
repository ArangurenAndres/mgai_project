import torch
import torch.nn as nn

class SimpleDenoiseNet(nn.Module):
    def __init__(self, in_channels: int, time_dim: int = 64, depth: int = 5):
        super().__init__()
        self.time_embed = nn.Embedding(1000, time_dim)

        # Initial projection from (in_channels + time_dim) -> in_channels
        self.input_conv = nn.Conv2d(in_channels + time_dim, in_channels, kernel_size=3, padding=1)

        # Residual blocks (same input/output channels)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.LeakyReLU()
            ))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Embed time and expand to match spatial dimensions
        t_embed = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)  # (B, time_dim, 1, 1)
        t_embed = t_embed.expand(-1, -1, x.shape[2], x.shape[3])  # (B, time_dim, H, W)

        # Concatenate and project
        x = torch.cat([x, t_embed], dim=1)  # (B, C + time_dim, H, W)
        x = self.input_conv(x)             # (B, C, H, W)

        # Residual blocks
        for block in self.blocks:
            x = x + block(x)

        return x
