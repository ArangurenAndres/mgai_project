import torch
import numpy as np

class GaussianDiffusion:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.model = model
        self.device = device
        self.timesteps = timesteps

        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x0, t, noise):
        a_bar = self.alpha_bar[t].view(-1, 1, 1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

    def p_sample(self, x, t):
        return self.model(x, t)

    def train_step(self, x0, optimizer):
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=self.device).long()
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred_x0 = self.model(xt, t)
        loss = torch.nn.functional.mse_loss(pred_x0, x0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, shape):
        x = torch.randn(shape).to(self.device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            pred_x0 = self.model(x, t_tensor)
            if t > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(self.alpha_bar[t - 1]) * pred_x0 + torch.sqrt(1 - self.alpha_bar[t - 1]) * noise
            else:
                x = pred_x0
        return x
