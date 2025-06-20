import math

import torch


class Diffusion:
    def __init__(self, noise_steps=500, schedule='linear', beta_start=1e-4, beta_end=0.02, s=0.008, device='cpu'):
        self.noise_steps = noise_steps

        if schedule == 'cosine':
            self.s = s
            beta, alpha_hat = self.cosine_noise_schedule()
        else:
            self.beta_start = beta_start
            self.beta_end = beta_end
            beta, alpha_hat = self.linear_noise_schedule()

        self.beta = beta.to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = alpha_hat.to(device)

    def cosine_noise_schedule(self):
        f = lambda t: torch.cos((t / self.noise_steps + self.s) / (1 + self.s) * math.pi * 0.5) ** 2

        x = torch.linspace(0, self.noise_steps, self.noise_steps + 1)
        alpha_hat = f(x) / f(torch.tensor(0.0))
        beta = torch.clip(1 - alpha_hat[1:] / alpha_hat[:-1], 0.0001, 0.999)

        return beta, alpha_hat[:-1]

    def linear_noise_schedule(self):
        beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        alpha = 1 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)

        return beta, alpha_hat

    def noise_images(self, x, t):
        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)
        sqrt_alpha_hat_t = torch.sqrt(alpha_hat_t)
        sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - alpha_hat_t)

        noise = torch.randn_like(x)
        noisy_x = sqrt_alpha_hat_t * x + sqrt_one_minus_alpha_hat_t * noise

        return noisy_x, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
