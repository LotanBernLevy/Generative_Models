from typing import Any, Tuple, Union
import utils
import torch
import math
from models.BaseModel import BaseModel
import torch.nn.functional as F
import math


class SampleZ(torch.nn.Module):
    def forward(self, mu, logvar):
        """
        Args:
            mu_and_logvar (torch.Tensor): B X 2 X latent_length
        """
        e = torch.randn(mu.size())
        return mu + torch.exp(0.5*logvar) * e


class VAE(BaseModel):
    def __init__(self, xshape: Tuple, latent_length: int, kl_weight:float=0.1):
        super(VAE, self).__init__()
        self.latent_length = latent_length
        self.kl_weight = kl_weight
        self.xshape = xshape
        self.encoder = self.build_encoder(xshape, latent_length)
        self.decoder = self.build_decoder(xshape, latent_length)
        self.sample_layer = SampleZ()

    def build_encoder(self, xshape: Tuple, latent_length: int) -> torch.Tensor:
        return torch.nn.Linear(int(torch.prod(torch.Tensor(xshape))), latent_length*2)

    def build_decoder(self, xshape: Tuple, latent_length: int) -> torch.Tensor:
        return torch.nn.Linear(latent_length, int(torch.prod(torch.Tensor(xshape))))

    def forward(self, *inputs: Any, **kwargs):
        _input = inputs
        x = torch.flatten(_input, start_dim=1)
        mu_and_logvar = self.encoder(x)
        mu_and_logvar = torch.reshape(mu_and_logvar, (mu_and_logvar.size()[0], 2, mu_and_logvar.size()[-1]//2))
        latent_shape = mu_and_logvar.size()[-1] 
        mu = mu_and_logvar[:, 0, :]
        logvar = mu_and_logvar[:, 1, :]
        z = self.sample_layer(mu, logvar)
        return _input, self.decoder(z), mu, logvar

    def loss(self, *inputs: Any, **kwargs) -> torch.Tensor:
        x, predicted_x, mu, logvar = inputs
        reconstruction_loss = torch.nn.MSELoss()(predicted_x, x)
        kl_loss = torch.mean(torch.sum(mu ** 2 + torch.exp(logvar)  - logvar - 1, dim=1), dim=0)
        loss = reconstruction_loss + self.kl_weight * kl_loss
        return {'loss': loss, 'Reconstruction_Loss':reconstruction_loss, 'kl loss':kl_loss}

    def sample(self, size, *args: Any, **kwargs):
        z = torch.randn((size, self.latent_length))
        return self.decoder(z)

    def reconstruct(self, *inputs: Any, **kwargs):
        x = inputs
        _x, reconstructed, mu, logvar = self(x)
        return reconstructed