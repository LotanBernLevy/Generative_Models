from typing import Any, Tuple
import utils
import torch
from models.BaseModel import BaseModel



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
        self.encoder = self.build_encoder(xshape, latent_length)
        self.decoder = self.build_decoder(xshape, latent_length)
        self.sample_layer = SampleZ()

    def build_encoder(self, xshape: Tuple, latent_length: int) -> torch.Tensor:
        return torch.nn.Linear(int(torch.prod(torch.Tensor(xshape))), latent_length*2)

    def build_decoder(self, xshape: Tuple, latent_length: int) -> torch.Tensor:
        return torch.nn.Linear(latent_length, int(torch.prod(torch.Tensor(xshape))))


    def forward(self, _input: torch.Tensor, **kwargs)-> torch.Tensor:
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

    def sample(self, size):
        z = torch.randn((size, self.latent_length))
        return self.decoder(z)

    def reconstruct(self,x):
        _x, reconstructed, mu, logvar = self(x)
        return reconstructed


class EncoderBlock(torch.nn.Module):
    def __init__(self, i, xshape, filters, kernel_size, stride):
        super(EncoderBlock, self).__init__()
        same_padding_0, same_padding_1 = int((kernel_size[0]-1) / 2), int((kernel_size[1]-1) / 2)
        self.layers = torch.nn.Sequential(torch.nn.Conv2d(xshape[0], filters, kernel_size, stride=stride, padding=(same_padding_0, same_padding_1)), torch.nn.BatchNorm2d(filters), torch.nn.LeakyReLU())
        self.get_output_size = lambda xshape: utils.conv2d_output_size(1,xshape[0],xshape[1],xshape[2], filters, kernel_size, stride=stride, padding=(same_padding_0, same_padding_1))

    def forward(self, x):
        return self.layers(x)

class DecoderBlock(torch.nn.Module):
    def __init__(self, i, xshape, filters, kernel_size, stride):
        super(DecoderBlock, self).__init__()
        same_padding_0, same_padding_1 = int((kernel_size[0]-1) / 2), int((kernel_size[1]-1) / 2)
        self.layers = torch.nn.Sequential(torch.nn.ConvTranspose2d(xshape[0], filters, kernel_size, stride=stride, padding=(same_padding_0, same_padding_1)), torch.nn.BatchNorm2d(filters), torch.nn.LeakyReLU())

    def forward(self, x):
        return self.layers(x)


class CVAE(VAE):
    def __init__(self, xshape: Tuple, latent_length: int, kl_weight:float=0.1):
        super(CVAE, self).__init__(xshape, latent_length, kl_weight)

    def forward(self, _input: torch.Tensor, **kwargs)-> torch.Tensor:
        mu_and_logvar = self.encoder(_input)
        mu_and_logvar = torch.reshape(mu_and_logvar, (mu_and_logvar.size()[0], 2, mu_and_logvar.size()[-1]//2))
        latent_shape = mu_and_logvar.size()[-1] 
        mu = mu_and_logvar[:, 0, :]
        logvar = mu_and_logvar[:, 1, :]
        z = self.sample_layer(mu, logvar)
        return _input, self.decoder(z), mu, logvar

    

    def build_encoder(self, xshape: Tuple, latent_length: int) -> torch.Tensor:
        encoders_blocks = []
        current_shape = xshape
        for i, (cout, stride) in enumerate([(1,1), (32,1), (64, 2), (64, 2), (64, 1)]):
            encoders_blocks.append(EncoderBlock(i, current_shape, cout, (3,3), (stride, stride)))
            current_shape = encoders_blocks[-1].get_output_size(current_shape)[1:]

        self.shape_before_flatten = current_shape

        encoder = torch.nn.Sequential(*encoders_blocks, torch.nn.Flatten(), torch.nn.Linear(int(torch.prod(torch.Tensor(self.shape_before_flatten))), latent_length*2))
        return encoder

    def build_decoder(self, xshape: Tuple, latent_length: int) -> torch.Tensor:
        decoders_blocks = []
        current_shape = self.shape_before_flatten
        for i, (cout, stride) in enumerate([(64,1), (64,2), (64, 2), (1, 1)]):
            decoders_blocks.append(DecoderBlock(i, current_shape, cout, (3,3), (stride, stride)))
            current_shape = (cout, None, None)

        decoder = torch.nn.Sequential(torch.nn.Linear(latent_length, int(torch.prod(torch.Tensor(self.shape_before_flatten)))), utils.Reshape(self.shape_before_flatten), *decoders_blocks)
        return decoder






    



