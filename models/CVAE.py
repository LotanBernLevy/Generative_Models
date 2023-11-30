from typing import Any, Tuple, Union
import utils
import torch
import math
from models.BaseModel import BaseModel
import torch.nn.functional as F
import math
from models.VAE import SampleZ


class CVAE(BaseModel):
    def __init__(self, num_classes, xshape: Tuple, latent_length: int, kl_weight:float=0.1, hidden_channels=[32, 64, 128, 256]):
        super(CVAE, self).__init__()
        self.num_classes = num_classes
        self.latent_length = latent_length
        self.kl_weight = kl_weight
        self.xshape = xshape
        self.sample_layer = SampleZ()

        self.classes_embedding = torch.nn.Linear(num_classes, xshape[1]*xshape[2])
        self.input_embedding = torch.nn.Conv2d(xshape[0],xshape[0], kernel_size=(1,1))

        self.hidden_channels = hidden_channels

        in_shape = (xshape[0] + 1, xshape[1], xshape[2]) # one more for the classes channel

        # encoder layers
        in_shape, encoder_blocks = self.get_encoder_blocks(in_shape)
        self.encoder = torch.nn.Sequential(*encoder_blocks)
        self.mu_var_layer = torch.nn.Linear(in_shape[0]*in_shape[1]*in_shape[2], latent_length*2)

        #decoder layers
        self.embed_latent_with_classes = torch.nn.Linear(latent_length + self.num_classes, in_shape[0]*in_shape[1]*in_shape[2])
        in_shape, decoder_blocks = self.get_decoder_blocks(in_shape)
        self.decoder = torch.nn.Sequential(*decoder_blocks)

    def get_encoder_blocks(self, in_shape:int):
        encoder_blocks = []
        for h in self.hidden_channels:
            encoder_blocks.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_shape[0], out_channels=h,
                              kernel_size= 3, stride= 2, padding  = 1),
                    torch.nn.BatchNorm2d(h),
                    torch.nn.LeakyReLU())
                    )
            in_shape = utils.conv2d_output_size(1,*in_shape, h, kernel_size=(3,3), stride=(2,2), padding=(1,1))[1:]
        return in_shape, encoder_blocks

    def get_decoder_blocks(self, in_shape:int):
        decoder_blocks = []        
        for h in reversed( [1] + self.hidden_channels[:-1]):
            out_pad = 0 if h==64 else 1 # to make sure that the decoder output has the same shape as the encoder input.
            decoder_blocks.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_shape[0], out_channels=h,
                              kernel_size= 3, stride= 2, padding  = 1, output_padding=out_pad),
                    torch.nn.BatchNorm2d(h),
                    torch.nn.LeakyReLU())
                    )
            
            in_shape =  utils.convTranspose2d_output_size(1,*in_shape, h, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(out_pad, out_pad))[1:]
        return in_shape, decoder_blocks

    def encode(self, x:torch.Tensor): # S(x, l) 
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu_and_logvar = self.mu_var_layer(x)
        mu_and_logvar = torch.reshape(mu_and_logvar, (mu_and_logvar.size()[0], 2, mu_and_logvar.size()[-1]//2))
        latent_shape = mu_and_logvar.size()[-1] 
        mu = mu_and_logvar[:, 0, :]
        logvar = mu_and_logvar[:, 1, :]
        z = self.sample_layer(mu, logvar)
        return z, mu, logvar

    def decode(self, z:torch.tensor, label:torch.tensor): # G(z, l) - the generator
        to_decode = torch.cat([z, label], dim = 1)
        to_decode = self.embed_latent_with_classes(to_decode)
        embedded_img_size = int(math.ceil(to_decode.shape[1]/(2*self.hidden_channels[-1])))
        to_decode = to_decode.view(-1, self.hidden_channels[-1], embedded_img_size, embedded_img_size)
        return self.decoder(to_decode)


    def forward(self, *inputs: Any, **kwargs):
        _input, labels = inputs
        if len(labels.shape) == 1: # a batch of classes nums instead of one-hot vectors
            labels = F.one_hot(labels.to(torch.int64), num_classes=self.num_classes).float()
        # embed the images and the labels and then concatenate them before the encoding step.
        embedded_labels = self.classes_embedding(labels)
        embedded_labels = embedded_labels.view(-1, self.xshape[1], self.xshape[2]).unsqueeze(1)
        embedded_input = self.input_embedding(_input)
        x = torch.cat([embedded_input, embedded_labels], dim = 1)
        z, mu, logvar = self.encode(x)
        return _input, self.decode(z, labels), mu, logvar


    def sample(self, size, *args: Any, **kwargs):
        labels = args[0]
        if len(labels.shape) == 1: # a batch of classes nums instead of one-hot vectors
            labels = F.one_hot(labels.to(torch.int64), num_classes=self.num_classes).float()
        z = torch.randn((size, self.latent_length))
        return self.decode(z, labels)


    def reconstruct(self, *inputs: Any, **kwargs):
        x, label = inputs
        _x, reconstructed, mu, logvar = self(x, label)
        return reconstructed


    def loss(self, *inputs: Any, **kwargs) -> torch.Tensor:
        x, predicted_x, mu, logvar = inputs
        reconstruction_loss = torch.nn.MSELoss()(predicted_x, x)
        kl_loss = torch.mean(torch.sum(mu ** 2 + torch.exp(logvar)  - logvar - 1, dim=1), dim=0)
        loss = reconstruction_loss + self.kl_weight * kl_loss
        return {'loss': loss, 'Reconstruction_Loss':reconstruction_loss, 'kl loss':kl_loss}