import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.models.utils import weights_init
from src.models import vae


class CLWithDecoder(vae.VAE):


    def __init__(self, data_dim, hidden_dim, latent_dim, name,
                 sequence_length, batch_size):
        """
        """
        super(CLWithDecoder, self).__init__(
            data_dim=data_dim, hidden_dim=hidden_dim,
            latent_dim=latent_dim, name=name,
            sequence_length=sequence_length,
        )
        self.log_q = self.create_log_q()
        self.log_q.apply(weights_init)

    def create_log_q(self):
        return nn.Sequential(*[
            nn.Linear(self.data_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
                               ])

    def get_log_q(self, x):
        return self.log_q(x)

    def forward(self, x):
        z_mu, z_logvar = self.encoder(x)
        z = self.z_sample(mu=z_mu, logvar=z_logvar)
        x_tilde = self.decoder(z)
        return x_tilde, z, z_mu, z_logvar

