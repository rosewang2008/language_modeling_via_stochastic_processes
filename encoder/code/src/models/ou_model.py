import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.models.utils import weights_init
from src.models import vae


class OUModel(nn.Module):

    def __init__(self, data_dim, hidden_dim, output_dim):
        """
        """
        super(OUModel, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Construct model
        self.feature_extractor = self.create_feature_extractor() # data_dim -> hidden_dim
        self.predictor = self.create_prediction_head() # hidden_dim -> output_dim
        self.log_q = self.create_log_q()
        self.C_eta = nn.Linear(1, 1)

        # # Turn off grad if needed
        # if not self.MLP_trainable:
        #     for param in self.feature_extractor.parameters():
        #         param.requires_grad = False
        # # Set biases to 0
        self.feature_extractor.apply(weights_init)
        self.predictor.apply(weights_init)
        self.log_q.apply(weights_init)
        self.C_eta.apply(weights_init)

    def set_to_train(self):
        for param in self.parameters():
            param.requires_grad = True

    def create_feature_extractor(self):
        return nn.Sequential(*[
            nn.Linear(self.data_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
                               ])

    def create_log_q(self):
        return nn.Sequential(*[
            nn.Linear(self.data_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
                               ])

    def forward(self, inputs):
        feats = None
        preds = self.predictor(inputs) # batch_size, output_dim
        return feats, preds

    def get_log_q(self, x):
        return self.log_q(x)

    def create_prediction_head(self):
        return nn.Linear(self.data_dim, self.output_dim)

class NoisyOUModel(OUModel):

    def __init__(self, data_dim, hidden_dim, latent_dim, name,
                 sequence_length):
        super().__init__(data_dim=data_dim, hidden_dim=hidden_dim,
                         output_dim=data_dim)

        self.vae_model = vae.VAE(data_dim=data_dim, hidden_dim=hidden_dim,
                           latent_dim=latent_dim, name=name,
                           sequence_length=sequence_length,)
        # Only train encoder
        for param in self.vae_model.decoder.parameters():
            param.requires_grad = False


    def forward(self, x):
        z_mu, z_logvar = self.vae_model.encoder(x)
        z = self.vae_model.z_sample(mu=z_mu, logvar=z_logvar)
        x_tilde = self.vae_model.decoder(z)
        return x_tilde, z, z_mu, z_logvar

class OUModel_Mu(OUModel):

    def __init__(self, data_dim, hidden_dim, output_dim):
        """
        """
        super().__init__(data_dim=data_dim, hidden_dim=hidden_dim, output_dim=output_dim)

        self.mu = nn.Linear(1, 1)
        self.mu.apply(weights_init)

    def forward(self, o_t, o_tp1):
        feats_t = self.feature_extractor(o_t) # batch_size, feat_dim
        preds_t = self.predictor(feats_t) # batch_size, output_dim
        preds_t = self.mu(preds_t) # \mu ~ 1-dt
        feats_tp1 = self.feature_extractor(o_tp1)
        preds_tp1 = self.predictor(feats_tp1)
        return (feats_t, preds_t), (feats_tp1, preds_tp1)


class OUModel_Bridge(OUModel):

    def __init__(self, data_dim, hidden_dim, output_dim, T):
        """
        """
        super().__init__(data_dim=data_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.eps = 1e-6
        self.dt = 1./T

    def forward(self, o_t, o_tp1, t):
        feats_t = self.feature_extractor(o_t) # batch_size, feat_dim
        preds_t = self.predictor(feats_t) # batch_size, output_dim
        preds_t = preds_t * t[:, None]
        feats_tp1 = self.feature_extractor(o_tp1)
        preds_tp1 = self.predictor(feats_tp1)
        return (feats_t, preds_t), (feats_tp1, preds_tp1)
