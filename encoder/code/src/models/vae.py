import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.models.utils import weights_init


class Encoder(nn.Module):

    def __init__(self, data_dim, hidden_dim, latent_dim):
        """X -> Z"""
        super(Encoder, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.base = self.create_base()
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        #
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_logvar.weight)

    def forward(self, x):
        x = self.base(x)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

    def create_base(self):
        return nn.Sequential(*[
            nn.Linear(self.data_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),])

class RecurrentEncoder(Encoder):
    def __init__(self, data_dim, hidden_dim, latent_dim, depth=2, dropout=0.0):
        """X -> Z"""
        self.depth = depth
        self.dropout = dropout
        super(RecurrentEncoder, self).__init__(
            data_dim=data_dim, hidden_dim=hidden_dim, latent_dim=latent_dim
        )

        self.rnn = nn.LSTM(input_size=self.data_dim,
                           hidden_size=self.hidden_dim,
                           num_layers=self.depth,
                           batch_first=True)


    def create_base(self):
        return nn.GRU(self.data_dim, self.hidden_dim, self.depth, dropout=self.dropout)

    def forward(self, x):
        """
        :param x: shape (bsz, seq_len, num_feats)
        :return: last hidden state of encoder, shape (bsz, hidden_size)
        """
        # batch_size = x.shape[0]

        # test_a, (h_n, c_n) = self.rnn(x)
        # h_end = h_n[-1]
        # c_end = c_n[-1]
        # # x_mu = self.fc_mu(h_end)
        # # x_logvar = self.fc_logvar(h_end)
        # x_mu = self.fc_mu(c_end)
        # x_logvar = self.fc_logvar(c_end)
        # return x_mu, x_logvar

        _, (h_end, c_end) = self.base(x.transpose(0, 1))
        # h_end = h_end[-1, :, :]
        x_mu = self.fc_mu(h_end)
        x_logvar = self.fc_logvar(h_end)
        return x_mu, x_logvar


class Decoder(nn.Module):

    def __init__(self, data_dim, hidden_dim, latent_dim):
        """Z -> X"""
        super(Decoder, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.base = self.create_base()

    def forward(self, z):
        x = self.base(z)
        return x

    def create_base(self):
        return nn.Sequential(*[
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.data_dim)
        ])

class RecurrentGRUDecoder(Decoder):

    def __init__(self, data_dim, hidden_dim, latent_dim,
                 sequence_length,
                 depth=2, dropout=0.0):
        """Z -> X

        https://github.com/kefirski/pytorch_RVAE/blob/master/model/decoder.py
        """


        self.depth = depth
        self.dropout = dropout
        self.sequence_length = sequence_length
        super(RecurrentGRUDecoder, self).__init__(
            data_dim=data_dim, hidden_dim=hidden_dim, latent_dim=latent_dim
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 512

        self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True, device=self.device)

        self.rnn = nn.GRU(input_size=1,
                          hidden_size=self.hidden_dim,
                          num_layers=self.depth)
        self.latent2hidden = nn.Linear(self.latent_dim, self.hidden_dim)
        self.hidden2output = nn.Linear(self.hidden_dim, self.data_dim)

        nn.init.xavier_uniform_(self.latent2hidden.weight)
        nn.init.xavier_uniform_(self.hidden2output.weight)

    def forward(self, z, initial_state=None):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        """
        # if len(z.shape) == 2:
        #     # z = z.unsqueeze(1)
        #     z = z.unsqueeze(0)

        h_state = self.latent2hidden(z)
        h_0 = torch.stack([h_state for _ in range(self.depth)])
        # decoder_output, _ = self.rnn(z, (h_0, self.c_0))
        # z = z.unsqueeze(0)
        decoder_output, _ = self.rnn(self.decoder_inputs, h_0)
        decoder_output = decoder_output[-1, :, :] # bsz, hidden_dim
        # rnn_out = rnn_out.contiguous().view(-1, self.hidden_dim)
        x = self.hidden2output(decoder_output)
        return x

        # h_state = self.latent2hidden(z)
        # h_0 = torch.stack([h_state for _ in range(self.depth)])
        # decoder_output, _ = self.base(self.decoder_inputs, h_0) # seq_length, bsz, hidden_dim
        # decoder_output = decoder_output[-1, :, :] # bsz, hidden_dim
        # x = self.hidden2output(decoder_output)
        # return x

class RecurrentDecoder(Decoder):

    def __init__(self, data_dim, hidden_dim, latent_dim,
                 sequence_length,
                 depth=2, dropout=0.0):
        """Z -> X

        https://github.com/kefirski/pytorch_RVAE/blob/master/model/decoder.py
        """


        self.depth = depth
        self.dropout = dropout
        self.sequence_length = sequence_length
        super(RecurrentDecoder, self).__init__(
            data_dim=data_dim, hidden_dim=hidden_dim, latent_dim=latent_dim
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # self.rnn = nn.LSTM(input_size=self.latent_dim,
        #                    hidden_size=self.hidden_dim,
        #                    num_layers=self.depth,
        #                    batch_first=True)
        self.rnn = nn.GRU(input_size=self.latent_dim,
                          hidden_size=self.hidden_dim,
                          num_layers=self.depth)
        self.latent2hidden = nn.Linear(self.latent_dim, self.hidden_dim)
        self.hidden2output = nn.Linear(self.hidden_dim, self.data_dim)

        nn.init.xavier_uniform_(self.latent2hidden.weight)
        nn.init.xavier_uniform_(self.hidden2output.weight)

    def forward(self, z, initial_state=None):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        """
        # if len(z.shape) == 2:
        #     # z = z.unsqueeze(1)
        #     z = z.unsqueeze(0)

        h_state = self.latent2hidden(z)
        h_0 = torch.stack([h_state for _ in range(self.depth)])
        # decoder_output, _ = self.rnn(z, (h_0, self.c_0))
        z = z.unsqueeze(0)
        rnn_out, _ = self.rnn(z, h_0)
        rnn_out = rnn_out.contiguous().view(-1, self.hidden_dim)
        x = self.hidden2output(rnn_out)
        return x

        # h_state = self.latent2hidden(z)
        # h_0 = torch.stack([h_state for _ in range(self.depth)])
        # decoder_output, _ = self.base(self.decoder_inputs, h_0) # seq_length, bsz, hidden_dim
        # decoder_output = decoder_output[-1, :, :] # bsz, hidden_dim
        # x = self.hidden2output(decoder_output)
        # return x


NAME2ENCODER = {
    "normal": Encoder,
    "recurrent": RecurrentEncoder
}

NAME2DECODER = {
    "normal": Decoder,
    "recurrent": RecurrentGRUDecoder
    # "recurrent": RecurrentDecoder
}


class VAE(nn.Module):


    def __init__(self, data_dim, hidden_dim, latent_dim, name,
                 sequence_length,):
        """
        """
        super(VAE, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length

        # Construct model
        self.encoder = self._get_encoder(name)
        self.decoder = self._get_decoder(name)
        # self.log_p = self.create_log_p()

        self.C_eta = nn.Linear(1, 1)
        self.C_eta.apply(weights_init)

    def _get_encoder(self, name):
        return NAME2ENCODER[name](
            data_dim=self.data_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim)

    def _get_decoder(self, name):
        if name == "normal":
            return NAME2DECODER[name](
                data_dim=self.data_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
            )
        else:
            return NAME2DECODER[name](
                data_dim=self.data_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim,
                sequence_length=self.sequence_length,
            )

    def set_to_train(self):
        for param in self.parameters():
            param.requires_grad = True

    def create_log_p(self):
        return nn.Sequential(*[
            nn.Linear(self.data_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
                               ])

    def z_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        z_mu, z_logvar = self.encoder(x)
        z = self.z_sample(mu=z_mu, logvar=z_logvar)
        x_tilde = self.decoder(z)
        return x_tilde, z, z_mu, z_logvar

    # def get_log_p(self, z):
    #     return self.log_p(z)


