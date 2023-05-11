import os
import math
import numpy as np
from dotmap import DotMap
from collections import OrderedDict
from typing import Callable, Optional
import wandb

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer
from pytorch_lightning.core.optimizer import LightningOptimizer
from src.datasets import load_dataset
from src.models import vae
from src.objectives.ou_objective import OULoss, BrownianBridgeLoss
# from src.objectives.vae import ReconstructionLoss, KLLoss

torch.autograd.set_detect_anomaly(True)


def create_dataloader(dataset, config, shuffle=True):
    loader = DataLoader(
        dataset,
        batch_size=config.optim_params.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=shuffle,
        num_workers=config.data_loader_workers,
    )
    return loader

class VAESystem(pl.LightningModule):
    """
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.train_dataset, self.test_dataset = load_dataset.get_datasets(self.config)
        self.model = vae.VAE(
            data_dim=self.config.data_params.data_dim,
            hidden_dim=self.config.data_params.hidden_dim,
            latent_dim=self.config.data_params.latent_dim,
            name=self.config.model_params.name,
            sequence_length=self.config.data_params.seq_len,
        )
        self.num_train_step = 0

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum)
        return [optimizer], []

    def forward(self, x):
        x_tilde, z, z_mu, z_logvar = self.model.forward(x)
        return x_tilde, z, z_mu, z_logvar

    def get_losses_for_batch(self, batch, train=True):
        x_t = batch['y_t'].float()
        x_tp1 = batch['y_tp1'].float()

        x_t_tilde, z_t, z_t_mu, z_t_logvar = self.forward(x_t)
        x_tp1_tilde, z_tp1, z_tp1_mu, z_tp1_logvar = self.forward(x_tp1)
        # reconstruction loss
        # r_loss1 = ReconstructionLoss(x=x_t, reconstruct_x=x_t_tilde)
        r_loss2 = ReconstructionLoss(x=x_tp1, reconstruct_x=x_tp1_tilde)
        # kl
        kl_1 = KLLoss(z=z_t, z_mu=z_t_mu, z_logvar=z_t_logvar)
        # kl_2 = KLLoss(z=z_tp1, z_mu=z_tp1_mu, z_logvar=z_tp1_logvar)
        # ou loss
        ou_loss = OULoss(
            preds_t=None, preds_tp1=None,
            log_q_y_tp1=0.0,
            x_t=z_t, x_tp1=z_tp1,
            dt=self.config.data_params.dt,
            sigma=self.config.data_params.sigma,
            eps=self.config.model_params.eps,
            loss_type=self.config.loss_params.name,
            C_eta=self.model.C_eta.weight)._logit(x_tp1=z_tp1, x_t=z_t)
        ou_loss = -ou_loss.mean()/z_t.shape[0]

        return {
            # 'reconstruction_loss_1': r_loss1.get_loss(),
            'reconstruction_loss': r_loss2.get_loss(),
            # 'prior_loss': prior_loss,
            'kl': kl_1.get_loss(), # NOTE bring this back in later
            # 'kl_2': kl_2.get_loss(),
            'ou_loss': ou_loss # NOTE bring this back in later
        }

    def training_step(self, batch, batch_idx):

        loss_dict = self.get_losses_for_batch(batch, train=True)
        wandb.log({k: v.cpu().detach().numpy() for k, v in loss_dict.items()},
                  step=self.num_train_step)

        loss = 0.0
        for k, v in loss_dict.items():
            loss += v

        wandb.log({"total_loss": loss.cpu().detach().numpy()},
                  step=self.num_train_step)

        self.log('train_loss', loss, prog_bar=True, on_step=True)
        self.num_train_step += 1
        return loss

    def save(self, directory):
        save_path = os.path.join(directory, "ou_model.pt")
        torch.save(self.model.state_dict(), save_path)
        print("Saved at {}".format(save_path))

    def test_step(self, batch, i):
        loss_dict = self.get_losses_for_batch(batch, train=True)
        wandb.log({k: v.cpu().detach().numpy() for k, v in loss_dict.items()},
                  step=self.num_train_step)

        loss = 0.0
        for k, v in loss_dict.items():
            loss += v
        wandb.log({'test_loss': loss.cpu().detach().numpy()}, step=i)
        self.log('test_loss', loss, prog_bar=True, on_step=True)
        return loss

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def test_dataloader(self):
        return create_dataloader(self.test_dataset, self.config, shuffle=False)


class VAEBridgeSystem(VAESystem):

    def __init__(self, config):
        # dt has to be 1./T
        # config.data_params.dt = 1./self.train_dataset._data.shape[-1]
        config.data_params.dt = 1./14871
        super().__init__(config=config)

    def get_losses_for_batch(self, batch, train=True):
        x_t = batch['y_t'].float()
        x_tp1 = batch['y_tp1'].float()
        ts = batch['t'].float()

        x_t_tilde, z_t, z_t_mu, z_t_logvar = self.forward(x_t)
        x_tp1_tilde, z_tp1, z_tp1_mu, z_tp1_logvar = self.forward(x_tp1)
        # reconstruction loss
        # r_loss1 = ReconstructionLoss(x=x_t, reconstruct_x=x_t_tilde)
        r_loss2 = ReconstructionLoss(x=x_tp1, reconstruct_x=x_tp1_tilde)
        # kl
        kl_1 = KLLoss(z=z_t, z_mu=z_t_mu, z_logvar=z_t_logvar)
        # kl_2 = KLLoss(z=z_tp1, z_mu=z_tp1_mu, z_logvar=z_tp1_logvar)
        # ou loss
        ou_loss = OUBridgeLoss(
            preds_t=None, preds_tp1=None,
            log_q_y_tp1=0.0,
            x_t=z_t, x_tp1=z_tp1,
            t=ts,
            dt=self.config.data_params.dt,
            sigma=self.config.data_params.sigma,
            eps=self.config.model_params.eps,
            loss_type=self.config.loss_params.name,
            C_eta=self.model.C_eta.weight)._logit(x_tp1=z_tp1, x_t=z_t)
        ou_loss = -ou_loss.mean()/z_t.shape[0]

        return {
            # 'reconstruction_loss_1': r_loss1.get_loss(),
            'reconstruction_loss': r_loss2.get_loss(),
            # 'prior_loss': prior_loss,
            'kl': kl_1.get_loss(), # NOTE bring this back in later
            # 'kl_2': kl_2.get_loss(),
            'ou_loss': ou_loss # NOTE bring this back in later
        }

