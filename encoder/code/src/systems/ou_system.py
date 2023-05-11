import os
import math
import numpy as np
from dotmap import DotMap
from collections import OrderedDict
from typing import Callable, Optional
import wandb
import random

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer
from pytorch_lightning.core.optimizer import LightningOptimizer
from src.datasets import load_dataset
from src.models import ou_model, vae
from src.objectives.ou_objective import (
    OULoss,
    BrownianBridgeLoss,
    BrownianBridgeEverythingLoss,
    BrownianBridgeRandomTLoss,
    BrownianBridgeRegLoss,
    BrownianBridgeRegPinLoss,
    BrownianBridgeRegBothLoss,
    NoisyOULoss
)

torch.autograd.set_detect_anomaly(True)


LOSSES = {
    'OULoss': OULoss,
    'BrownianBridgeLoss': BrownianBridgeLoss,
    'BrownianBridgeEverythingLoss': BrownianBridgeEverythingLoss,
    'BrownianBridgeRandomTLoss': BrownianBridgeRandomTLoss,
    'BrownianBridgeRegLoss': BrownianBridgeRegLoss,
    'BrownianBridgeRegPinLoss': BrownianBridgeRegPinLoss,
    'BrownianBridgeRegBothLoss': BrownianBridgeRegBothLoss,
}

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


def create_dataloader(dataset, config, shuffle=True):
    loader = DataLoader(
        dataset,
        batch_size=config.optim_params.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=shuffle,
        num_workers=config.data_loader_workers,
        worker_init_fn=seed_worker,
        generator=g
    )
    return loader

class OUSystem(pl.LightningModule):
    """
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.train_dataset, self.test_dataset = load_dataset.get_datasets(self.config)
        self.model = ou_model.OUModel(
            data_dim=self.config.data_params.data_dim,
            hidden_dim=self.config.data_params.hidden_dim,
            output_dim=self.config.data_params.data_dim,
        )

        self.num_train_step = 0


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum)
        return [optimizer], []

    def forward(self, observations):
        feats, preds = self.model.forward(observations)
        preds = torch.clamp(preds, -10, 10)
        return feats, preds

    def get_losses_for_batch(self, batch, train=True):
        y_t = batch['y_t'].float()
        y_tp1 = batch['y_tp1'].float()

        # preds = latent variables x_t
        feats_t, preds_t = self.forward(y_t)
        feats_tp1, preds_tp1 = self.forward(y_tp1)
        log_q_y_tp1 = self.model.get_log_q(y_tp1)


        loss_fn = OULoss(
            preds_t=preds_t,
            preds_tp1=preds_tp1,
            log_q_y_tp1=log_q_y_tp1,
            x_t=y_t,
            x_tp1=y_tp1,
            dt=self.config.data_params.dt,
            sigma=self.config.data_params.sigma,
            eps=self.config.model_params.eps,
            loss_type=self.config.loss_params.name
        )
        loss = loss_fn.get_loss()
        return loss

    def _print_debug(self, batch_idx):
        if not ("LT" in self.config.data_params.name):
            return

        pred_inv_A = self.model.predictor.weight.cpu().detach().numpy()
        real_inv_A = np.linalg.inv(self.train_dataset.A)
        if (batch_idx == 0):
            # Check for identity
            print("predicted inverse: {}".format(pred_inv_A))
            fb = pred_inv_A.dot(self.train_dataset.A)
            print("predicted_inverse.dot(true_transformation)=\n{}".format(fb))

            # Checking for rotation R between predicted inverse and true inverse
            # predicted_inverse = R real_inverse
            # <> predicted_inverse (real_inverse)^{-1} = R
            # R^T = R^{-1}i and real_inverse^{-1} = real_transformation A
            print("rotation matrix? check identity: {}".format(fb.dot(fb.T)))

        # Rotation determinant
        R = np.dot(pred_inv_A, self.train_dataset.A)
        rotation_det = np.linalg.det(R)
        wandb.log({"rotation_determinant": rotation_det,},)
        # step=self.num_train_step)

    def training_step(self, batch, batch_idx):
        self._print_debug(batch_idx)

        loss = self.get_losses_for_batch(batch, train=True)
        wandb.log({'train_loss': loss.cpu().detach().numpy(),
                   'epoch': self.num_train_step})
        # wandb.log({'train_loss': loss.cpu().detach().numpy(),
        #            'epoch': self.num_train_step},
        #           step=self.num_train_step)
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        self.num_train_step += 1
        return loss

    def save(self, directory):
        save_path = os.path.join(directory, "ou_model.pt")
        torch.save(self.model.state_dict(), save_path)
        print("Saved at {}".format(save_path))

    def test_step(self, batch, i):
        loss = self.get_losses_for_batch(batch, train=True)
        # wandb.log({'test_loss': loss.cpu().detach().numpy()}, step=i)
        wandb.log({'test_loss': loss.cpu().detach().numpy()})
        self.log('test_loss', loss, prog_bar=True, on_step=True)
        return loss

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def test_dataloader(self):
        return create_dataloader(self.test_dataset, self.config, shuffle=False)

class BrownianBridgeSystem(OUSystem):

    def __init__(self, config):
        # dt has to be 1./T
        config.data_params.dt = 1./config.data_params.samples_per_seq

        super().__init__(config=config)

    def get_losses_for_batch(self, batch, train=True):
        y_t = batch['y_t'].float()
        y_tp1 = batch['y_tp1'].float()
        ts = batch['t'].float()

        # preds = latent variables x_t
        feats_t, preds_t = self.forward(y_t)
        feats_tp1, preds_tp1 = self.forward(y_tp1)
        log_q_y_tp1 = self.model.get_log_q(y_tp1)

        # loss_fn = BrownianBridgeLoss(
        loss_fn = LOSSES[self.config.loss_params.loss](
            preds_t=preds_t,
            preds_tp1=preds_tp1,
            log_q_y_tp1=log_q_y_tp1,
            t=ts,
            x_t=y_t,
            x_tp1=y_tp1,
            dt=self.config.data_params.dt,
            sigma=self.config.data_params.sigma,
            eps=self.config.model_params.eps,
            loss_type=self.config.loss_params.name
        )
        loss = loss_fn.get_loss()
        return loss

class BrownianBridgeEverythingSystem(OUSystem):

    def __init__(self, config):
        config.data_params.dt = 1./config.data_params.samples_per_seq
        super().__init__(config=config)

    def get_losses_for_batch(self, batch, train=True):
        y_0 = batch['y_0'].float()
        y_t = batch['y_t'].float()
        y_T = batch['y_T'].float()
        t_ = batch['t_'].float()
        t = batch['t'].float()
        T = batch['T'].float()
        alpha = batch['alpha'].float()
        var = batch['var'].float()

        # preds = latent variables x_t
        feats_0, preds_0 = self.forward(y_0)
        feats_t, preds_t = self.forward(y_t)
        feats_T, preds_T = self.forward(y_T)
        log_q_y_T = self.model.get_log_q(y_T)

        # loss_fn = BrownianBridgeLoss(
        loss_fn = LOSSES[self.config.loss_params.loss](
            z_0=preds_0,
            z_t=preds_t,
            z_T=preds_T,
            t_=t_,
            t=t,
            T=T,
            alpha=alpha,
            var=var,
            log_q_y_T=log_q_y_T,
            loss_type=self.config.loss_params.name,
            eps=self.config.model_params.eps,
            max_seq_len=self.config.data_params.samples_per_seq
        )
        loss = loss_fn.get_loss()
        return loss

class NoisyOUSystem(OUSystem):

    def __init__(self, config):
        super().__init__(config=config)
        self.model = ou_model.NoisyOUModel(
            data_dim=self.config.data_params.data_dim,
            hidden_dim=self.config.data_params.hidden_dim,
            latent_dim=self.config.data_params.latent_dim,
            name=self.config.model_params.name,
            sequence_length=self.config.data_params.seq_len,
        )

    def forward(self, x):
        x_tilde, z, z_mu, z_logvar = self.model.forward(x)
        return x_tilde, z, z_mu, z_logvar

    def get_losses_for_batch(self, batch, train=True):
        y_t = batch['y_t'].float()
        y_tp1 = batch['y_tp1'].float()

        # preds = latent variables x_t
        (y_tilde_t, z_t, z_mu_t, z_logvar_t) = self.forward(y_t)
        (y_tilde_tp1, z_tp1, z_mu_tp1, z_logvar_tp1) = self.forward(y_tp1)
        log_q_y_tp1 = self.model.get_log_q(y_tp1)

        loss_fn = NoisyOULoss(
            preds_t=y_tilde_t,
            preds_tp1=y_tilde_tp1,
            log_q_y_tp1=log_q_y_tp1,
            x_t=y_t,
            x_tp1=y_tp1,
            z_t=z_t, z_mu_t=z_mu_t, z_logvar_t=z_logvar_t,
            z_tp1=z_tp1, z_mu_tp1=z_mu_tp1, z_logvar_tp1=z_logvar_tp1,
            dt=self.config.data_params.dt,
            sigma=self.config.data_params.sigma,
            eps=self.config.model_params.eps,
            loss_type=self.config.loss_params.name
        )
        loss = loss_fn.get_loss()
        return loss



class OUSystem_Classification(OUSystem):
    def __init__(self, config):
        super().__init__(config=config)

    def get_losses_for_batch(self, batch, train=True):
        y_t = batch['y_t'].float()
        y_tp1 = batch['y_tp1'].float()
        label = batch['label'].float()

        feats_t, preds_t = self.forward(y_t)
        feats_tp1, preds_tp1 = self.forward(y_tp1)
        log_q_y_tp1 = self.model.get_log_q(y_tp1)

        C_eta = self.model.C_eta.weight

        self.loss_fn = OULoss(
            preds_t=preds_t,
            preds_tp1=preds_tp1,
            label=label,
            C_eta=C_eta,
            log_q_y_tp1=log_q_y_tp1,
            x_t=y_t,
            x_tp1=y_tp1,
            dt=self.config.data_params.dt,
            sigma=self.config.data_params.sigma,
            eps=self.config.model_params.eps,
            loss_type=self.config.loss_params.name
        )
        loss = self.loss_fn.get_loss()
        return loss

    def training_step(self, batch, batch_idx):
        self._print_debug(batch_idx)

        loss = self.get_losses_for_batch(batch, train=True)
        # wandb.log({'train_loss': loss.cpu().detach().numpy(),
        #            'acc': self.loss_fn.acc,
        #            'epoch': self.num_train_step},
        #           step=self.num_train_step)
        wandb.log({'train_loss': loss.cpu().detach().numpy(),
                   'acc': self.loss_fn.acc,
                   'epoch': self.num_train_step})
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        self.num_train_step += 1
        return loss


class OUSystem_LearnMu(OUSystem):
    """Learning mu"""

    def __init__(self, config):
        super().__init__(config=config)

        self.model = ou_model.OUModel_Mu(
            data_dim=self.config.data_params.data_dim,
            hidden_dim=self.config.data_params.hidden_dim,
            output_dim=self.config.data_params.data_dim,
        )

    def forward(self, o_t, o_tp1):
        (feats_t, preds_t), (feats_tp1, preds_tp1) = self.model.forward(o_t, o_tp1)
        preds_t = torch.clamp(preds_t, -10, 10)
        preds_tp1 = torch.clamp(preds_tp1, -10, 10)
        return (feats_t, preds_t), (feats_tp1, preds_tp1)

    def get_losses_for_batch(self, batch, train=True):
        y_t = batch['y_t'].float()
        y_tp1 = batch['y_tp1'].float()

        (feats_t, preds_t), (feats_tp1, preds_tp1) = self.forward(o_t=y_t, o_tp1=y_tp1)

        loss_fn = LOSSES[self.config.loss_params.loss](
            preds_t=preds_t,
            preds_tp1=preds_tp1,
            x_t=y_t,
            x_tp1=y_tp1,
            dt=self.config.data_params.dt,
            sigma=self.config.data_params.sigma,
            eps=self.config.model_params.eps,
            loss_type=self.config.loss_params.name
        )
        loss = loss_fn.get_loss()
        return loss

    def training_step(self, batch, batch_idx):
        y_t = batch['y_t'].float()
        loss = self.get_losses_for_batch(batch, train=True)
        # wandb.log({'train_loss': loss.cpu().detach().numpy()}, step=self.num_train_step)
        wandb.log({'train_loss': loss.cpu().detach().numpy()})
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        self.num_train_step += 1
        return loss

