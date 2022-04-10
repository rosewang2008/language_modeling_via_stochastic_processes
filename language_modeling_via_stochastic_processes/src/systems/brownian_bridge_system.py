import os
import pickle
from language_modeling_via_stochastic_processes.src import constants

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb

from language_modeling_via_stochastic_processes.src.datasets import (
    wikisection,
    recipe,
    wikihow,
    tm2,
    tickettalk,
    roc_stories
)

import datasets

NAME2DATASET = {
    'wikisection': wikisection.WikisectionTriplet,
    'recipe': recipe.RecipeTriplet,
    'wikihow': wikihow.WikihowTriplet,
    'roc_stories': roc_stories.ROCStoriesTriplet,
    'tm2': tm2.TM2Triplet,
    'tickettalk': tickettalk.TicketTalkTriplet,
}

from language_modeling_via_stochastic_processes.src.models import language
from language_modeling_via_stochastic_processes.src.objectives import brownian_bridge

torch.autograd.set_detect_anomaly(True)

def create_dataloader(dataset, config, shuffle=True):
    loader = DataLoader(
        dataset,
        batch_size=config.optim_params.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=shuffle,
        num_workers=config.experiment_params.data_loader_workers,
    )
    return loader

class BrownianBridgeSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._set_dataset()
        self._set_language_encoder()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum)
        return [optimizer], []

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def test_dataloader(self):
        return create_dataloader(self.test_dataset, self.config, shuffle=False)

    def _set_dataset(self):
        dname = self.config.data_params.name
        if 'recipe' == dname:
            self.data_dir = constants.PATH2RECIPENLG
            self.all_dataset = datasets.load_dataset("recipe_nlg", data_dir=self.data_dir)['train']
        elif 'wikihow' == dname:
            self.data_name = constants.PATH2WIKIHOW
            with open(self.data_name, 'rb') as f:
                self.all_dataset = pickle.load(f)
        else:
            self.all_dataset = None

        dataset = NAME2DATASET[dname]
        self.train_dataset = dataset(
            train=True,
            seed=self.config.data_params.data_seed,
            all_dataset=self.all_dataset,
            config=self.config
        )
        self.test_dataset = dataset(
            train=False,
            seed=self.config.data_params.data_seed,
            all_dataset=self.all_dataset,
            config=self.config
        )

    def set_to_train(self):
        pass

    def _set_language_encoder(self):
        self.model = language.GPT2OUEncoder(
            hidden_dim=self.config.model_params.hidden_size,
            latent_dim=self.config.model_params.latent_dim,
            finetune_gpt2=False)

        self.model.model.resize_token_embeddings(len(self.train_dataset.tokenizer))
        for p in self.model.model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        feats = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        return feats

    def get_feats(self, obs):
        input_ids_i, attention_mask_i = self.train_dataset.tokenize_caption(
            obs, device=self.device)
        input_ids_i = input_ids_i[:, :self.train_dataset.max_length]
        attention_mask_i = attention_mask_i[:, :self.train_dataset.max_length]
        feats_i = self.forward(input_ids=input_ids_i, attention_mask=attention_mask_i)
        return feats_i

    def get_losses_for_batch(self, batch, batch_idx):
        torch.cuda.empty_cache()
        obs_0 = batch['y_0']
        obs_t = batch['y_t']
        obs_T = batch['y_T']
        t_s = batch['t_'].float()
        ts = batch['t'].float()
        Ts = batch['T'].float()
        feats_0 = self.get_feats(obs_0)
        feats_t = self.get_feats(obs_t)
        feats_T = self.get_feats(obs_T)
        log_q_y_tp1 = self.model.get_log_q(feats_t)
        loss_fn = brownian_bridge.BrownianBridgeLoss(
            z_0=feats_0,
            z_t=feats_t,
            z_T=feats_T,
            t_=t_s,
            t=ts,
            T=Ts,
            alpha=0,
            var=0,
            log_q_y_T=log_q_y_tp1,
            loss_type=self.config.loss_params.name,
            eps=self.config.model_params.eps,
            max_seq_len=batch['total_t'].float(),
        )
        loss = loss_fn.get_loss()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, batch_idx)
        wandb.log({'train_loss': loss.cpu().detach().numpy(),
                   'epoch': self.trainer.current_epoch})
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        return loss

    def test_step(self, batch, i):
        loss = self.get_losses_for_batch(batch=batch, batch_idx=i)
        wandb.log({'test_loss': loss.cpu().detach().numpy(),
                   'epoch': self.trainer.current_epoch})
        self.log('test_loss', loss.cpu().detach().numpy(), prog_bar=True, on_step=True)
        return loss

    def save(self, directory):
        torch.save(self.model.mlp.state_dict(), os.path.join(directory, "mlp.pt"))
        torch.save(self.model.feature_extractor.state_dict(), os.path.join(directory, "feature_extractor.pt"))
