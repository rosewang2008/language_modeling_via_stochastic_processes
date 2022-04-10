import os
import torch
import torch.nn as nn
import pickle

from language_modeling_via_stochastic_processes.src.objectives import infonce
from language_modeling_via_stochastic_processes.src import constants
from language_modeling_via_stochastic_processes.src.systems import brownian_bridge_system

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
    'wikisection': wikisection.WikisectionTPK,
    'recipe': recipe.RecipeTPK,
    'wikihow': wikihow.WikihowTPK,
    'roc_stories': roc_stories.ROCStoriesTPK,
    'tm2': tm2.TM2TPK,
    'tickettalk': tickettalk.TicketTalkTPK,
}

torch.autograd.set_detect_anomaly(True)

class InfoNCESystem(brownian_bridge_system.BrownianBridgeSystem):

    def __init__(self, config):
        super().__init__(config=config)
        # g_{ar}:
        #   - encode each sentence and pass through a GRU
        # g_{en}: GPT2 encoder
        dim = self.config.model_params.latent_dim
        self.g_ar = nn.GRU(input_size=dim,
                           hidden_size=2400, # default number in infoNCE for langauge
                           num_layers=3,
                           batch_first=True
                           )
        self.W_k = nn.Linear(2400, dim)
        self.model.g_ar = self.g_ar
        self.model.W_k = self.W_k

    def _set_dataset(self):
        """
        The dataset for InfoNCE is formatted as in the paper
        (https://arxiv.org/pdf/1807.03748v2.pdf) where
        """
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
    def save(self, directory):
        torch.save(self.model.mlp.state_dict(), os.path.join(directory, "mlp.pt"))
        torch.save(self.model.feature_extractor.state_dict(), os.path.join(directory, "feature_extractor.pt"))
        torch.save(self.model.g_ar.state_dict(), os.path.join(directory, "g_ar.pt"))
        torch.save(self.model.W_k.state_dict(), os.path.join(directory, "W_k.pt"))

    def get_context(self, batch):
        z_t = self.get_feats(batch['y_t']).unsqueeze(1)
        z_tm1 = self.get_feats(batch['y_tm1']).unsqueeze(1)
        z_tm2 = self.get_feats(batch['y_tm2']).unsqueeze(1)
        z_tm3 = self.get_feats(batch['y_tm3']).unsqueeze(1)
        # Pass through GPT2 -> then summarizing in 1 sequence.
        seq = torch.cat((z_tm3, z_tm2, z_tm1, z_t), dim=1) # bsz, seq_len, dim
        c_t = self.model.g_ar(seq)[0] # bsz, seq len, hidden_size
        c_t = c_t[:, -1, :] # get last element
        return c_t

    def get_losses_for_batch(self, batch, batch_idx):
        del batch_idx
        torch.cuda.empty_cache()
        # each batch has x_{t}, x_{t-1}, x_{t-2},
        c_t = self.get_context(batch)
        z_tpk = self.get_feats(batch['y_tpk'])
        loss_fn = infonce.InfoNCE(
            c_t=c_t,
            z_tpk=z_tpk,
            W_k=self.W_k,
            config=self.config
        )
        loss = loss_fn.get_loss()
        return loss
