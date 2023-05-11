import os

import numpy as np
from sklearn.decomposition import FastICA

import os
import math
import numpy as np
from dotmap import DotMap
from collections import OrderedDict
from typing import Callable, Optional
import pickle

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from pytorch_lightning.core.optimizer import LightningOptimizer
from src import utils

from transformers import GPT2Config, GPT2Tokenizer, GPT2TimeLMHeadModel

from src.datasets.synthetic_data import SyntheticData
from src.datasets.wikisection import (
    WikiSectionData,
    BinaryWikiSectionData,
    WikiOUData,
    WikiTPKData,
    TaskmasterTPKData,
    WikiRandomTData,
    LongerWikiOUData,
    StoriesOUData,
    Taskmaster,
    TaskmasterRandomT,
    ROCStories,
    ROCStoriesRandomT,
    ROCStoriesTPKData,
)

from src.datasets.recipe import (
    RecipeNLGData,
    RecipeRandomT,
    RecipeTPKData,
)
from src.datasets.wikihow import (
    WikihowData,
    WikihowRandomT,
    WikihowTPKData,
)
import datasets
NAME2LOADER = {
    "WikiOUData": WikiOUData,
    "WikiRandomTData": WikiRandomTData,
    "WikiTPKData": WikiTPKData,
    "TaskmasterTPKData": TaskmasterTPKData,
    "StoriesOUData": StoriesOUData,
    "LongerWikiOUData": LongerWikiOUData,
    "Taskmaster": Taskmaster,
    "TaskmasterRandomT": TaskmasterRandomT,
    "ROCStories": ROCStories,
    "ROCStoriesRandomT": ROCStoriesRandomT,
    "ROCStoriesTPKData": ROCStoriesTPKData,
}

from src.models import tcl_model
from src.models import language
from src.models.utils import weights_init
from src.objectives.tcl import TCLLoss
from src.objectives.pcl import PCLLoss
from src.objectives import infonce
from src.objectives import vae
from src.objectives.ou_objective import * # BrownianBridgeLoss, BrownianBridgeRegBothLoss, OULoss
from src.evaluation import utils as evaluate
from src.evaluation import mcc

import random
import pytorch_lightning as pl
import wandb
import random
torch.autograd.set_detect_anomaly(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


NAME2LOSSES = {
    'BrownianBridgeLoss': BrownianBridgeLoss,
    'BrownianBridgeRegBothLoss': BrownianBridgeRegBothLoss,
    'BrownianBridgeRandomTLoss': BrownianBridgeRandomTLoss,
    'BrownianRandomTLoss': BrownianRandomTLoss,
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


class TCLSystem(pl.LightningModule):
    """
    Instance Discrimination.
    https://arxiv.org/abs/1805.01978
    We impose two memory banks (one for each modality) and
    optimize the two cross-modality objectives.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.train_dataset = SyntheticData(
            train=True, data_dim=self.config.data_params.data_dim,
            n_segments=self.config.data_params.n_segments,
            n_obs_seg=self.config.data_params.n_obs_per_seg,
            n_layers=self.config.data_params.n_layers,
            seed=self.config.data_params.data_seed,
            one_hot_labels=False)

        # NOTE: i'm using the same seed to generate the data - should be the same data
        self.test_dataset = SyntheticData(
            train=False, data_dim=self.config.data_params.data_dim,
            n_segments=self.config.data_params.n_segments,
            n_obs_seg=self.config.data_params.n_obs_per_seg,
            n_layers=self.config.data_params.n_layers,
            seed=self.config.data_params.data_seed,
            one_hot_labels=False)

        self.model = tcl_model.TCLModel(
            data_dim=self.config.data_params.data_dim,
            num_classes=self.config.data_params.n_segments,
            num_components=self.config.data_params.data_dim, # NOTE this is the same
            num_layers=self.config.model_params.n_layers,
            MLP_trainable=False
        )


        self.num_train_step = 0
        self.num_test_step = 0



    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum)
        return [optimizer], []

    def forward(self, observations):
        logits, feats = self.model.forward(observations)
        return logits, feats

    def get_losses_for_batch(self, batch, batch_idx):
        obs = batch['observation'].float()
        label = batch['label']

        logits, feats = self.forward(obs)
        loss_fn = TCLLoss(logits=logits, labels=label)
        loss = loss_fn.get_loss()
        acc = loss_fn.acc

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, batch_idx)
        wandb.log(
            {
                'train_loss': loss.cpu().detach().numpy(),
                'epoch': self.trainer.current_epoch,
                'step': self.num_train_step
            })

        self.num_train_step += 1

        if batch_idx == 0 and self.trainer.current_epoch % 10 == 0:
            utils.calculate_zero_shot(
                self.model, dataset=self.train_dataset,
                batch_size=self.config.optim_params.batch_size, notes=f"{self.trainer.current_epoch}_train")
            utils.calculate_zero_shot(
                self.model, dataset=self.test_dataset,
                batch_size=self.config.optim_params.batch_size, notes=f"{self.trainer.current_epoch}_test")

        # self.log('train_loss', loss, prog_bar=True, on_step=True)
        return loss

    def test_step(self, batch, i):
        obs = batch['observation'].float()
        source = batch['source'].float()
        label = batch['label']

        logits, feats = self.forward(obs)
        loss_fn = TCLLoss(logits=logits, labels=label)
        loss = loss_fn.get_loss()
        acc = loss_fn.acc

        predictions = torch.argmax(logits, 1)
        confmat = evaluate.calc_confusion_matrix(
            pred=predictions, label=label) # both should be bsz

        # Apply fastICA
        ica = FastICA(random_state=self.config.eval_params.seed)
        feats_ica = torch.tensor(ica.fit_transform(feats))

        # Evaluation metrics
        s2_mcc_no_ica = mcc.mean_corr_coef(feats, evaluate._squared(source))
        s2_mcc_ica = mcc.mean_corr_coef(feats_ica, evaluate._squared(source))
        print('Squared nonlinear function: TCL mcc (no ICA): {}\t mcc: {}'.format(
            s2_mcc_no_ica, s2_mcc_ica))
        abs_mcc_no_ica = mcc.mean_corr_coef(feats, evaluate._abs(source))
        abs_mcc_ica = mcc.mean_corr_coef(feats_ica, evaluate._abs(source))
        print('Absolute function: TCL mcc (no ICA): {}\t mcc: {}'.format(
            abs_mcc_no_ica, abs_mcc_ica))

        self.log('test_loss', loss, prog_bar=True, on_step=True)
        self.log('test_acc', acc, prog_bar=True, on_step=True)
        self.log('s2_mcc_no_ica', s2_mcc_no_ica, prog_bar=True, on_step=True)
        self.log('s2_mcc_ica', s2_mcc_ica, prog_bar=True, on_step=True)
        self.log('abs_mcc_no_ica', abs_mcc_no_ica, prog_bar=True, on_step=True)
        self.log('abs_mcc_ica', abs_mcc_ica, prog_bar=True, on_step=True)
        return loss

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def test_dataloader(self):
        return create_dataloader(self.test_dataset, self.config, shuffle=False)

class WikiTCLSystem(TCLSystem):
    """
    Instance Discrimination.
    https://arxiv.org/abs/1805.01978
    We impose two memory banks (one for each modality) and
    optimize the two cross-modality objectives.
    """

    def __init__(self, config):
        super().__init__(config=config)
        self._set_dataset()
        self._set_language_encoder()

    def _set_dataset(self):
        self.train_dataset = WikiSectionData(
            filepath=self.config.data_params.train_path,
            train=True,
            tokenizer_name=self.config.model_params.language_encoder,
            unit=self.config.data_params.unit,
            data_dim=self.config.data_params.data_dim,
            n_segments=self.config.data_params.n_segments,
            n_obs_seg=self.config.data_params.n_obs_per_seg,
            seed=self.config.data_params.data_seed,
            one_hot_labels=False)

        self.test_dataset = WikiSectionData(
            filepath=self.config.data_params.test_path,
            train=False,
            tokenizer_name=self.config.model_params.language_encoder,
            unit=self.config.data_params.unit,
            data_dim=self.config.data_params.data_dim,
            n_segments=self.config.data_params.n_segments,
            n_obs_seg=self.config.data_params.n_obs_per_seg,
            seed=self.config.data_params.data_seed,
            one_hot_labels=False)


    def _set_language_encoder(self):
        if self.config.model_params.language_encoder == "GPT2":
            self.model = language.GPT2LanguageEncoder(
                hidden_size=self.config.model_params.hidden_size,
                num_classes=self.config.data_params.n_segments)
        elif self.config.model_params.language_encoder == "BERT":
            self.model = language.BERTLanguageEncoder(
                hidden_size=self.config.model_params.hidden_size,
                num_classes=self.config.data_params.n_segments)
        else:
            raise ValueError("Dont recognize name {}".format(self.tokenizer_name))


    def forward(self, input_ids, attention_mask):
        feats, logits = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        return logits, feats

    def set_to_train(self):
        self.model.set_to_train()

    def get_losses_for_batch(self, batch, train=True):
        obs = batch['observation']
        sentences = obs
        input_ids, attention_mask = self.train_dataset.tokenize_caption(sentences, device=self.device)
        label = batch['label']

        logits, feats = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        loss_fn = TCLLoss(logits=logits, labels=label)
        loss = loss_fn.get_loss()
        acc = loss_fn.acc

        return loss

    def test_step(self, batch, i):
        obs = batch['observation']
        sentences = obs
        input_ids, attention_mask = self.train_dataset.tokenize_caption(sentences, device=self.device)
        label = batch['label']

        logits, feats = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        loss_fn = TCLLoss(logits=logits, labels=label)
        loss = loss_fn.get_loss()
        acc = loss_fn.acc

        self.log('test_loss', loss, prog_bar=True, on_step=True)
        self.log('test_acc', acc, prog_bar=True, on_step=True)
        return loss

    def save(self, directory):
        torch.save(self.model.mlp.state_dict(), os.path.join(directory, "mlp.pt"))
        torch.save(self.model.l2.state_dict(), os.path.join(directory, "l2.pt"))

class BinaryWikiSystem(WikiTCLSystem):
    """
    """

    def __init__(self, config):
        super().__init__(config=config)
        self.train_dataset = BinaryWikiSectionData(
            filepath=self.config.data_params.train_path,
            train=True,
            tokenizer_name=self.config.model_params.language_encoder,
            unit=self.config.data_params.unit,
            data_dim=self.config.data_params.data_dim,
            n_segments=self.config.data_params.n_segments,
            n_obs_seg=self.config.data_params.n_obs_per_seg,
            seed=self.config.data_params.data_seed,
            positive_rate=self.config.data_params.p,
            one_hot_labels=False)

        self.test_dataset = BinaryWikiSectionData(
            filepath=self.config.data_params.test_path,
            train=False,
            tokenizer_name=self.config.model_params.language_encoder,
            unit=self.config.data_params.unit,
            data_dim=self.config.data_params.data_dim,
            n_segments=self.config.data_params.n_segments,
            n_obs_seg=self.config.data_params.n_obs_per_seg,
            positive_rate=self.config.data_params.p,
            seed=self.config.data_params.data_seed,
            one_hot_labels=False)

        # binary predictor
        self.predictor = nn.Linear(self.model.hidden_size*2, 1) # sigmoid
        for param in self.predictor.parameters():
            param.requires_grad = False
        self.predictor.apply(weights_init) # zero bias

    def set_to_train(self):
        self.model.set_to_train()
        for param in self.predictor.parameters():
            param.requires_grad = True
        self.predictor.apply(weights_init) # zero bias

    def forward(self, input_ids, attention_mask):
        feats, _ = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        return feats

    def get_logits(self, batch):
        obs_i = batch['o_t']
        obs_j = batch['o_t*']

        input_ids_i, attention_mask_i = self.train_dataset.tokenize_caption(obs_i, device=self.device)
        feats_i = self.forward(input_ids=input_ids_i, attention_mask=attention_mask_i)
        input_ids_j, attention_mask_j = self.train_dataset.tokenize_caption(obs_j, device=self.device)
        feats_j = self.forward(input_ids=input_ids_j, attention_mask=attention_mask_j)
        feats = torch.cat((feats_i, feats_j), dim=1)
        logits = self.predictor(feats)
        return logits

    def get_losses_for_batch(self, batch, train=True):
        label = batch['label']
        logits = self.get_logits(batch)
        loss_fn = PCLLoss(logits=logits, labels=label)
        loss = loss_fn.get_loss()
        acc = loss_fn.acc
        return loss

    def test_step(self, batch, i):
        label = batch['label']
        logits = self.get_logits(batch)
        loss_fn = PCLLoss(logits=logits, labels=label)
        loss = loss_fn.get_loss()
        acc = loss_fn.acc
        wandb.log({
            'test_loss': loss.item(),
            'test_acc': acc.item(),
            'step': self.num_test_step
        })
        self.num_test_step += 1
        return loss

    def save(self, directory):
        torch.save(self.model.mlp.state_dict(), os.path.join(directory, "mlp.pt"))
        torch.save(self.predictor.state_dict(), os.path.join(directory, "predictor.pt"))

class WikiBrownianBridgeSystem(WikiTCLSystem):

    def __init__(self, config):
        super().__init__(config=config)
        self.dim_norm = False

    def _set_dataset(self):
        if 'recipe' in self.config.data_params.dataset_loader.lower():
            self.data_dir="/nlp/scr/rewang/data/recipe_nlg/dataset"
            self.all_dataset = datasets.load_dataset("recipe_nlg", data_dir=self.data_dir)['train']

            NAME2DATASET = {
                'RecipeNLGData': RecipeNLGData,
                'RecipeRandomT': RecipeRandomT,
                'RecipeTPKData': RecipeTPKData,
            }
            dataset = NAME2DATASET[self.config.data_params.dataset_loader]
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
        elif 'wikihow' in self.config.data_params.dataset_loader.lower():
            self.data_name ="/nlp/scr/rewang/data/wiki_how_data.pkl"

            with open(self.data_name, 'rb') as f:
                self.all_dataset = pickle.load(f)

            NAME2DATASET = {
                'WikihowData': WikihowData,
                'WikihowRandomT': WikihowRandomT,
                'WikihowTPKData': WikihowTPKData,
            }
            dataset = NAME2DATASET[self.config.data_params.dataset_loader]
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

        else:
            self.train_dataset = NAME2LOADER[self.config.data_params.dataset_loader](
                filepath=self.config.data_params.train_path,
                train=True,
                tokenizer_name=self.config.model_params.language_encoder,
                unit=self.config.data_params.unit,
                data_dim=self.config.data_params.data_dim,
                n_segments=self.config.data_params.n_segments,
                n_obs_seg=self.config.data_params.n_obs_per_seg,
                seed=self.config.data_params.data_seed,
                one_hot_labels=False,
                config=self.config
            )

            self.test_dataset = NAME2LOADER[self.config.data_params.dataset_loader](
                filepath=self.config.data_params.test_path,
                train=False,
                tokenizer_name=self.config.model_params.language_encoder,
                unit=self.config.data_params.unit,
                data_dim=self.config.data_params.data_dim,
                n_segments=self.config.data_params.n_segments,
                n_obs_seg=self.config.data_params.n_obs_per_seg,
                seed=self.config.data_params.data_seed,
                one_hot_labels=False,
                config=self.config
            )


    def set_to_train(self):
        pass

    def _set_language_encoder(self):
        # NOTE should finetune if i'm adding new section id tokens
        # finetune = self.config.use_section_ids
        finetune=False # to be efficient...
        if "GPT2"  == self.config.model_params.language_encoder:
            self.model = language.GPT2OUEncoder(
                hidden_dim=self.config.model_params.hidden_size,
                latent_dim=self.config.data_params.latent_dim,
                finetune_gpt2=finetune)
        else:
            self.model = language.BERTOUEncoder(
                hidden_dim=self.config.model_params.hidden_size,
                latent_dim=self.config.data_params.latent_dim,
                finetune=finetune)

        self.model.model.resize_token_embeddings(len(self.train_dataset.tokenizer))
        # NOTE This cut down model trainable parameters - need to figure out why
        # I thought I turned off all the param training in the class...
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


        if 'RandomT' in self.config.loss_params.loss:
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
            loss_fn = NAME2LOSSES[self.config.loss_params.loss](
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
                config=self.config
            )

        else:
            obs_i = batch['y_t']
            obs_j = batch['y_tp1']
            ts = batch['t'].float()
            dts = batch['dt'].float()
            feats_i = self.get_feats(obs_i)
            feats_j = self.get_feats(obs_j)
            log_q_y_tp1 = self.model.get_log_q(feats_j)
            loss_fn = NAME2LOSSES[self.config.loss_params.loss](
                preds_t=feats_i,
                preds_tp1=feats_j,
                log_q_y_tp1=log_q_y_tp1,
                t=ts,
                x_t=None,
                x_tp1=None,
                dt=dts,
                sigma=self.config.data_params.sigma,
                eps=self.config.model_params.eps,
                loss_type=self.config.loss_params.name
            )

        loss = loss_fn.get_loss()

        if self.dim_norm:
            loss = loss / self.config.data_params.latent_dim
        return loss

    def test_step(self, batch, i):
        loss = self.get_losses_for_batch(batch=batch, batch_idx=i)
        wandb.log({
            'test_loss': loss.cpu().detach().numpy(),
            'epoch': self.trainer.current_epoch,
            'step': self.num_test_step
        })
        self.num_test_step += 1
        return loss

    def save(self, directory):
        torch.save(self.model.mlp.state_dict(), os.path.join(directory, "mlp.pt"))
        torch.save(self.model.feature_extractor.state_dict(), os.path.join(directory, "feature_extractor.pt"))

class WikiInfoNCESystem(WikiBrownianBridgeSystem):

    def __init__(self, config):
        super().__init__(config=config)
        # g_{ar}:
        #   - encode each sentence and pass through a GRU
        # g_{en}: GPT2 encoder
        dim = self.config.data_params.latent_dim
        self.g_ar = nn.GRU(input_size=dim,
                           hidden_size=2400, # default number in infoNCE for langauge
                           num_layers=3,
                           batch_first=True
                           )
        self.W_k = nn.Linear(2400, dim)
        self.model.g_ar = self.g_ar
        self.model.W_k = self.W_k

    def save(self, directory):
        torch.save(self.model.mlp.state_dict(), os.path.join(directory, "mlp.pt"))
        torch.save(self.model.feature_extractor.state_dict(), os.path.join(directory, "feature_extractor.pt"))

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

        if self.dim_norm:
            loss = loss / self.config.data_params.latent_dim
        return loss

class WikiVAESystem(WikiBrownianBridgeSystem):

    def __init__(self, config):
        super().__init__(config=config)
        model_type = GPT2TimeLMHeadModel
        gpt2_config = GPT2Config()
        gpt2_config.use_contrastive_embeddings = True
        gpt2_config.debug_ids = False
        gpt2_config.embedding_type = "entireSection"
        gpt2_config.use_section_ids = False
        gpt2_config.use_section_null = False
        gpt2_config.use_noisy_embeddings = False
        gpt2_config.max_num_sections = len(self.train_dataset.section_names)
        gpt2_config.dataset_name = self.config.data_params.dataset_loader.lower()
        gpt2_config.cl_latent_dim = self.config.data_params.latent_dim
        self.time_model = model_type.from_pretrained('gpt2', config=gpt2_config)
        self.time_model.resize_token_embeddings(len(self.train_dataset.tokenizer))
        # TODO I should save the encoder & the LM
        self.model.fc_mu = nn.Linear(self.config.data_params.latent_dim, self.config.data_params.latent_dim)
        self.model.fc_var = nn.Linear(self.config.data_params.latent_dim, self.config.data_params.latent_dim)


    def save(self, directory):
        torch.save(self.model.mlp.state_dict(), os.path.join(directory, "mlp.pt"))
        torch.save(self.model.feature_extractor.state_dict(), os.path.join(directory, "feature_extractor.pt"))

    def get_losses_for_batch(self, batch, batch_idx):
        torch.cuda.empty_cache()
        # each batch has x_{t}, x_{t-1}, x_{t-2},
        # \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p(x_0|z_0)p(x_t|z_t)p(x_T|z_T)]
        # D_{\text{KL}}(q(z_t|z_0, z_T, x_t) \| p(z_t|z_0, z_T))
        # D_{\text{KL}}(q(z_0|x_0) \| p(z_0))
        # D_{\text{KL}}(q(z_T|x_T) \| p(z_T))

        obs_0 = batch['y_0']
        obs_t = batch['y_t']
        obs_T = batch['y_T']
        total_T = batch['total_t']
        t_s = batch['t_'].float()
        ts = batch['t'].float()
        Ts = batch['T'].float()
        z_0 = self.get_feats(obs_0)
        z_t = self.get_feats(obs_t)
        z_T = self.get_feats(obs_T)

        r_0 = vae.Reconstruction(
            obs=obs_0,
            z=z_0,
            decoder=self.time_model,
            config=self.config,
            tokenizer=self.train_dataset.tokenizer
        ).get_loss()

        r_t = vae.Reconstruction(
            obs=obs_t,
            z=z_t,
            decoder=self.time_model,
            config=self.config,
            tokenizer=self.train_dataset.tokenizer
        ).get_loss()

        r_T = vae.Reconstruction(
            obs=obs_T,
            z=z_T,
            decoder=self.time_model,
            config=self.config,
            tokenizer=self.train_dataset.tokenizer
        ).get_loss()

        loss_fn = vae.KL(
            z_0=z_0,
            z_t=z_t,
            z_T=z_T,
            t_=t_s,
            t=ts,
            T=Ts,
            total_t=total_T,
            fc_mu=self.model.fc_mu,
            fc_var=self.model.fc_var,
            config=self.config
        )
        kl_loss = loss_fn.get_loss()
        loss = r_0 + r_t + r_T + kl_loss

        wandb.log({'kl': kl_loss, 'r0': r_0, 'rt': r_t, 'rT': r_T})

        return loss


class WikiOUSystem(WikiBrownianBridgeSystem):

    def __init__(self, config):
        super().__init__(config=config)

    def get_losses_for_batch(self, batch, batch_idx):
        obs_i = batch['y_t']
        obs_j = batch['y_tp1']
        ts = batch['t'].float()
        dts = batch['dt'].float()

        input_ids_i, attention_mask_i = self.train_dataset.tokenize_caption(
            obs_i, device=self.device)
        feats_i = self.forward(input_ids=input_ids_i, attention_mask=attention_mask_i)
        input_ids_j, attention_mask_j = self.train_dataset.tokenize_caption(
            obs_j, device=self.device)
        feats_j = self.forward(input_ids=input_ids_j, attention_mask=attention_mask_j)

        log_q_y_tp1 = self.model.get_log_q(feats_j)

        loss_fn = OULoss(
            preds_t=feats_i,
            preds_tp1=feats_j,
            log_q_y_tp1=log_q_y_tp1,
            # t=ts,
            x_t=None,
            x_tp1=None,
            dt=dts,
            sigma=self.config.data_params.sigma,
            eps=self.config.model_params.eps,
            loss_type=self.config.loss_params.name
        )
        loss = loss_fn.get_loss()

        if self.dim_norm:
            loss = loss / self.config.data_params.latent_dim
        return loss



class StoriesBrownianBridgeSystem(WikiBrownianBridgeSystem):

    def __init__(self, config):
        super().__init__(config=config)
        self.train_dataset = StoriesOUData(
            filepath=self.config.data_params.train_path,
            train=True,
            tokenizer_name=self.config.model_params.language_encoder,
            unit=self.config.data_params.unit,
            data_dim=self.config.data_params.data_dim,
            n_segments=self.config.data_params.n_segments,
            n_obs_seg=self.config.data_params.n_obs_per_seg,
            seed=self.config.data_params.data_seed,
            one_hot_labels=False)

        self.test_dataset = StoriesOUData(
            filepath=self.config.data_params.test_path,
            train=False,
            tokenizer_name=self.config.model_params.language_encoder,
            unit=self.config.data_params.unit,
            data_dim=self.config.data_params.data_dim,
            n_segments=self.config.data_params.n_segments,
            n_obs_seg=self.config.data_params.n_obs_per_seg,
            seed=self.config.data_params.data_seed,
            one_hot_labels=False)

class BinaryDeltaWikiSystem(BinaryWikiSystem):
    """
    """

    def __init__(self, config):
        super().__init__(config=config)
        self.predictor = nn.Linear(self.model.hidden_size, 1) # sigmoid

        for param in self.predictor.parameters():
            param.requires_grad = False
        self.predictor.apply(weights_init) # zero bias

    def get_logits(self, batch):
        obs_i = batch['o_t']
        obs_j = batch['o_t*']

        input_ids_i, attention_mask_i = self.train_dataset.tokenize_caption(obs_i, device=self.device)
        feats_i = self.forward(input_ids=input_ids_i, attention_mask=attention_mask_i)
        input_ids_j, attention_mask_j = self.train_dataset.tokenize_caption(obs_j, device=self.device)
        feats_j = self.forward(input_ids=input_ids_j, attention_mask=attention_mask_j)
        feats = feats_j - feats_i
        feats = F.normalize(feats, dim=1)

        #########################
        # Direction invariance?
        #########################
        mask = torch.abs(batch['j'] - batch['i']).ge(2.0).repeat(feats.shape[-1], 1).T
        feats = torch.where(mask, feats*(-1), feats)

        logits = self.predictor(feats)
        return logits


class RecipeBrownianBridgeSystem(WikiBrownianBridgeSystem):

    def __init__(self, config):
        super().__init__(config=config)
        # self.model = language.GPT2OUEncoder(
        #     hidden_dim=self.config.model_params.hidden_size,
        #     latent_dim=self.config.data_params.latent_dim,
        #     finetune_gpt2=False)
        # self.dim_norm = False
        # self._load_data()
        # self.model.model.resize_token_embeddings(len(self.train_dataset.tokenizer))
        # for p in self.model.model.parameters():
        #     p.requires_grad = False


    def _set_dataset(self):
        self.data_dir="/nlp/scr/rewang/data/recipe_nlg/dataset"
        self.all_dataset = datasets.load_dataset("recipe_nlg", data_dir=self.data_dir)['train']

        NAME2DATASET = {
            'RecipeNLGData': RecipeNLGData,
            'RecipeRandomT': RecipeRandomT,
            'RecipeTPKData': RecipeTPKData,
        }
        dataset = NAME2DATASET[self.config.data_params.dataset_loader]
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


class WikihowBrownianBridgeSystem(RecipeBrownianBridgeSystem):

    def __init__(self, config):
        super().__init__(config=config)

    def _set_dataset(self):
        self.data_name ="/nlp/scr/rewang/data/wiki_how_data.pkl"

        with open(self.data_name, 'rb') as f:
            self.all_dataset = pickle.load(f)

        NAME2DATASET = {
            'WikihowData': WikihowData,
            'WikihowRandomT': WikihowRandomT,
        }
        dataset = NAME2DATASET[self.config.data_params.dataset_loader]
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

