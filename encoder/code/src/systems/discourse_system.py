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

import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from pytorch_lightning.core.optimizer import LightningOptimizer

from src.datasets.synthetic_data import SyntheticData
from src.datasets import (
    wikihow,
    wikisection,
    recipe,
)

import datasets
from src.models import tcl_model
from src.models import language
from src.models.utils import weights_init
from src.evaluation import utils as evaluate
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertForNextSentencePrediction,
)
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer
from transformers import AlbertTokenizer, AlbertModel
import random

import random
import sys
sys.path.append('/nlp/scr/rewang/public_language_modeling_via_stochastic_processes/decoder/examples/pytorch/language-modeling')
from run_time_clm import get_checkpoint
import random

import random
import pytorch_lightning as pl
import wandb
torch.autograd.set_detect_anomaly(True)


NAME2DATASET = {
    'wikisection': wikisection.WikisectionDiscourse,
    'long_wikisection': wikisection.LongWikisectionDiscourse,
    'taskmaster': wikisection.TaskmasterDiscourse,
    'wikihow': wikihow.WikihowDiscourse,
    'recipe': recipe.RecipeDiscourse,
    'stories': wikisection.ROCStoriesDiscourse,
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

# TODO need to specify CL encoder
class DiscourseSystem(pl.LightningModule):
    """
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_train_step = 0
        self.num_validation_step = 0

        if self.config.dataset == 'recipe':
            self.data_dir="/nlp/scr/rewang/data/recipe_nlg/dataset"
            self.all_dataset = datasets.load_dataset("recipe_nlg", data_dir=self.data_dir)['train']
        elif self.config.dataset == 'wikihow':
            self.data_name ="/nlp/scr/rewang/data/wiki_how_data.pkl"
            with open(self.data_name, 'rb') as f:
                self.all_dataset = pickle.load(f)
        else: # wikisection
            self.all_dataset = None

        # self.one_hot_labels = (self.config.model != "bert")
        self.one_hot_labels =True  # (self.config.model != "bert")

        self.train_dataset = NAME2DATASET[self.config.dataset](
            train=True, all_dataset=self.all_dataset, filepath=self.config.data_params.train_path,
            one_hot_labels=self.one_hot_labels,
            config=self.config,
            tokenizer_name=self.config.model_params.language_encoder,
        )
        self.validation_dataset = NAME2DATASET[self.config.dataset](
            train=False, all_dataset=self.all_dataset, filepath=self.config.data_params.test_path,
            one_hot_labels=self.one_hot_labels,
            config=self.config,
            tokenizer_name=self.config.model_params.language_encoder,
        )

        self.set_model()

    def set_model(self):
        if self.config.model == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
            for p in self.model.bert.parameters():
                p.requires_grad = False #  self.config.overfit
            dim = 768
        elif self.config.model == "albert":
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.model = AlbertModel.from_pretrained('albert-base-v2')
            for p in self.model.encoder.parameters():
                p.requires_grad = False #  self.config.overfit
            dim = 768
            # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            # outputs = model(**inputs)
            # last_hidden_states = outputs.last_hidden_state
        elif self.config.model == 'nsp_bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
            dim = 768
        elif self.config.model == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2ForSequenceClassification.from_pretrained('gpt2')
            for p in self.model.parameters():
                p.requires_grad = False # self.config.overfit
            dim = 768
        elif self.config.model == "sbert":
            self.model = SentenceTransformer('paraphrase-mpnet-base-v2')
            dim = 768
            for p in self.model.parameters():
                p.requires_grad = False # self.config.overfit
        elif self.config.model == "simcse":
            self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
            self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
            dim = 768
            for p in self.model.parameters():
                p.requires_grad = False # self.config.overfit
        else: # CL models
            self.base_model = get_checkpoint(
                dataset_name=self.config.dataset,
                latent_dim=self.config.data_params.latent_dim,
                base_model=self.config.model_params.language_encoder,
                sec_id=self.config.use_section_ids, # section ids used in the raw input text
                token_size=len(self.train_dataset.tokenizer),
                filepath="encoder_models/" + self.config.fpath
            )
            dim = self.config.data_params.latent_dim

            for p in self.base_model.parameters():
                p.requires_grad = False #  self.config.overfit

        # take in two embeddigns -> output
        self.classifier3 = nn.Linear(dim * 3, 2)
        self.classifier2 = nn.Linear(dim * 2, 2)
        self.classifier1 = nn.Linear(dim, 2)

        self.sigmoid= nn.Sigmoid()
        self.loss_f = nn.BCELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum)
        return [optimizer], []

    def get_losses_for_batch(self, batch, batch_idx):
        obs_t = batch['y_t']
        obs_tp1 = batch['y_tp1']
        label = batch['label']

        if self.config.model == 'bert':
            # TRIAL 3: play around with latent concat
            if True:
                # t
                inputs_t = self.tokenizer(obs_t, return_tensors="pt",
                                          max_length=512, padding=True).to(self.device)
                inputs_t = {k: v[:, :512] for k, v in inputs_t.items()}
                feats_t = self.model.bert(**inputs_t)[1]
                # t + 1
                inputs_tp1 = self.tokenizer(obs_tp1, return_tensors="pt",
                                          max_length=512, padding=True).to(self.device)
                inputs_tp1 = {k: v[:, :512] for k, v in inputs_tp1.items()}
                feats_tp1 = self.model.bert(**inputs_tp1)[1]
                diffs = feats_tp1 -feats_t
                feats = torch.cat((feats_t, feats_tp1, diffs), dim=1)
                # logits = self.classifier2(feats)
                logits = self.classifier3(feats)
                probs = self.sigmoid(logits)
                loss = self.loss_f(probs, label)
                acc = torch.mean((torch.argmax(label, dim=1) == torch.argmax(probs, dim=1)).float())
            # TRIAL 1: use seq classification
            elif False:
                obs  = [o_t + " . " + o_tp1 for (o_t, o_tp1) in zip(obs_t, obs_tp1)]
                inputs = self.tokenizer(obs, return_tensors="pt", max_length=512, padding=True).to(self.device)
                inputs = {k: v[:, :512] for k, v in inputs.items()}
                outputs = self.model(**inputs, labels=label)
                loss = outputs.loss
                acc = torch.mean((label == torch.argmax(self.sigmoid(outputs.logits), dim=1)).float())
            # TRIAL 2: encode each sentence separately, then concat, then classify
            # Most close to current CL setup. Check whether it's even linearly separable.
            else:
                # t
                inputs_t = self.tokenizer(obs_t, return_tensors="pt",
                                          max_length=512, padding=True).to(self.device)
                inputs_t = {k: v[:, :512] for k, v in inputs_t.items()}
                feats_t = self.model.bert(**inputs_t)[1]

                # t + 1
                inputs_tp1 = self.tokenizer(obs_tp1, return_tensors="pt",
                                          max_length=512, padding=True).to(self.device)
                inputs_tp1 = {k: v[:, :512] for k, v in inputs_tp1.items()}
                feats_tp1 = self.model.bert(**inputs_tp1)[1]
                feats = torch.cat((feats_t, feats_tp1), dim=1)
                logits = self.classifier2(feats)
                probs = self.sigmoid(logits)
                loss = self.loss_f(probs, label)
                acc = torch.mean((torch.argmax(label, dim=1) == torch.argmax(probs, dim=1)).float())

        elif self.config.model == "albert":
            # t
            inputs_t = self.tokenizer(obs_t, return_tensors="pt",
                                      max_length=512, padding=True).to(self.device)
            inputs_t = {k: v[:, :512] for k, v in inputs_t.items()}
            feats_t = self.model(**inputs_t).last_hidden_state[:, -1, :]
            # t + 1
            inputs_tp1 = self.tokenizer(obs_tp1, return_tensors="pt",
                                      max_length=512, padding=True).to(self.device)
            inputs_tp1 = {k: v[:, :512] for k, v in inputs_tp1.items()}
            feats_tp1 = self.model(**inputs_tp1).last_hidden_state[:, -1, :]
            diffs = feats_tp1 -feats_t
            feats = torch.cat((feats_t, feats_tp1, diffs), dim=1)
            # logits = self.classifier2(feats)
            logits = self.classifier3(feats)
            probs = self.sigmoid(logits)
            loss = self.loss_f(probs, label)
            acc = torch.mean((torch.argmax(label, dim=1) == torch.argmax(probs, dim=1)).float())

        elif self.config.model == "nsp_bert":
            encoding = self.tokenizer(obs_t, obs_tp1, return_tensors="pt",
                                      max_length=512, padding=True).to(self.device)
            encoding = {k: v[:, :512] for k, v in encoding.items()}
            outputs = self.model(**encoding, labels=torch.argmax(label, dim=-1))
            probs = self.sigmoid(outputs.logits)
            acc = torch.mean((torch.argmax(label, dim=-1) == torch.argmax(probs, dim=1)).float())
            loss = outputs.loss

        elif self.config.model == "gpt2":
            # Double checking that normal gpt2 can/cant do this task
            # TRIAL 1: CONCAT FEATS
            max_length= 1024
            if True:
                inputs_t = self.tokenizer(obs_t, return_tensors="pt",
                                          max_length=max_length, padding=True).to(self.device)
                inputs_t = {k: v[:, :max_length] for k, v in inputs_t.items()}
                feats_t = self.model.transformer(**inputs_t)[0]
                feats_t = feats_t[:, -1] # Take last hidden state

                # t + 1
                inputs_tp1 = self.tokenizer(obs_tp1, return_tensors="pt",
                                          max_length=max_length, padding=True).to(self.device)
                inputs_tp1 = {k: v[:, :max_length] for k, v in inputs_tp1.items()}
                feats_tp1 = self.model.transformer(**inputs_tp1)[0]
                feats_tp1 = feats_tp1[:, -1] # Take last hidden state

                diffs = feats_tp1 - feats_t
                feats = torch.cat((feats_t, feats_tp1, diffs), dim=1)
                logits = self.classifier3(feats)
                # feats = torch.cat((feats_t, feats_tp1), dim=1)
                # logits = self.classifier2(feats)
                probs = self.sigmoid(logits)
                loss = self.loss_f(probs, label)
                acc = torch.mean((torch.argmax(label, dim=1) == torch.argmax(probs, dim=1)).float())
            # TRIAL 2: CONCAT SENTENCES
            # Two things: try encoding separately and concating, and try encoding together.
            pass

        elif self.config.model == "sbert":
            feats_t = self.model.encode(obs_t, convert_to_tensor=True)
            feats_tp1 = self.model.encode(obs_tp1, convert_to_tensor=True)
            diffs = feats_tp1 - feats_t
            feats = torch.cat((feats_t, feats_tp1, diffs), dim=1)
            logits = self.classifier3(feats)
            probs = self.sigmoid(logits)
            loss = self.loss_f(probs, label)
            acc = torch.mean((torch.argmax(label, dim=1) == torch.argmax(probs, dim=1)).float())

        elif self.config.model == "simcse":
            # t
            inputs = self.tokenizer(obs_t, padding=True, truncation=True, return_tensors="pt")
            inputs['input_ids'] = inputs['input_ids'].to(self.model.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.model.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.model.device)
            feats_t = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            # tp1
            inputs = self.tokenizer(obs_tp1, padding=True, truncation=True, return_tensors="pt")
            inputs['input_ids'] = inputs['input_ids'].to(self.model.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.model.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.model.device)
            feats_tp1 = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            # diffs
            diffs = feats_tp1 - feats_t
            feats = torch.cat((feats_t, feats_tp1, diffs), dim=1)
            logits = self.classifier3(feats)
            probs = self.sigmoid(logits)
            loss = self.loss_f(probs, label)
            acc = torch.mean((torch.argmax(label, dim=1) == torch.argmax(probs, dim=1)).float())

        else:
            # TRIAL 2: put sentence together.
            # Kinda OOD from what was done in training...
            if False:
                obs  = [o_t + " . " + o_tp1 for (o_t, o_tp1) in zip(obs_t, obs_tp1)]
                input_ids, attention_mask = self.train_dataset.tokenize_caption(
                    obs, device=self.device)
                input_ids = input_ids[:, :512]
                attention_mask = attention_mask[:, :512]
                feats = self.base_model.forward(input_ids=input_ids,
                                                attention_mask=attention_mask)
                logits = self.classifier1(feats)
                probs = self.sigmoid(logits)
                loss = self.loss_f(probs, label)
                acc = torch.mean((torch.argmax(label, dim=1) == torch.argmax(probs, dim=1)).float())
            # TRIAL 1: concat feats together.
            if True:
                input_ids_t, attention_mask_t = self.train_dataset.tokenize_caption(obs_t, device=self.device)
                input_ids_t = input_ids_t[:, :512]
                attention_mask_t = attention_mask_t[:, :512]
                feats_t = self.base_model.forward(input_ids=input_ids_t, attention_mask=attention_mask_t)
                input_ids_tp1, attention_mask_tp1 = self.train_dataset.tokenize_caption(obs_tp1, device=self.device)
                input_ids_tp1 = input_ids_tp1[:, :512]
                attention_mask_tp1 = attention_mask_tp1[:, :512]
                feats_tp1 = self.base_model.forward(input_ids=input_ids_tp1, attention_mask=attention_mask_tp1)
                diffs = feats_tp1 - feats_t

                # feats = torch.cat((feats_t, feats_tp1), dim=1)
                feats = torch.cat((feats_t, feats_tp1, diffs), dim=1)
                # logits = self.classifier2(feats)
                logits = self.classifier3(feats)
                probs = self.sigmoid(logits)
                loss = self.loss_f(probs, label)
                acc = torch.mean((torch.argmax(label, dim=1) == torch.argmax(probs, dim=1)).float())


        acc_all = (torch.argmax(label, dim=1) == torch.argmax(probs, dim=1))
        if batch_idx % int(len(self.train_dataset)/ (len(obs_t) * 10)) == 0 :
            for o_t, o_tp1, l, a in zip(obs_t, obs_tp1, label, acc_all):
                print("label: {} | acc: {} | o_t: {} | o_tp1: {}".format(l, a, o_t, o_tp1))

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.get_losses_for_batch(batch, batch_idx)
        wandb.log({'train_loss': loss.cpu().detach().numpy(),
                   'train_acc': acc.cpu().detach().numpy(),
                   'epoch': self.trainer.current_epoch,
                   'train_step': self.num_train_step
                   })# , step=self.num_train_step)
        self.num_train_step += 1

        # if batch_idx == 0 : # only evaluate once every epoch
        #     print("Testing at current epoch: {}".format(self.trainer.current_epoch))
        #     start = time.time()
        #     for bat_i, bat in enumerate(self.test_dataloader()):
        #         bat['label'] = bat['label'].to(self.device)
        #         if len(bat['y_t']) > 1:
        #             self.validation_step(bat, 1) # 1 s.t. I don't print
        #     end = time.time()
        #     diff = end - start
        #     print(f"Eval took {diff}s")

        return loss

    def validation_step(self, batch, i):
        loss, acc = self.get_losses_for_batch(batch, i)
        wandb.log({
            'test_loss': loss.cpu().detach().numpy(),
                   'test_acc': acc.cpu().detach().numpy(),
                   'epoch': self.trainer.current_epoch,
                   'validation_step': self.num_validation_step
                   })# , step=self.num_validation_step)
        self.num_validation_step += 1
        return loss

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def val_dataloader(self):
        return create_dataloader(self.validation_dataset, self.config, shuffle=False)

    def save(self, directory):
        if self.config.model == 'bert':
            classifier = self.model.classifier
        else:
            classifier = self.classifier1
        torch.save(classifier.state_dict(), os.path.join(directory, "classifier.pt"))
