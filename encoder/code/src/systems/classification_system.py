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
    wikisection,
)

import datasets
from src.models import tcl_model
from src.models import language
from src.models.utils import weights_init
from src.evaluation import utils as evaluate
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import BertTokenizer, BertForNextSentencePrediction
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer

import sys
sys.path.append('/nlp/scr/rewang/transformers/examples/pytorch/language-modeling')
from run_time_clm import get_checkpoint

import pytorch_lightning as pl
import wandb
torch.autograd.set_detect_anomaly(True)


NAME2DATASET = {
    'wikisection': wikisection.WikisectionClassification,
    'wikisection_eos': wikisection.WikisectionEOS,
    'long_wikisection': wikisection.LongWikisectionClassification,
    'long_wikisection_eos': wikisection.LongWikisectionEOS,
    'stories': None,
}

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

class ClassificationSystem(pl.LightningModule):
    """
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_train_step = 0
        self.num_test_step = 0

        self.one_hot_labels =True  # (self.config.model != "bert")
        self.all_dataset = None

        self.train_dataset = NAME2DATASET[self.config.dataset](
            train=True, all_dataset=self.all_dataset, filepath=self.config.data_params.train_path,
            one_hot_labels=self.one_hot_labels,
            config=self.config,
            tokenizer_name=self.config.model_params.language_encoder,
        )
        self.test_dataset = NAME2DATASET[self.config.dataset](
            train=False, all_dataset=self.all_dataset, filepath=self.config.data_params.test_path,
            one_hot_labels=self.one_hot_labels,
            config=self.config,
            tokenizer_name=self.config.model_params.language_encoder,
        )

        self.set_model()

    def set_model(self):
        if self.config.fpath == "sbert":
            self.base_model = SentenceTransformer('paraphrase-mpnet-base-v2')
            dim = 768
        elif self.config.fpath == "simcse":
            self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
            self.base_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
            dim = 768
        else: # cl
            self.base_model = get_checkpoint(
                dataset_name=self.config.dataset,
                latent_dim=self.config.data_params.latent_dim,
                base_model=self.config.model_params.language_encoder,
                sec_id=self.config.use_section_ids, # section ids used in the raw input text
                token_size=len(self.train_dataset.tokenizer),
                filepath=self.config.fpath
            )
            dim = self.config.data_params.latent_dim

        for p in self.base_model.parameters():
            p.requires_grad = False

        self.classifier = nn.Linear(dim, self.train_dataset.num_labels)
        self.sigmoid= nn.Sigmoid()
        self.loss_f = nn.BCELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum)
        return [optimizer], []

    def get_losses_for_batch(self, batch, batch_idx):
        max_length = 1024
        obs_t = batch['y_t']
        label = batch['label']

        if self.config.fpath == "sbert":
            feats = self.base_model.encode(obs_t, convert_to_tensor=True)
        elif self.config.fpath == "simcse":
            inputs = self.tokenizer(obs_t, padding=True, truncation=True, return_tensors="pt")
            inputs['input_ids'] = inputs['input_ids'].to(self.base_model.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.base_model.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.base_model.device)
            feats = self.base_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        else: # cl
            input_ids_t, attention_mask_t = self.train_dataset.tokenize_caption(obs_t, device=self.device)
            input_ids_t = input_ids_t[:, :max_length]
            attention_mask_t = attention_mask_t[:, :max_length]
            feats = self.base_model.forward(input_ids=input_ids_t, attention_mask=attention_mask_t)

        logits = self.classifier(feats)
        probs = self.sigmoid(logits)
        loss = self.loss_f(probs, label)
        acc = torch.mean((torch.argmax(label, dim=1) == torch.argmax(probs, dim=1)).float())

        return loss, acc

    def training_epoch_end(self, training_step_outputs):
        epoch_acc = np.mean([i['acc'].item() for i in training_step_outputs])
        wandb.log({'train/acc': epoch_acc, 'epoch': self.trainer.current_epoch})

    def training_step(self, batch, batch_idx):
        loss, acc = self.get_losses_for_batch(batch, batch_idx)
        wandb.log({
            'TRAIN/Loss': loss.item(),
            'TRAIN/Acc': acc.item(),
            'epoch': self.trainer.current_epoch
           }, step=self.num_train_step)
        self.num_train_step += 1

        if batch_idx == 0 : # only evaluate once every epoch
            print("Testing at current epoch: {}".format(self.trainer.current_epoch))
            start = time.time()
            all_accs = []
            for bat_i, bat in enumerate(self.test_dataloader()):
                bat['label'] = bat['label'].to(self.device)
                info = self.test_step(bat, 1) # 1 s.t. I don't print
                all_accs.append(info['acc'].item())
            wandb.log({'test/acc': np.array(all_accs).mean(), })
            end = time.time()
            diff = end - start
            print(f"Eval took {diff}s")

        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, training_step_outputs):
        epoch_acc = np.mean([i['acc'].item() for i in training_step_outputs])
        wandb.log({'val/acc': epoch_acc, 'epoch': self.trainer.current_epoch})

    def test_step(self, batch, i):
        loss, acc = self.get_losses_for_batch(batch, i)
        wandb.log({'VALID/Loss': loss.cpu().detach().numpy(),
                   'VALID/Acc': acc.cpu().detach().numpy(),
                   'epoch': self.trainer.current_epoch
                   }, step=self.num_test_step)
        self.num_test_step += 1
        return {'loss': loss, 'acc': acc}

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def test_dataloader(self):
        return create_dataloader(self.test_dataset, self.config, shuffle=False)

    def save(self, directory):
        torch.save(self.classifier.state_dict(), os.path.join(directory, "classifier.pt"))


class BertNSPSystem(ClassificationSystem):
    """
    """

    def __init__(self, config):
        super().__init__(config)

    def set_model(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        pass

    def get_losses_for_batch(self, batch, batch_idx):
        max_length = 1024
        obs_t = batch['y_t']
        label = batch['label']

        encoding = tokenizer(prompt, next_sentence, return_tensors='pt')
        outputs = model(**encoding, labels=torch.LongTensor([1]))

        if self.config.fpath == "sbert":
            feats = self.base_model.encode(obs_t, convert_to_tensor=True)
        elif self.config.fpath == "simcse":
            inputs = self.tokenizer(obs_t, padding=True, truncation=True, return_tensors="pt")
            inputs['input_ids'] = inputs['input_ids'].to(self.base_model.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.base_model.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.base_model.device)
            feats = self.base_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        else: # cl
            input_ids_t, attention_mask_t = self.train_dataset.tokenize_caption(obs_t, device=self.device)
            input_ids_t = input_ids_t[:, :max_length]
            attention_mask_t = attention_mask_t[:, :max_length]
            feats = self.base_model.forward(input_ids=input_ids_t, attention_mask=attention_mask_t)

        logits = self.classifier(feats)
        probs = self.sigmoid(logits)
        loss = self.loss_f(probs, label)
        acc = torch.mean((torch.argmax(label, dim=1) == torch.argmax(probs, dim=1)).float())

        return loss, acc
