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
from src.datasets.recipe import (
    RecipeNLGData
)
from src.datasets.wikihow import (
    WikihowData
)

import datasets
from src.models import tcl_model
from src.models import language

import sys
sys.path.append('/nlp/scr/rewang/transformers/examples/pytorch/language-modeling')
from run_time_clm import get_checkpoint

import pytorch_lightning as pl
import wandb
torch.autograd.set_detect_anomaly(True)

NAME2DATASET = {
    # "WikiOUData": wikisection.WikiOUData,
    # "StoriesOUData": wikisection.StoriesOUData,
    # "LongerWikiOUData": wikisection.LongerWikiOUData,
    # "Taskmaster": wikisection.Taskmaster
    "wikisection": wikisection.WikiOUData,
    "stories": wikisection.StoriesOUData,
    "long_wikisection": wikisection.LongerWikiOUData,
    "taskmaster": wikisection.Taskmaster
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

# TODO need to specify CL encoder
class ZeroShotSystem(pl.LightningModule):
    """
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_train_step = 0
        self.num_test_step = 0

        tokenizer_name = 'BERT' if 'bert' in self.config.fpath.lower() else 'GPT2'

        if self.config.dataset == 'wikisection':
            train_path = "/nlp/scr/rewang/Wikisection/processed/HGD_en_city_train.json"
            test_path = "/nlp/scr/rewang/Wikisection/processed/HGD_en_city_test.json"
        elif self.config.dataset == 'long_wikisection':
            train_path = "/nlp/scr/rewang/Wikisection/final/en_city_train.json"
            test_path = "/nlp/scr/rewang/Wikisection/final/en_city_test.json"
        elif self.config.dataset == 'stories':
            train_path = "/nlp/scr/rewang/nonstationarity/data/stories/writingPrompts/train_processed.json"
            test_path = "/nlp/scr/rewang/nonstationarity/data/stories/writingPrompts/test_processed.json"
        else: # dummy paths
            train_path = "/nlp/scr/rewang/Wikisection/processed/HGD_en_city_train.json"
            test_path = "/nlp/scr/rewang/Wikisection/processed/HGD_en_city_test.json"

        if self.config.dataset == 'recipe':
            self.data_dir="/nlp/scr/rewang/data/recipe_nlg/dataset"
            self.all_dataset = datasets.load_dataset("recipe_nlg", data_dir=self.data_dir)['train']
            self.train_dataset = RecipeNLGData(
                train=True,
                seed=self.config.data_params.data_seed,
                all_dataset=self.all_dataset)
            self.test_dataset = RecipeNLGData(
                train=False,
                seed=self.config.data_params.data_seed,
                all_dataset=self.all_dataset)
        elif self.config.dataset == 'wikihow':
            self.data_name ="/nlp/scr/rewang/data/wiki_how_data.pkl"
            with open(self.data_name, 'rb') as f:
                self.all_dataset = pickle.load(f)
            self.train_dataset = WikihowData(
                train=True,
                seed=self.config.data_params.data_seed,
                all_dataset=self.all_dataset)
            self.test_dataset = WikihowData(
                train=False,
                seed=self.config.data_params.data_seed,
                all_dataset=self.all_dataset)
        else: # wikisection
            self.all_dataset = None
            self.train_dataset = NAME2DATASET[self.config.dataset](
                train=True,
                # filepath=self.config.data_params.train_path,
                filepath=train_path,
                unit="sentence",
                data_dim=None,
                n_segments=None,
                n_obs_seg=None,
                config=self.config,
                # tokenizer_name=self.config.model_params.language_encoder,
                tokenizer_name=tokenizer_name,
                seed=0,
                one_hot_labels=False,
            )
            self.test_dataset = NAME2DATASET[self.config.dataset](
                train=False,
                # filepath=self.config.data_params.test_path,
                filepath=test_path,
                unit="sentence",
                data_dim=None,
                n_segments=None,
                n_obs_seg=None,
                config=self.config,
                # tokenizer_name=self.config.model_params.language_encoder,
                tokenizer_name=tokenizer_name,
                seed=0,
                one_hot_labels=False,
            )

        print('train:', self.train_dataset)
        print('test:', self.test_dataset)
        self.set_model()

    def set_model(self):
        self.base_model = get_checkpoint(
            dataset_name=self.config.dataset,
            latent_dim=self.config.data_params.latent_dim,
            with_norm=True,
            base_model=self.config.model_params.language_encoder,
            sec_id=self.config.use_section_ids, # section ids used in the raw input text
            token_size=len(self.train_dataset.tokenizer),
            filepath=self.config.fpath
        )
        self.base_model.eval()


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

