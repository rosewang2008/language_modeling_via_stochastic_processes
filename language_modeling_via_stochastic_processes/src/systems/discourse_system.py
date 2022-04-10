import os
import pickle
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from language_modeling_via_stochastic_processes.src import constants
from language_modeling_via_stochastic_processes.src.datasets import (
    wikihow,
    wikisection,
    recipe,
    tm2, 
    tickettalk,
    roc_stories
)
import pytorch_lightning as pl
import wandb
import datasets
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from transformers import AlbertTokenizer, AlbertModel

lm_path = os.path.join(
    constants.PATH2TRANSFORMERS, 
    'examples/pytorch/language-modeling'
    )
from run_time_clm import get_checkpoint


torch.autograd.set_detect_anomaly(True)

NAME2DATASET = {
    'wikisection': wikisection.WikisectionDiscourse,
    'wikihow': wikihow.WikihowDiscourse,
    'recipe': recipe.RecipeDiscourse,
    'stories': roc_stories.ROCStoriesDiscourse,
    'tm2': tm2.TM2Discourse,
    'tickettalk': tickettalk.TicketTalkDiscourse
}

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

class DiscourseSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_train_step = 0
        self.num_test_step = 0

        dataset_name = self.config.data_params.name
        if dataset_name == 'recipe':
            self.data_dir = constants.PATH2RECIPENLG
            self.all_dataset = datasets.load_dataset("recipe_nlg", data_dir=self.data_dir)['train']
        elif dataset_name == 'wikihow':
            self.data_name = constants.PATH2WIKIHOW
            with open(self.data_name, 'rb') as f:
                self.all_dataset = pickle.load(f)
        else: # wikisection
            self.all_dataset = None

        self.train_dataset = NAME2DATASET[dataset_name](
            train=True, all_dataset=self.all_dataset,
            filepath=self.config.data_params.train_path,
            one_hot_labels=True,
            config=self.config,
            tokenizer_name=self.config.model_params.language_encoder,
        )
        self.test_dataset = NAME2DATASET[dataset_name](
            train=False, all_dataset=self.all_dataset,
            filepath=self.config.data_params.test_path,
            one_hot_labels=True,
            config=self.config,
            tokenizer_name=self.config.model_params.language_encoder,
        )

        self.set_model()

    def set_model(self):
        model_name = self.config.model_params.encoder
        dim = 768

        if model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
            for p in self.model.bert.parameters():
                p.requires_grad = False
        elif model_name == "albert":
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.model = AlbertModel.from_pretrained('albert-base-v2')
            for p in self.model.encoder.parameters():
                p.requires_grad = False
        elif model_name == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2ForSequenceClassification.from_pretrained('gpt2')
            for p in self.model.parameters():
                p.requires_grad = False
        elif model_name == "sbert":
            self.model = SentenceTransformer('paraphrase-mpnet-base-v2')
            for p in self.model.parameters():
                p.requires_grad = False
        elif model_name == "simcse":
            self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
            self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
            for p in self.model.parameters():
                p.requires_grad = False
        else: # CL models
            self.base_model = get_checkpoint(
                dataset_name=self.config.data_params.name,
                latent_dim=self.config.model_params.latent_dim,
                base_model=self.config.model_params.language_encoder,
                sec_id=self.config.data_params.include_section_ids_in_tokenizer,
                token_size=len(self.train_dataset.tokenizer),
                filepath=constants.NAME2PRETRAINEDMODELPATH[self.config.model_params.pretrained_name]
            )
            dim = self.config.model_params.latent_dim

            for p in self.base_model.parameters():
                p.requires_grad = False

        # take in two embeddigns -> output
        self.classifier = nn.Linear(dim * 3, 2)
        self.sigmoid = nn.Sigmoid()
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
        model_name = self.config.model_params.encoder

        if model_name == 'bert':
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
            logits = self.classifier(feats)
            probs = self.sigmoid(logits)
            loss = self.loss_f(probs, label)
            acc = torch.mean((torch.argmax(label, dim=1) == torch.argmax(probs, dim=1)).float())

        elif model_name == "albert":
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
            logits = self.classifier(feats)
            probs = self.sigmoid(logits)
            loss = self.loss_f(probs, label)
            acc = torch.mean((torch.argmax(label, dim=1) == torch.argmax(probs, dim=1)).float())

        elif model_name == "gpt2":
            max_length= 1024
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
            logits = self.classifier(feats)
            probs = self.sigmoid(logits)
            loss = self.loss_f(probs, label)
            acc = torch.mean((torch.argmax(label, dim=1) == torch.argmax(probs, dim=1)).float())

        elif model_name == "sbert":
            feats_t = self.model.encode(obs_t, convert_to_tensor=True)
            feats_tp1 = self.model.encode(obs_tp1, convert_to_tensor=True)
            diffs = feats_tp1 - feats_t
            feats = torch.cat((feats_t, feats_tp1, diffs), dim=1)
            logits = self.classifier(feats)
            probs = self.sigmoid(logits)
            loss = self.loss_f(probs, label)
            acc = torch.mean((torch.argmax(label, dim=1) == torch.argmax(probs, dim=1)).float())

        elif model_name == "simcse":
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
            logits = self.classifier(feats)
            probs = self.sigmoid(logits)
            loss = self.loss_f(probs, label)
            acc = torch.mean((torch.argmax(label, dim=1) == torch.argmax(probs, dim=1)).float())

        else:
            input_ids_t, attention_mask_t = self.train_dataset.tokenize_caption(obs_t, device=self.device)
            input_ids_t = input_ids_t[:, :512]
            attention_mask_t = attention_mask_t[:, :512]
            feats_t = self.base_model.forward(input_ids=input_ids_t, attention_mask=attention_mask_t)
            input_ids_tp1, attention_mask_tp1 = self.train_dataset.tokenize_caption(obs_tp1, device=self.device)
            input_ids_tp1 = input_ids_tp1[:, :512]
            attention_mask_tp1 = attention_mask_tp1[:, :512]
            feats_tp1 = self.base_model.forward(input_ids=input_ids_tp1, attention_mask=attention_mask_tp1)
            diffs = feats_tp1 - feats_t
            feats = torch.cat((feats_t, feats_tp1, diffs), dim=1)
            logits = self.classifier(feats)
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
                   'epoch': self.trainer.current_epoch
                   }, step=self.num_train_step)
        self.num_train_step += 1

        if batch_idx == 0 : # only evaluate once every epoch
            print("Testing at current epoch: {}".format(self.trainer.current_epoch))
            start = time.time()
            for bat_i, bat in enumerate(self.test_dataloader()):
                bat['label'] = bat['label'].to(self.device)
                if len(bat['y_t']) > 1:
                    self.test_step(bat, 1) # 1 s.t. I don't print
            end = time.time()
            diff = end - start
            print(f"Eval took {diff}s")

        return loss

    def test_step(self, batch, i):
        loss, acc = self.get_losses_for_batch(batch, i)
        wandb.log({'test_loss': loss.cpu().detach().numpy(),
                   'test_acc': acc.cpu().detach().numpy(),
                   'epoch': self.trainer.current_epoch
                   }, step=self.num_test_step)
        self.num_test_step += 1
        return loss

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def test_dataloader(self):
        return create_dataloader(self.test_dataset, self.config, shuffle=False)

    def save(self, directory):
        if self.config.model_params.encoder == 'bert':
            classifier = self.model.classifier
        else:
            classifier = self.classifier1
        torch.save(classifier.state_dict(), os.path.join(directory, "classifier.pt"))
