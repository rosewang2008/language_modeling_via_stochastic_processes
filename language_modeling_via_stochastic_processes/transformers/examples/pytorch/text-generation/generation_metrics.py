
import re

from tqdm import tqdm
import torch
import numpy as np
from collections import defaultdict
import pandas as pd
import os
import wandb

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    BertForSequenceClassification
)
import sys
sys.path.append("../language-modeling/")
from run_time_clm import get_special_tokens

def get_classification_model(model_args):
    model_path = model_args.classification_model
    cache_dir = "/nlp/scr/rewang/huggingface/"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(model_args.device)
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_path,
    #     cache_dir=cache_dir,
    #     use_fast=True, # model_args.use_fast_tokenizer,
    #     revision="main", # model_args.model_revision,
    #     # use_auth_token=True if model_args.use_auth_token else None,
    #     use_auth_token=None,
    # )
    # config = AutoConfig.from_pretrained(
    #     # model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     model_path,
    #     num_labels=4, # num_labels,
    #     finetuning_task="wikisection", # data_args.task_name,
    #     cache_dir=cache_dir,
    #     revision="main",
    #     use_auth_token=None, # True if model_args.use_auth_token else None,
    # )
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_path,
    #     # model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_path),
    #     config=config,
    #     cache_dir=cache_dir,
    #     revision="main", # model_args.model_revision,
    #     use_auth_token=None,
    # )
    return tokenizer, model

class GenerationMetrics:

    def __init__(self, model, device, tokenizer, dataset_name, fname,
                 model_args,
                 subclass=''):
        if subclass:
            self.prepend_ = subclass + "/"
        else:
            self.prepend_ = ""
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self._max_length = model.config.n_positions
        self.dataset_name = dataset_name
        self.section_names, self.section_ids, _ = get_special_tokens(
            dataset_name=dataset_name, tokenizer=tokenizer, add_tokens=False )
        if ' . ' == self.section_names[-1]:
            self.section_names = self.section_names[:-1]
            self.section_ids = self.section_ids[:-1]
        self._info = []
        self._examples = []
        self._classification_examples = dict()
        self.metrics = defaultdict(lambda: [])
        self.examples = {}
        self.fname = fname

        self.classification_tokenizer, self.classification_model = get_classification_model(model_args)
        self.mode = "section" if ("splitsection" in dataset_name) else "doc"

    def calculate(self, input_ids, raw_seq, section_name=None,
                  cl_feats=None, section_id=None, gt_raw_seq=None):

        if 'stories' in self.dataset_name:
            self._stories(input_ids, raw_seq)
        elif self.mode == "doc": # wikisection
            self._document(input_ids=input_ids, raw_seq=raw_seq,
                           gt_raw_seq=gt_raw_seq)
            self._track_doc_examples(raw_seq)
        else:
            raise ValueError()
        self._examples.append({'text': raw_seq})

    def _stories(self, input_ids, raw_seq):
        # TODO get story classification
        # Check for redundancy in WP and Prompt
        info = {}
        for special_tok, name in zip([50257, 50258], ['[ WP ]', '[ RESPONSE ]']):
            idxs = (input_ids == special_tok).nonzero(as_tuple=True)
            is_not_null = len(idxs[-1]) > 0
            # present
            present = 1 if len(idxs) > 0 and is_not_null else 0
            self.metrics['{} present'.format(name)].append(present)
            info['{} present'.format(name)] = present
            # redundant
            redundant = 1 if len(idxs) > 1 and is_not_null else 0
            self.metrics['{} redundant'.format(name)].append(redundant)
            info['{} redundant'.format(name)] = redundant

        # check if response is there
        if len(idxs) > 0 and is_not_null:
            start_idx = idxs[-1].detach().cpu().numpy()[0]
            length = input_ids.shape[0] - start_idx
            self.metrics['{} length'.format(name)].append(length)
            info['{} length'.format(name)] = length

        most_recent = dict()
        df = pd.DataFrame(self._info)
        for key in df.keys():
            k = self.prepend_ + key
            most_recent[k + " recent"] = df[key].mean()
            most_recent[k + " ste recent"] = df[key].sem()
            wandb.run.summary[k + " recent"] = df[key].mean()
            wandb.run.summary[k + " ste recent"] = df[key].sem()
        wandb.log(most_recent)

    def _track_doc_examples(self, raw_seq):
        self.examples['ordering = {}'.format(self.metrics['ordering'][-1])] = raw_seq

        for k, v in self._classification_examples.items():
            self.examples[k] = v

        for section_i, section_name in enumerate(self.section_names):
            is_present = self.metrics['{} present'.format(section_name)]
            is_redundant = self.metrics['{} redundant'.format(section_name)]
            if is_present:
                self.examples[
                    '{} present = {}'.format(section_name, is_present[-1])] = raw_seq
            if is_redundant:
                self.examples[
                    '{} redundant = {}'.format(section_name, is_redundant[-1])] = raw_seq
            lengths = self.metrics['{} length'.format(section_name)]
            if not lengths:
                continue
            if np.max(lengths) == lengths[-1]: # max length sec
                self.examples['{} max length'.format(section_name)] = raw_seq
            if np.min(lengths) == lengths[-1]: # min length sec
                self.examples['{} min length'.format(section_name)] = raw_seq

    def _check_present(self, idxs, section_name, info):
        try:
            idx = idxs[0].detach().cpu().numpy()[0]
            is_present = True
        except:
            is_present = False

        self.metrics['{} present'.format(section_name)].append(is_present)
        info['{} present'.format(section_name)] = is_present
        return info

    def _check_redundancy(self, idxs, section_name, info):
        """
        False if only 1 idx
        True if > 1 idx
        None if not present, ie don't track.
        """
        if len(idxs[0]) > 1:
            is_redundant = True
        elif info['{} present'.format(section_name)]:
            is_redundant = False
        else:
            is_redundant = None

        if is_redundant is not None:
            self.metrics['{} redundant'.format(section_name)].append(is_redundant)
            info['{} redundant'.format(section_name)] = is_redundant
        return info

    def _check_total_length(self, input_ids, info):
        self.metrics['total length'] = input_ids.shape[-1]
        info['total length'] = input_ids.shape[-1]
        return info

    def _check_classification(self, raw_seq, info):
        classification_results = self._get_classification(raw_seq)
        histograms = dict()
        for k, v in classification_results.items():
            # if list, create a histogram and include mean
            if isinstance(v, list):
                histograms[self.prepend_ + k +  " hist"] = wandb.Histogram(v)
                v = np.mean(v)
            info[k] = v
        wandb.log(histograms)
        return info


    def _taskmaster_section_length(self, input_ids, idxs, section_name, info):
        lengths = []
        other_id = 50258 if 'USER' in section_name else 50257
        input_ids = input_ids.tolist()
        for start_idx in idxs:
            start_idx = start_idx.item()
            rest = input_ids[start_idx:]
            if other_id in rest:
                length = rest.index(other_id)
                lengths.append(length)
        length = np.mean(lengths)
        self.metrics['{} length'.format(section_name)].append(length)
        info['{} length'.format(section_name)] = length
        return info

    def _check_section_length(self, input_ids, idxs, section_name, info):
        """Only track length if it's present"""
        if info['{} present'.format(section_name)]:
            if 'taskmaster' in self.dataset_name:
                return self._taskmaster_section_length(input_ids, idxs[0], section_name, info)
            start_idx = idxs[0].detach().cpu().numpy()[0]
            # Check for next section id
            text_rest = input_ids[start_idx+1:].detach().cpu().numpy() # +1 because you want to skip current token
            other_idxs = [start_idx+1+idx for idx, token in enumerate(text_rest)
                          if token in self.section_ids]
            if other_idxs: # non empty list
                end_idx = min(other_idxs)
            else: # last section
                end_idx = input_ids.shape[-1]
            length = end_idx - start_idx
            self.metrics['{} length'.format(section_name)].append(length)
            info['{} length'.format(section_name)] = length
        return info

    def _taskmaster_ordering(self, input_ids, raw_seq, info):
        correct_ordering = True
        user = (input_ids == 50257).nonzero(as_tuple=True)[0]
        assistant = (input_ids == 50258).nonzero(as_tuple=True)[0]
        for a_id, b_id in zip(user, assistant):
            correct_ordering = correct_ordering and (a_id.item() < b_id.item())
        self.metrics['ordering'].append(correct_ordering)
        info['ordering'] = correct_ordering
        return info


    def _check_ordering(self, input_ids, raw_seq, info):
        # Check ordering
        if 'taskmaster' in self.dataset_name:
            return self._taskmaster_ordering(input_ids, raw_seq, info)
        section_order_idxs = []
        correct_ordering = True
        for section_name in self.section_names:
            if section_name in raw_seq:
                idxs = [i for i in range(len(raw_seq))
                        if raw_seq.startswith(section_name, i)]
                if len(idxs) > 1:
                    if 'wikihow' in self.dataset_name and 'STEP' in section_name:
                        # Get step numbers
                        step_num = [raw_seq[i:].split(' ')[4] for i in idxs]
                        # intify:
                        try:
                            step_num = [int(i) for i in step_num]
                            correct_ordering = all([step_num[i] < step_num[i+1] for i in range(len(step_num)-1)])
                        except:
                            correct_ordering = False # not an int following STEP
                    else:
                        correct_ordering = False
                section_order_idxs += idxs

        for i in range(len(section_order_idxs)):
            for j in range(len(section_order_idxs)):
                if j <= i: continue
                correct_ordering = (
                    correct_ordering and
                    section_order_idxs[i] < section_order_idxs[j])
        self.metrics['ordering'].append(correct_ordering)
        info['ordering'] = correct_ordering
        return info

    def _check_gt_present(self, generated_raw_seq, gt_raw_seq, section_name, info):
        """
        Only checks in long wikisection setting & if section_name in the ground truth sequence.

        is True if generated also contains the section
        False otherwise.

        This is not run on wikisection_filter because all sections need to present
        Long wikisection can have a mixture of sections - subset sections
        """
        # TODO
        if "long" in self.dataset_name:
            if section_name in gt_raw_seq:
                is_present = section_name in generated_raw_seq
                self.metrics['{} GT present'.format(section_name)].append(is_present)
                info['{} GT present'.format(section_name)] = is_present
        return info

    def _document(self, input_ids, raw_seq, gt_raw_seq):
        info = {}

        info = self._check_total_length(input_ids=input_ids, info=info)
        if 'taskmaster' not in self.dataset_name:
            info = self._check_classification(raw_seq=raw_seq, info=info)
        info = self._check_ordering(input_ids=input_ids, raw_seq=raw_seq, info=info)
        for section_id, section_name in zip(self.section_ids, self.section_names):
            idxs = (input_ids == section_id).nonzero(as_tuple=True)
            info = self._check_present(idxs=idxs, section_name=section_name, info=info)
            info = self._check_redundancy(idxs=idxs, section_name=section_name, info=info)
            info = self._check_section_length(
                input_ids=input_ids,
                idxs=idxs,
                section_name=section_name,
                info=info)

            info = self._check_gt_present(
                generated_raw_seq=raw_seq,
                gt_raw_seq=gt_raw_seq,
                section_name=section_name,
                info=info
            )


        # wandb.log({self.prepend_+k:v for k, v in info.items()})
        self._info.append(info)
        most_recent = dict()
        df = pd.DataFrame(self._info)
        for key in df.keys():
            k = self.prepend_ + key
            most_recent[k + " recent"] = df[key].mean()
            most_recent[k + " ste recent"] = df[key].sem()
            wandb.run.summary[k + " recent"] = df[key].mean()
            wandb.run.summary[k + " ste recent"] = df[key].sem()

        if "long" in self.dataset_name:
            most_recent['ABSTRACT GT'] = 88.65
            most_recent['GEOGRAPHY GT'] = 81.22
            most_recent['HISTORY GT'] = 189.63
            most_recent['CLIMATE GT'] = 81.87
            most_recent['GEOGRAPHY | CLIMATE GT'] = 87.56
            most_recent['DEMOGRAPHICS GT'] = 232
            most_recent['ECONOMY GT'] = 129.43
            most_recent['GOVERNMENT GT'] = 90.0
            most_recent['TRANSPORTATION GT'] = 111.21
            sec_names = ['ABSTRACT', 'HISTORY', 'GEOGRAPHY', 'DEMOGRAPHICS',
                         'CLIMATE', 'GEOGRAPHY | ClIMATE', 'ECONOMY', 'TRANSPORTATION',
                         'GOVERNMENT']
        elif 'wikihow' in self.dataset_name:
            most_recent['TITLE GT'] = 12.58
            most_recent['METHOD GT'] = 9.56
            most_recent['STEP GT'] = 101.02 # 466.10
            sec_names = ['TITLE', 'METHOD', 'STEP']
        elif 'recipe' in self.dataset_name:
            most_recent['TITLE GT'] = 9.67
            most_recent['INGREDIENTS GT'] = 23.81
            most_recent['DIRECTIONS GT'] = 61.96
            sec_names = ['TITLE', 'INGREDIENTS', 'DIRECTIONS']
        elif 'taskmaster' in self.dataset_name:
            most_recent['USER GT'] = 11.79
            most_recent['ASSISTANT GT'] = 17.96
            sec_names = ['USER', 'ASSISTANT']
        else:
            most_recent['ABSTRACT GT'] = 73.46
            most_recent['HISTORY GT'] = 179.51
            most_recent['GEOGRAPHY GT'] = 84.36
            most_recent['DEMOGRAPHICS GT'] = 331.57
            sec_names = ['ABSTRACT', 'HISTORY', 'GEOGRAPHY', 'DEMOGRAPHICS']

        for s_name in sec_names:
            if f'GT/[ {s_name} ] length recent' in most_recent.keys():
                most_recent[f'GT/{s_name} length diff'] = (
                    most_recent[f'GT/[ {s_name} ] length recent'] - most_recent[f'{s_name} GT'])
                most_recent[f'GT/{s_name} length abs diff'] = abs(
                    most_recent[f'GT/[ {s_name} ] length recent'] - most_recent[f'{s_name} GT'])
            if f'BRIDGE/[ {s_name} ] length recent' in most_recent.keys():
                most_recent[f'BRIDGE/{s_name} length diff'] = (
                    most_recent[f'BRIDGE/[ {s_name} ] length recent'] - most_recent[f'{s_name} GT'])
                most_recent[f'BRIDGE/{s_name} length abs diff'] = abs(
                    most_recent[f'BRIDGE/[ {s_name} ] length recent'] - most_recent[f'{s_name} GT'])
            if f'RANDOM/[ {s_name} ] length recent' in most_recent.keys():
                most_recent[f'RANDOM/{s_name} length diff'] = (
                    most_recent[f'RANDOM/[ {s_name} ] length recent'] - most_recent[f'{s_name} GT'])
                most_recent[f'RANDOM/{s_name} length abs diff'] = abs(
                    most_recent[f'RANDOM/[ {s_name} ] length recent'] - most_recent[f'{s_name} GT'])

        wandb.log(most_recent)

    def _get_classification(self, raw_seq):
        results = defaultdict(lambda: [])
        self._classification_examples = dict()
        raw_seq = raw_seq.replace("<|endoftext|> ", "")
        split_seq = raw_seq.split(". ")
        sec_id = 0
        seq_idxs = []
        for seq_idx, seq in enumerate(split_seq):
            if not seq:
                continue
            seq_idxs.append(seq_idx)
            seq += "."
            for tok in self.section_names:
                if tok in seq:
                    sec_id = self.section_names.index(tok)
                    seq = seq.replace(tok+" ", "")
                    try:
                        assert tok not in seq
                    except:
                        seq = seq.replace(tok, "")

            tokenized_seq = self.classification_tokenizer(seq, return_tensors='pt').to(
                self.classification_model.device
            )
            result = self.classification_model(input_ids=tokenized_seq['input_ids'][:, :512])
            probs = torch.nn.functional.softmax(result.logits, dim=1)

            acc = int(torch.argmax(probs) == sec_id)
            entropy = -torch.sum(probs * torch.log(probs)).detach().cpu().numpy()
            prob_sec_id = probs[0, sec_id].detach().cpu().numpy()

            # uniform_p = torch.tensor([0.25]*4)
            # y_entropy = -torch.sum(uniform_p * torch.log(uniform_p))
            # mi = float(y_entropy - entropy)

            self.metrics["{} class acc".format(self.section_names[sec_id])].append(acc)
            self.metrics["{} class entropy".format(self.section_names[sec_id])].append(entropy)
            # self.metrics["{} MI".format(self.section_names[sec_id])].append(mi)
            self.metrics["{} p(section_id*|x)".format(self.section_names[sec_id])].append(prob_sec_id)
            results["{} class acc".format(self.section_names[sec_id])].append(acc)
            results["{} class entropy".format(self.section_names[sec_id])].append(entropy)
            # results["{} MI".format(self.section_names[sec_id])].append(mi)
            results["{} p(section_id*|x)".format(self.section_names[sec_id])].append(prob_sec_id)

            # sentences that are induce high/low acc/entropy/mi
            for key, metric in zip(["{} class acc", "{} class entropy"], [acc, entropy,]):
                key = key.format(self.section_names[sec_id])
                if results[key] and max(results[key]) == metric:
                    self._classification_examples[key + " MAX"] = seq
                    self.metrics[key + " MAX IDX"].append(
                        len(results["{} class acc".format(self.section_names[sec_id])]))
                    results[key + " MAX IDX"].append(
                        len(results["{} class acc".format(self.section_names[sec_id])]))
                if results[key] and min(results[key]) == metric:
                    self._classification_examples[key + " MIN"] = seq
                    self.metrics[key + " MIN IDX"].append(
                        len(results["{} class acc".format(self.section_names[sec_id])]))
                    results[key + " MIN IDX"].append(
                        len(results["{} class acc".format(self.section_names[sec_id])]))

        return results

    def print_results(self):
        print("Examples")
        extreme_ex = []
        for k in sorted(self.examples.keys()):
            print("[ {} ] : {}".format(k, self.examples[k]))
            extreme_ex.append({'label': k, 'text': self.examples[k]})

        fname = os.path.join('results', self.fname+"_extreme_examples.csv")
        print(f"saving extreme examples at {fname}")
        with open(fname, "wb") as f:
            df = pd.DataFrame(extreme_ex)
            df.to_csv(f)

        print("Num samples: {}".format(len(self.metrics['ppl'])))
        for k in sorted(self.metrics.keys()):
            v = self.metrics[k]
            print("[ {} ] = {} +- {}".format(k, np.mean(v), np.std(v)))

        fname = os.path.join('results', self.fname+"_metrics.csv")
        print(f"saving metrics at {fname}")
        with open(fname, "wb") as f:
            df = pd.DataFrame(self._info)
            df.to_csv(f)

        fname = os.path.join('results', self.fname+"_examples.csv")
        with open(fname, "wb") as f:
            df = pd.DataFrame(self._examples)
            df.to_csv(f)
