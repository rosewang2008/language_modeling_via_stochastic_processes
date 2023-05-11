import os
import sys
import json
import torch
import random
import string
import pickle
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
from sklearn.preprocessing import scale
from transformers import GPT2Tokenizer, BertTokenizer, AlbertTokenizer
import re
sys.path.append("/nlp/scr/rewang/ilm")
import ilm

class WikiSectionData(data.Dataset):
    """WikiSection data"""


    def __init__(
            self, filepath, train, data_dim, n_segments, n_obs_seg,
            tokenizer_name, unit,
            seed, one_hot_labels):
        """
        Args:
            filepath: str; filepath to json data file
            train: bool; whether to train
            n_obs_seg: int; top k sentences to take from each section
        """
        super().__init__()
        self.filepath = filepath
        self.train = train
        self.data_dim = data_dim # num docs used
        self.n_segments = n_segments # num sections
        self.n_obs_seg = n_obs_seg # num sentences in a section
        self.seed = seed
        self.one_hot_labels = one_hot_labels
        self.tokenizer_name = tokenizer_name
        self.unit = unit

        with open(self.filepath, 'rb') as f:
            self.data = json.load(f)

        self.section_names = ['abstract', 'History', 'Geography', 'Demographics']
        assert len(self.section_names) == self.n_segments

        self.split_pattern = '. '
        self._set_tokenizer()

    def _set_tokenizer(self):
        if self.tokenizer_name == "GPT2":
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.end_token = self.tokenizer.eos_token_id
        elif self.tokenizer_name == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        else:
            raise ValueError("Dont recognize name {}".format(self.tokenizer_name))

    def get_section(self, doc_idx, section_idx):
        doc = self.data[doc_idx]
        if section_idx == 0:
            try:
                text = doc['abstract']
            except:
                text = doc['text']
                # text = doc['text'].split(self.split_pattern)[0]
        else:
            info = list(filter(
                lambda x: x['sectionHeading']==self.section_names[section_idx],
                doc['annotations']))[0]
            text = doc['text'][info['begin']:info['begin']+info['length']]

        # Whether to randomly sample
        truncated = self._get_text_sample(text)
        return truncated

    def _get_text_sample(self, text):
        truncated = text.replace(".\n", ". ").split(self.split_pattern)[:-1]
        if len(truncated) == 0:
            # NOTE: bug noted in log 04/25/2021; see pic for context
            truncated = [text]
        if self.unit == "sentence": # randomly sample a sentence
            truncated = truncated[np.random.randint(len(truncated))]
            # Add a period
            truncated += "."
        elif self.unit == "section": # take first k sentneces
            truncated = truncated[:self.n_obs_seg]
            truncated = self.split_pattern.join(truncated)
        return truncated

    def tokenize_caption(self, caption, device):
        if self.tokenizer_name == "GPT2":
            output = self.tokenizer(
                caption,
                padding=True,
                return_tensors='pt',
            )
            input_ids = output['input_ids'].squeeze(0)
            attention_mask = output['attention_mask'].squeeze(0)
            eos_input_ids = torch.tensor([[self.end_token]*input_ids.shape[0]])
            eos_attention = torch.tensor([[0]*input_ids.shape[0]])
            input_ids = torch.cat((input_ids, eos_input_ids.T), dim=1)
            attention_mask = torch.cat((attention_mask, eos_attention.T), dim=1)
        elif self.tokenizer_name == "BERT":
            # Prepend [CLS] so I can use the first embedding
            output = self.tokenizer(
                caption,
                padding=True,
                return_tensors='pt',
            )
            input_ids = output['input_ids'].squeeze(0)
            attention_mask = output['attention_mask'].squeeze(0)
        return input_ids.to(device), attention_mask.to(device)

    def __getitem__(self, index):
        doc_idx = index // self.n_segments # data_dim*doc_idx:data_dim*(doc_idx+1)
        section_idx = index % self.n_segments
        section_sentence = self.get_section(doc_idx=doc_idx, section_idx=section_idx)

        result = {
            'observation': section_sentence,
            'label': np.array(section_idx)
        }
        return result

    def __len__(self):
        return int(len(self.data) * self.n_segments)


class WikiOUData(data.Dataset):

    def __init__(
            self, filepath, train, data_dim, n_segments, n_obs_seg,
            tokenizer_name, unit,
            seed, one_hot_labels, config):
        super().__init__()
        self.filepath = filepath
        self.train = train
        self.data_dim = data_dim # num docs used
        self.n_segments = n_segments # num sections
        self.n_obs_seg = n_obs_seg # num sentences in a section
        self.seed = seed
        self.one_hot_labels = one_hot_labels
        self.tokenizer_name = tokenizer_name
        self.unit = unit
        self.config=config

        with open(self.filepath, 'rb') as f:
            self.data = json.load(f)

        self._set_section_names()

        self.split_pattern = '. '
        self._set_tokenizer()
        self._process_data()
        # print examples
        print("Examples: {}".format(self.processed_data[0]))
        print("Examples: {}".format(self.processed_data[10]))

        print("Exmaples: {}".format(self.processed_data[0]))
        print("Exmaples: {}".format(self.processed_data[10]))

    def _set_section_names(self):
        self.section_names = ['abstract', 'History', 'Geography', 'Demographics']
        self.section_ids = ['[ {} ]'.format(name.upper()) for name in self.section_names]

    def _process_data(self):
        self.processed_data = []
        # Cut down data
        # for doc_id in range(len(self.data)):
        for doc_id in range(int(len(self.data)/2)):
            doc_info = []
            sentence_counter = 0
            for section_id, section_name in enumerate(self.section_names):
                if section_id == 0: # abstract
                    try:
                        text = self.data[doc_id][section_name]
                    except:
                        break
                else:
                    doc = self.data[doc_id]
                    info = list(filter(
                        lambda x: x['sectionHeading']==section_name, doc['annotations']))[0]
                    text = doc['text'][info['begin']:info['begin']+info['length']]
                truncated = text.replace(".\n", ". ").split(self.split_pattern)[:-1]
                if len(truncated) == 0:
                    # NOTE: bug noted in log 04/25/2021; see pic for context
                    truncated = [text]

                for sentence_i, sentence in enumerate(truncated):
                    if not sentence:
                        continue

                    if sentence_i == 0 and self.config.use_section_ids:
                        # adding " . " EOS
                        sentence = "{} {} . ".format(self.section_ids[section_id], sentence)
                    else:
                        sentence += " . "

                    input_ids, attention_mask = self.get_tokenized(sentence)

                    sentence_info = {
                        "sentence": sentence,
                        "sentence_id": sentence_counter,
                        "section": section_name,
                        "section_id": section_id,
                        "doc_id": doc_id,
                        # "input_ids": input_ids,
                        # "attention_mask": attention_mask,
                    }
                    doc_info.append(sentence_info)
                    sentence_counter += 1

            # Track total number of sentences in a document
            for info in doc_info:
                info['total_doc_sentences'] = sentence_counter

            self.processed_data += doc_info

        # print examples
        print("Examples: {}".format(self.processed_data[0]))
        print("Examples: {}".format(self.processed_data[10]))

    def get_tokenized(self, sentence):
        tokenized = self.tokenizer(sentence, truncation=True, max_length=self.max_length)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        input_ids += [self.tokenizer.eos_token_id] * (self.max_length - len(input_ids))
        attention_mask += [0] * (self.max_length - len(attention_mask))

        return input_ids, attention_mask

    def _set_tokenizer(self):
        if self.tokenizer_name == "GPT2":
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.end_token = self.tokenizer.eos_token_id
            self.max_length = 1024
        elif self.tokenizer_name == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.max_length = 512
        elif self.tokenizer_name == "ALBERT":
            # self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.max_length = 512
        else:
            raise ValueError("Dont recognize name {}".format(self.tokenizer_name))
        # Add section ids
        self.cl_eos_str = " . "
        self.tokenizer.add_tokens(self.section_ids + [self.cl_eos_str])
        try:
            self.special_tokens = [_[0] for _ in self.tokenizer(self.section_ids)['input_ids']]
        except:
            print("no section ids: {}".format(self.section_ids))
        self.cl_eos_id = self.tokenizer(self.cl_eos_str)['input_ids'][0]

    def tokenize_caption(self, caption, device):
        if self.tokenizer_name == "GPT2":
            output = self.tokenizer(
                caption,
                padding=True,
                return_tensors='pt',
            )
            input_ids = output['input_ids'].squeeze(0)
            attention_mask = output['attention_mask'].squeeze(0)
            eos_input_ids = torch.tensor([[self.end_token]*input_ids.shape[0]])
            eos_attention = torch.tensor([[0]*input_ids.shape[0]])
            input_ids = torch.cat((input_ids, eos_input_ids.T), dim=1)
            attention_mask = torch.cat((attention_mask, eos_attention.T), dim=1)
        elif self.tokenizer_name == "BERT" or self.tokenizer_name == "ALBERT":
            # Prepend [CLS] so I can use the first embedding
            output = self.tokenizer(
                caption,
                padding=True,
                return_tensors='pt',
            )
            input_ids = output['input_ids'].squeeze(0)
            attention_mask = output['attention_mask'].squeeze(0)
        return input_ids.to(device), attention_mask.to(device)

    def __getitem__(self, index):
        if self.config.k == 1:
            if self.processed_data[index]['doc_id'] != self.processed_data[index+1]['doc_id']:
                index -= 1

            y_t = self.processed_data[index]['sentence']
            y_tp1 = self.processed_data[index+1]['sentence']
            t = self.processed_data[index]['sentence_id']/self.processed_data[index]['total_doc_sentences']
            dt = 1./self.processed_data[index]['total_doc_sentences']
        else:
            assert self.config.k > 1
            # k sampling
            utterance = self.processed_data[index]
            tp1 = min(utterance['total_doc_sentences']-1,
                      utterance['sentence_id']+self.config.k)
            t = max(0, tp1-self.config.k)

            dt = (tp1 - t)/utterance['total_doc_sentences']
            y_t = self.processed_data[index + (t - utterance['sentence_id'])]['sentence']
            y_tp1 = self.processed_data[index + (tp1 - utterance['sentence_id'])]['sentence']
            t = self.processed_data[index + (t - utterance['sentence_id'])]['sentence_id']/utterance['total_doc_sentences']


        result = {
            'y_t': y_t,
            # 'input_ids_t': self.processed_data[index]['input_ids'],
            # 'attention_mask_t': self.processed_data[index]['attention_mask'],
            'y_tp1': y_tp1,
            # 'input_ids_tp1': self.processed_data[index+1]['input_ids'],
            # 'attention_mask_tp1': self.processed_data[index+1]['attention_mask'],
            't': t,
            'dt': dt
        }
        return result

    def __len__(self):
        return len(self.processed_data) - 1

class WikiTPKData(WikiOUData):
    # t, t + k
    # and t-1, t-2, t-3

    def __init__(
            self, filepath, train, data_dim, n_segments, n_obs_seg,
            tokenizer_name, unit,
            seed, one_hot_labels, config):

        super().__init__(
            filepath=filepath,
            train=train,
            data_dim=data_dim,
            n_segments=n_segments,
            n_obs_seg=n_obs_seg,
            tokenizer_name=tokenizer_name,
            unit=unit,
            seed=seed,
            one_hot_labels=one_hot_labels,
            config=config
        )

    def __getitem__(self, index):
        if self.config.k == 1:
            if self.processed_data[index]['doc_id'] != self.processed_data[index+1]['doc_id']:
                index -= 1

            y_t = self.processed_data[index]['sentence']
            y_tp1 = self.processed_data[index+1]['sentence']
            t = self.processed_data[index]['sentence_id']/self.processed_data[index]['total_doc_sentences']
            dt = 1./self.processed_data[index]['total_doc_sentences']
        else:
            assert self.config.k > 1
            # k sampling
            utterance = self.processed_data[index]
            tp1 = min(utterance['total_doc_sentences']-1,
                      utterance['sentence_id']+self.config.k)
            t = max(0, tp1-self.config.k)

            dt = (tp1 - t)/utterance['total_doc_sentences']
            y_t = self.processed_data[index + (t - utterance['sentence_id'])]['sentence']
            y_tp1 = self.processed_data[index + (tp1 - utterance['sentence_id'])]['sentence']
            t = self.processed_data[index + (t - utterance['sentence_id'])]['sentence_id']/utterance['total_doc_sentences']

        y_tm1 = (self.processed_data[index] if (index - 1 < 0 or self.processed_data[index]['doc_id'] != self.processed_data[index-1]['doc_id']) else self.processed_data[index-1])
        y_tm1 = y_tm1['sentence']
        y_tm2 = (self.processed_data[index] if (index - 2 < 0 or self.processed_data[index]['doc_id'] != self.processed_data[index-2]['doc_id']) else self.processed_data[index-2])
        y_tm2 = y_tm2['sentence']
        y_tm3 = (self.processed_data[index] if (index - 3 < 0 or self.processed_data[index]['doc_id'] != self.processed_data[index-3]['doc_id']) else self.processed_data[index-3])
        y_tm3 = y_tm3['sentence']


        result = {
            'y_t': y_t,
            'y_tm1': y_tm1,
            'y_tm2': y_tm2,
            'y_tm3': y_tm3,
            'y_tpk': y_tp1,
        }
        return result

class WikiRandomTData(WikiOUData):

    def __init__(
            self, filepath, train, data_dim, n_segments, n_obs_seg,
            tokenizer_name, unit,
            seed, one_hot_labels, config):

        super().__init__(
            filepath=filepath,
            train=train,
            data_dim=data_dim,
            n_segments=n_segments,
            n_obs_seg=n_obs_seg,
            tokenizer_name=tokenizer_name,
            unit=unit,
            seed=seed,
            one_hot_labels=one_hot_labels,
            config=config
        )

    def __getitem__(self, index):
        utterance = self.processed_data[index]
        sentence_num = utterance['sentence_id']

        # Check if index is start of a seq. If so -> +2
        if sentence_num == 0:
            index += 2
        if sentence_num == 1:
            index += 1

        # Update
        utterance = self.processed_data[index]
        sentence_num = utterance['sentence_id']

        # # TRIAL 1:"""Sample 0, t , t'"""
        # T = sentence_num
        # # t is a random point in between
        # t = np.random.randint(1, T)
        # t_ = 0
        # y_0 = self.processed_data[index - T]['sentence']
        # y_t = self.processed_data[index - T + t]['sentence']
        # y_T = self.processed_data[index]['sentence']


        # TRIAL 2: Sample all random points, t, t', t''
        T = sentence_num
        # t is a random point in between
        nums = list(range(T))
        t1 = random.choice(nums)
        nums.remove(t1)
        t2 = random.choice(nums)
        if t2 < t1:
            t = t2
            t2 = t1
            t1 = t

        assert t1 < t2 and t2 < T
        y_0 = self.processed_data[index - T + t1]['sentence']
        y_t = self.processed_data[index - T + t2]['sentence']
        y_T = self.processed_data[index]['sentence']

        t_ = t1
        t = t2

        total_doc = utterance['total_doc_sentences']
        result = {
            'y_0': y_0,
            'y_t': y_t,
            'y_T': y_T,
            't_': t_,
            't': t,
            'T': T,
            'total_t': total_doc,
        }
        return result

    def __len__(self):
        return len(self.processed_data) - 1

class Taskmaster(WikiOUData):

    def __init__(
            self, filepath, train, data_dim, n_segments, n_obs_seg,
            tokenizer_name, unit,
            seed, one_hot_labels, config):
        super().__init__(
            filepath, train, data_dim, n_segments, n_obs_seg,
            tokenizer_name, unit,
            seed, one_hot_labels, config
        )

    def _set_section_names(self):
        # For movies: https://github.com/google-research-datasets/Taskmaster/tree/master/TM-3-2020
        self.section_names = ['user', 'assistant']
        self.section_ids = ['[ {} ]'.format(name.upper()) for name in self.section_names]

        print("tm type: ", self.config.data_params.tm_type)
        if 'movie' in self.config.data_params.tm_type:
            self.data_dir = "/nlp/scr/rewang/data/Taskmaster/TM-3-2020/data"
            if self.train:
                # self.data_files = ['data_0{}.json'.format(i) for i in range(0, 8)] + [
                #     # 'data_{}.json'.format(i) for i in range(10,13)
                # ]
                self.data_files = ['data_0{}.json'.format(i) for i in range(0, 3)]
            else:
                self.data_files = ['data_{}.json'.format(i) for i in range(13, 14)]
        else: # should be restaurant
            self.data_dir = "/nlp/scr/rewang/data/Taskmaster/TM-2-2020/data"
            if self.train:
                self.data_files = ['restaurant-search.json'] # 3276 conversations
                self.start_conversation = 0
                self.end_conversation = 2000
            else:
                self.data_files = ['restaurant-search.json']
                self.start_conversation = 2000
                self.end_conversation = 3276

    def _process_data(self):
        self.processed_data = []
        doc_id = 0
        min_length = np.inf
        for fname in self.data_files:
            data = json.load(open(os.path.join(self.data_dir, fname), 'rb'))
            if 'movie' not in self.config.data_params.tm_type:
            # if self.config.data_params.tm_type != "movie":
                data = data[self.start_conversation:self.end_conversation]
            print('loading from', fname)
            print('num conversations loading', len(data))
            for conversation in data:
                for sentence_counter, utterance in enumerate(conversation['utterances']):
                    text = "[ {} ] {}".format(utterance['speaker'].upper(), utterance['text'])
                    sentence_info = {
                        "sentence": text,
                        "sentence_id": sentence_counter,
                        "doc_id": doc_id,
                        "total_doc_sentences": len(conversation['utterances'])
                    }
                    self.processed_data.append(sentence_info)
                    min_length = min(min_length, len(conversation['utterances']))
                doc_id += 1
                if len(self.processed_data) > 25000:
                    break
            if len(self.processed_data) > 25000:
                break
        print(f'MIN LENGTH DISCOURSE: {min_length}')
        print('length of dataset: {}'.format(len(self.processed_data)))
        print(f"last doc id {doc_id}")

class TaskmasterDiscourse(Taskmaster):

    def __init__(
            self,
            filepath,
            train,
            config,
            all_dataset=None,
            data_dim=0,
            n_segments=0,
            n_obs_seg=0,
            tokenizer_name='GPT2',
            unit='sentence',
            seed=1,
            one_hot_labels=False,
    ):
        super().__init__(
            filepath=filepath,
            train=train,
            config=config,
            data_dim=data_dim,
            n_segments=n_segments,
            n_obs_seg=n_obs_seg,
            tokenizer_name=tokenizer_name,
            unit=unit,
            seed=seed,
            one_hot_labels=one_hot_labels
        )
        self.k = self.config.k

    def __getitem__(self, index):
        label = random.randint(0, 1) # either in- or out-of-order

        # # Setup 1: sample t+1 utterance.
        # if label: # in-order
        #     if self.processed_data[index]['doc_id'] != self.processed_data[index+1]['doc_id']:
        #         index -= 1
        #     y_t = self.processed_data[index]['sentence']
        #     y_tp1 = self.processed_data[index+1]['sentence']
        # else:
        #     y_t = self.processed_data[index]['sentence']
        #     random_idx = random.randint(0, len(self.processed_data)-1) # either in- or out-of-order
        #     y_tp1 = self.processed_data[random_idx]['sentence']

        # Setup 2: sample t+k utterance
        utterance = self.processed_data[index]
        tp1 = min(utterance['total_doc_sentences']-1, utterance['sentence_id']+self.k)
        t = max(0, tp1-self.k)

        y_t = self.processed_data[index + (t - utterance['sentence_id'])]
        y_tp1 = self.processed_data[index + (tp1 - utterance['sentence_id'])]

        assert y_t['doc_id'] == y_tp1['doc_id']

        y_t = y_t['sentence']
        y_tp1 = y_tp1['sentence']

        if label: # in order
            pass # do nothing
        else:
            tmp = y_tp1
            y_tp1 = y_t
            y_t = tmp

        if self.one_hot_labels:
            labels = torch.zeros(2)
            labels[label] = 1.0
            label = labels

        result = {
            'y_t': y_t,
            'y_tp1': y_tp1,
            't': t,
            'tp1': tp1,
            'label': label,
            'idx': index
        }
        return result

class TaskmasterTPKData(TaskmasterDiscourse):
    # t, t + k
    # and t-1, t-2, t-3

    def __init__(
            self,
            filepath,
            train,
            config,
            all_dataset=None,
            data_dim=0,
            n_segments=0,
            n_obs_seg=0,
            tokenizer_name='GPT2',
            unit='sentence',
            seed=1,
            one_hot_labels=False,
    ):
        super().__init__(
            filepath=filepath,
            train=train,
            config=config,
            data_dim=data_dim,
            n_segments=n_segments,
            n_obs_seg=n_obs_seg,
            tokenizer_name=tokenizer_name,
            unit=unit,
            seed=seed,
            one_hot_labels=one_hot_labels
        )

    def __getitem__(self, index):
        if self.config.k == 1:
            if self.processed_data[index]['doc_id'] != self.processed_data[index+1]['doc_id']:
                index -= 1

            y_t = self.processed_data[index]['sentence']
            y_tp1 = self.processed_data[index+1]['sentence']
            t = self.processed_data[index]['sentence_id']/self.processed_data[index]['total_doc_sentences']
            dt = 1./self.processed_data[index]['total_doc_sentences']
        else:
            assert self.config.k > 1
            # k sampling
            utterance = self.processed_data[index]
            tp1 = min(utterance['total_doc_sentences']-1,
                      utterance['sentence_id']+self.config.k)
            t = max(0, tp1-self.config.k)

            dt = (tp1 - t)/utterance['total_doc_sentences']
            y_t = self.processed_data[index + (t - utterance['sentence_id'])]['sentence']
            y_tp1 = self.processed_data[index + (tp1 - utterance['sentence_id'])]['sentence']
            t = self.processed_data[index + (t - utterance['sentence_id'])]['sentence_id']/utterance['total_doc_sentences']

        y_tm1 = (self.processed_data[index] if (index - 1 < 0 or self.processed_data[index]['doc_id'] != self.processed_data[index-1]['doc_id']) else self.processed_data[index-1])
        y_tm1 = y_tm1['sentence']
        y_tm2 = (self.processed_data[index] if (index - 2 < 0 or self.processed_data[index]['doc_id'] != self.processed_data[index-2]['doc_id']) else self.processed_data[index-2])
        y_tm2 = y_tm2['sentence']
        y_tm3 = (self.processed_data[index] if (index - 3 < 0 or self.processed_data[index]['doc_id'] != self.processed_data[index-3]['doc_id']) else self.processed_data[index-3])
        y_tm3 = y_tm3['sentence']


        result = {
            'y_t': y_t,
            'y_tm1': y_tm1,
            'y_tm2': y_tm2,
            'y_tm3': y_tm3,
            'y_tpk': y_tp1,
        }
        return result

class TaskmasterRandomT(Taskmaster):

    def __init__(
            self,
            filepath,
            train,
            config,
            all_dataset=None,
            data_dim=0,
            n_segments=0,
            n_obs_seg=0,
            tokenizer_name='GPT2',
            unit='sentence',
            seed=1,
            one_hot_labels=False,
    ):
        super().__init__(
            filepath=filepath,
            train=train,
            config=config,
            data_dim=data_dim,
            n_segments=n_segments,
            n_obs_seg=n_obs_seg,
            tokenizer_name=tokenizer_name,
            unit=unit,
            seed=seed,
            one_hot_labels=one_hot_labels
        )
        self.k = self.config.k

    def __getitem__(self, index):
        utterance = self.processed_data[index]
        sentence_num = utterance['sentence_id']

        # Check if index is start of a seq. If so -> +2
        if sentence_num == 0:
            index += 2
        if sentence_num == 1:
            index += 1

        # Update
        utterance = self.processed_data[index]
        sentence_num = utterance['sentence_id']

        # TRIAL 2: Sample all random points, t, t', t''
        T = sentence_num
        # t is a random point in between
        nums = list(range(T))
        t1 = random.choice(nums)
        nums.remove(t1)
        t2 = random.choice(nums)
        if t2 < t1:
            t = t2
            t2 = t1
            t1 = t

        assert t1 < t2 and t2 < T
        y_0 = self.processed_data[index - T + t1]['sentence']
        y_t = self.processed_data[index - T + t2]['sentence']
        y_T = self.processed_data[index]['sentence']

        t_ = t1
        t = t2

        total_doc = utterance['total_doc_sentences']
        result = {
            'y_0': y_0,
            'y_t': y_t,
            'y_T': y_T,
            't_': t_,
            't': t,
            'T': T,
            'total_t': total_doc,
        }
        return result

class WikisectionClassification(WikiOUData):

    def __init__(
            self,
            filepath,
            train,
            config,
            all_dataset=None,
            data_dim=0,
            n_segments=0,
            n_obs_seg=0,
            tokenizer_name='GPT2',
            unit='sentence',
            seed=1,
            one_hot_labels=False,
    ):
        super(WikisectionClassification, self).__init__(
            filepath=filepath,
            train=train,
            config=config,
            data_dim=data_dim,
            n_segments=n_segments,
            n_obs_seg=n_obs_seg,
            tokenizer_name=tokenizer_name,
            unit=unit,
            seed=seed,
            one_hot_labels=one_hot_labels
        )
        self.num_labels = len(self.section_names)

    def __getitem__(self, index):
        y_t = self.processed_data[index]['sentence']
        label = self.processed_data[index]['section_id']

        if self.one_hot_labels:
            labels = torch.zeros(self.num_labels)
            labels[label] = 1.0
            label = labels

        result = {
            'y_t': y_t,
            'label': label,
        }
        return result

class WikisectionEOS(WikisectionClassification):
    """
    Classification of whether the sentence is the last sentence
    """

    def __init__(
            self,
            filepath,
            train,
            config,
            all_dataset=None,
            data_dim=0,
            n_segments=0,
            n_obs_seg=0,
            tokenizer_name='GPT2',
            unit='sentence',
            seed=1,
            one_hot_labels=False,
    ):
        super(WikisectionEOS, self).__init__(
            filepath=filepath,
            train=train,
            config=config,
            data_dim=data_dim,
            n_segments=n_segments,
            n_obs_seg=n_obs_seg,
            tokenizer_name=tokenizer_name,
            unit=unit,
            seed=seed,
            one_hot_labels=one_hot_labels
        )
        self.num_labels = 2

    def __getitem__(self, index):
        label = random.randint(0, 1) # either in- or out-of-order

        # Find the first sentence that matches the label condition
        while int(self.processed_data[index]['section_id']
                  != self.processed_data[index+1]['section_id']) != label:
            if index + 1 == len(self.processed_data)-1:
                break
            index += 1

        y_t = self.processed_data[index]['sentence']

        if self.one_hot_labels:
            labels = torch.zeros(self.num_labels)
            labels[label] = 1.0
            label = labels

        result = {
            'y_t': y_t,
            'label': label,
        }
        return result

    def __len__(self):
        return len(self.processed_data) - 2


class WikisectionDiscourse(WikiOUData):

    def __init__(
            self,
            filepath,
            train,
            config,
            all_dataset=None,
            data_dim=0,
            n_segments=0,
            n_obs_seg=0,
            tokenizer_name='GPT2',
            unit='sentence',
            seed=1,
            one_hot_labels=False,
    ):
        super(WikisectionDiscourse, self).__init__(
            filepath=filepath,
            train=train,
            config=config,
            data_dim=data_dim,
            n_segments=n_segments,
            n_obs_seg=n_obs_seg,
            tokenizer_name=tokenizer_name,
            unit=unit,
            seed=seed,
            one_hot_labels=one_hot_labels
        )

    def _find_first_sentence_section(self, target_section, index, direction):
        sentence = self.processed_data[index]['sentence']
        while target_section not in sentence:
            if direction == 'forward':
                index += 1
            else:
                index -= 1
            sentence = self.processed_data[index]['sentence']
        return sentence, index

    def __getitem__(self, index):
        label = random.randint(0, 1) # either in- or out-of-order

        # SETUP 1 & 2:
        # 1: sample t and t+1
        # 2: sample first sentence of section i and first sentence of section i+1
        # if label: # in-order
        #     # if self.processed_data[index]['doc_id'] != self.processed_data[index+1]['doc_id']:
        #     #     index -= 1
        #     # y_t = self.processed_data[index]['sentence']
        #     # y_tp1 = self.processed_data[index+1]['sentence']

        #     ### DEBUGGING ###
        #     y_t = self.processed_data[index]
        #     if y_t['section_id'] == len(self.section_ids) - 1: # last section id
        #         target_section_t = self.section_ids[-2]
        #         target_section_tp1 = self.section_ids[-1]
        #         y_t = self._find_first_sentence_section(target_section_t, index, direction='backward')
        #         y_tp1 = self._find_first_sentence_section(target_section_tp1, index, direction='backward')
        #     else:
        #         target_section_t = self.section_ids[y_t['section_id']]
        #         target_section_tp1 = self.section_ids[y_t['section_id'] + 1]
        #         y_t = self._find_first_sentence_section(target_section_t, index, direction='backward')
        #         y_tp1 = self._find_first_sentence_section(target_section_tp1, index, direction='forward')

        # else:
        #     y_t = self.processed_data[index]['sentence']
        #     random_idx = random.randint(0, len(self.processed_data)-1) # either in- or out-of-order
        #     y_tp1 = self.processed_data[random_idx]['sentence']

        # # SETUP 3: Sample first sentence of section i and i+1. if +, do in order. otherwise, flip order.
        # y_t = self.processed_data[index]
        # if y_t['section_id'] == len(self.section_ids) - 1: # last section id
        #     target_section_t = self.section_ids[-2]
        #     target_section_tp1 = self.section_ids[-1]
        #     y_t, t = self._find_first_sentence_section(target_section_t, index, direction='backward')
        #     y_tp1, tp1 = self._find_first_sentence_section(target_section_tp1, index, direction='backward')
        # else:
        #     target_section_t = self.section_ids[y_t['section_id']]
        #     target_section_tp1 = self.section_ids[y_t['section_id'] + 1]
        #     y_t, t = self._find_first_sentence_section(target_section_t, index, direction='backward')
        #     y_tp1, tp1 = self._find_first_sentence_section(target_section_tp1, index, direction='forward')

        # SETUP 4: sample t+k utterance
        utterance = self.processed_data[index]
        tp1 = min(utterance['total_doc_sentences']-1, utterance['sentence_id']+self.config.k)
        t = max(0, tp1-self.config.k)

        y_t = self.processed_data[index + (t - utterance['sentence_id'])]
        y_tp1 = self.processed_data[index + (tp1 - utterance['sentence_id'])]

        assert y_t['doc_id'] == y_tp1['doc_id']

        y_t = y_t['sentence']
        y_tp1 = y_tp1['sentence']

        if label:
            pass # do nothing
        else:
            tmp = y_tp1
            y_tp1 = y_t
            y_t = tmp

        if self.one_hot_labels:
            labels = torch.zeros(2)
            labels[label] = 1.0
            label = labels

        result = {
            'y_t': y_t,
            'y_tp1': y_tp1,
            'label': label,
            'idx': index,
            't': index + (t - utterance['sentence_id']),
            'tp1': index + (tp1 - utterance['sentence_id']),
        }
        return result


class LongerWikiOUData(WikiOUData):
    """
    with more sections:

    required = ['History',
            'Geography',
            'Geography | Climate',
            'Climate',
            'Demographics',
            'Economy',
            'Transportation',
            'Government',
            'Sports'
           ]
    """

    def __init__(
            self, filepath, train, data_dim, n_segments, n_obs_seg,
            tokenizer_name, unit,
            seed, one_hot_labels, config):
        super().__init__(
            filepath=filepath,
            train=train,
            data_dim=data_dim,
            n_segments=n_segments,
            n_obs_seg=n_obs_seg,
            tokenizer_name=tokenizer_name,
            unit=unit,
            seed=seed,
            config=config,
            one_hot_labels=one_hot_labels
        )

    def _set_section_names(self):
        self.section_names = [  'abstract',
                                'History',
                                'Geography',
                                'Geography | Climate',
                                'Climate',
                                'Demographics',
                                'Economy',
                                'Transportation',
                                'Government',
                                'Sports'
                               ]
        self.section_ids = ['[ {} ]'.format(name.upper()) for name in self.section_names]

    def _process_data(self):
        self.processed_data = []
        # Cut down data
        # for doc_id in range(len(self.data)):
        # Original data docs: 12593
        # num processed datapoint: 310515
        for doc_id in range(int(len(self.data)/12)):
            doc_info = []
            sentence_counter = 0
            for section_id, section_name in enumerate(self.section_names):
                doc = self.data[doc_id]
                # (a) Check if doc has an abstract
                if section_name == 'abstract':
                    if section_name in doc.keys():
                        text = doc[section_name]
                    else: # Don't include docs without abstract
                        break
                # (a) All other sections
                else:
                    # (b) Check if doc has this section
                    info = list(filter(
                        lambda x: x['sectionHeading']==section_name, doc['annotations']))
                    # Doc contains this section
                    if info:
                        info = info[0]
                        text = doc['text'][info['begin']:info['begin']+info['length']]
                    else:
                        continue # Doc doesn't contain section

                truncated = text.replace(".\n", ". ").split(self.split_pattern)[:-1]
                if len(truncated) == 0:
                    # NOTE: bug noted in log 04/25/2021; see pic for context
                    truncated = [text]

                for sentence_i,  sentence in enumerate(truncated):
                    if not sentence:
                        continue

                    if sentence_i == 0 and self.config.use_section_ids:
                        # adding " . " EOS
                        sentence = "{} {} . ".format(self.section_ids[section_id], sentence)
                    else:
                        sentence += " . "

                    sentence_info = {
                        "sentence": sentence,
                        "sentence_id": sentence_counter,
                        "section": section_name,
                        "section_id": section_id,
                        "doc_id": doc_id
                    }
                    doc_info.append(sentence_info)
                    sentence_counter += 1

            # Track total number of sentences in a document
            for info in doc_info:
                info['total_doc_sentences'] = sentence_counter

            self.processed_data += doc_info

        # print examples
        print("Exmaples: {}".format(self.processed_data[0]))
        print("Exmaples: {}".format(self.processed_data[10]))


class LongWikisectionClassification(LongerWikiOUData):

    def __init__(
            self,
            filepath,
            train,
            config,
            all_dataset=None,
            data_dim=0,
            n_segments=0,
            n_obs_seg=0,
            tokenizer_name='GPT2',
            unit='sentence',
            seed=1,
            one_hot_labels=False,
    ):
        super(LongWikisectionClassification, self).__init__(
            filepath=filepath,
            train=train,
            config=config,
            data_dim=data_dim,
            n_segments=n_segments,
            n_obs_seg=n_obs_seg,
            tokenizer_name=tokenizer_name,
            unit=unit,
            seed=seed,
            one_hot_labels=one_hot_labels
        )
        self.num_labels = len(self.section_names)

    def __getitem__(self, index):
        label = random.randint(0, 1) # either in- or out-of-order

        # Find the first sentence that matches the label condition
        while int(self.processed_data[index]['section_id']
                  != self.processed_data[index+1]['section_id']) != label:
            if index + 1 == len(self.processed_data)-1:
                break
            index += 1

        y_t = self.processed_data[index]['sentence']

        if self.one_hot_labels:
            labels = torch.zeros(self.num_labels)
            labels[label] = 1.0
            label = labels

        result = {
            'y_t': y_t,
            'label': label,
        }
        return result

    def __len__(self):
        return len(self.processed_data) - 2

class LongWikisectionEOS(LongWikisectionClassification):

    def __init__(
            self,
            filepath,
            train,
            config,
            all_dataset=None,
            data_dim=0,
            n_segments=0,
            n_obs_seg=0,
            tokenizer_name='GPT2',
            unit='sentence',
            seed=1,
            one_hot_labels=False,
    ):
        super(LongWikisectionEOS, self).__init__(
            filepath=filepath,
            train=train,
            config=config,
            data_dim=data_dim,
            n_segments=n_segments,
            n_obs_seg=n_obs_seg,
            tokenizer_name=tokenizer_name,
            unit=unit,
            seed=seed,
            one_hot_labels=one_hot_labels
        )

        self.num_labels = 2

    def __getitem__(self, index):
        label = random.randint(0, 1) # either in- or out-of-order
        # Find the first sentence that matches the label condition
        while int(self.processed_data[index]['section_id']
                  != self.processed_data[index+1]['section_id']) != label:
            if index + 1 == len(self.processed_data)-1:
                break
            index += 1
        y_t = self.processed_data[index]['sentence']
        if self.one_hot_labels:
            labels = torch.zeros(self.num_labels)
            labels[label] = 1.0
            label = labels
        result = {
            'y_t': y_t,
            'label': label,
        }
        return result

    def __len__(self):
        return len(self.processed_data) - 2

class LongWikisectionDiscourse(LongerWikiOUData):

    def __init__(
            self,
            filepath,
            train,
            config,
            all_dataset=None,
            data_dim=0,
            n_segments=0,
            n_obs_seg=0,
            tokenizer_name='GPT2',
            unit='sentence',
            seed=1,
            one_hot_labels=False,
    ):
        super(LongWikisectionDiscourse, self).__init__(
            filepath=filepath,
            train=train,
            config=config,
            data_dim=data_dim,
            n_segments=n_segments,
            n_obs_seg=n_obs_seg,
            tokenizer_name=tokenizer_name,
            unit=unit,
            seed=seed,
            one_hot_labels=one_hot_labels
        )

    def __getitem__(self, index):
        label = random.randint(0, 1) # either in- or out-of-order

        # SETUP 4: sample t+k utterance
        utterance = self.processed_data[index]
        tp1 = min(utterance['total_doc_sentences']-1, utterance['sentence_id']+self.config.k)
        t = max(0, tp1-self.config.k)

        y_t = self.processed_data[index + (t - utterance['sentence_id'])]
        y_tp1 = self.processed_data[index + (tp1 - utterance['sentence_id'])]

        assert y_t['doc_id'] == y_tp1['doc_id']

        y_t = y_t['sentence']
        y_tp1 = y_tp1['sentence']

        if label:
            pass # do nothing
        else:
            tmp = y_tp1
            y_tp1 = y_t
            y_t = tmp

        if self.one_hot_labels:
            labels = torch.zeros(2)
            labels[label] = 1.0
            label = labels

        result = {
            'y_t': y_t,
            'y_tp1': y_tp1,
            'label': label,
            'idx': index,
            't': index + (t - utterance['sentence_id']),
            'tp1': index + (tp1 - utterance['sentence_id']),
        }
        return result

class StoriesOUData(WikiOUData):

    def __init__(
            self, filepath, train, data_dim, n_segments, n_obs_seg,
            tokenizer_name, unit,
            seed, one_hot_labels, config):

        config.use_section_ids = False

        super(StoriesOUData, self).__init__(
            filepath=filepath, train=train,
            data_dim=data_dim, n_segments=n_segments,
            n_obs_seg=n_obs_seg,
            tokenizer_name=tokenizer_name,
            unit=unit, seed=seed,
            one_hot_labels=one_hot_labels,
            config=config
        )


    def _process_data(self):
        with open(self.filepath, 'rb') as f:
            self.data = json.load(f)

        self.data = self.data[:2635185]
        self.processed_data = self.data

        pass

    def __getitem__(self, index):
        if self.data[index]['story_id'] != self.data[index+1]['story_id']:
            index -= 1

        y_t = self.data[index]['sentence']
        y_tp1 = self.data[index+1]['sentence']
        t = self.data[index]['sentence_id']/self.data[index]['total_sentences']
        dt = 1./self.data[index]['total_sentences']

        result = {
            'y_t': y_t,
            'y_tp1': y_tp1,
            't': t,
            'dt': dt
        }
        return result

    def __len__(self):
        return len(self.data) - 1

class BinaryWikiSectionData(WikiSectionData):
    def __init__(
            self, filepath, train, data_dim, n_segments, n_obs_seg,
            unit,
            positive_rate, tokenizer_name, seed, one_hot_labels):
        """
        Args:
            filepath: str; filepath to json data file
            train: bool; whether to train
            n_obs_seg: int; top k sentences to take from each section
        """
        super().__init__(
            filepath=filepath, train=train, data_dim=data_dim, tokenizer_name=tokenizer_name,
            unit=unit,
            n_segments=n_segments, n_obs_seg=n_obs_seg, seed=seed, one_hot_labels=one_hot_labels)

        self.positive_rate = positive_rate

    def get_pairing(self, label, index):
        doc_idx_i = index // self.n_segments
        sec_idx_i = index % self.n_segments
        if label: # positive
            doc_idx_j = doc_idx_i
            if sec_idx_i == self.n_segments - 1:
                sec_idx_i -= 1
                sec_idx_j = self.n_segments-1
            else:
                sec_idx_j = sec_idx_i + 1
        else: # uniform sampling
            doc_idx_j = np.random.randint(0, high=self.__len__()) // self.n_segments
            # Ensure uniqueness
            while doc_idx_j == doc_idx_i:
                doc_idx_j = np.random.randint(0, high=self.__len__()) // self.n_segments
            sec_idx_j = np.random.randint(0, len(self.section_names))
            # sec_idx_j = sec_idx_i

        sentence_i = self.get_section(doc_idx=doc_idx_i, section_idx=sec_idx_i)
        sentence_j = self.get_section(doc_idx=doc_idx_j, section_idx=sec_idx_j)
        return sentence_i, sentence_j, sec_idx_i, sec_idx_j

    def __getitem__(self, index):
        label = np.array(np.random.binomial(1, self.positive_rate))
        o_t, o_x, i, j = self.get_pairing(label=label, index=index)

        result = {
            'o_t': o_t,
            'o_t*': o_x,
            'i': i,
            'j': j,
            'label': label
        }
        return result



## TODO EDIT FOR STORIES
class ROCStories(WikiOUData):

    def __init__(
            self, filepath, train, data_dim, n_segments, n_obs_seg,
            tokenizer_name, unit,
            seed, one_hot_labels, config):
        super().__init__(
            filepath, train, data_dim, n_segments, n_obs_seg,
            tokenizer_name, unit,
            seed, one_hot_labels, config
        )

    def _set_section_names(self):
        # For movies: https://github.com/google-research-datasets/Taskmaster/tree/master/TM-3-2020
        self.section_names = []
        self.section_ids = []

        self.data_dir = "/nlp/scr/rewang/ilm/data/char_masks/roc_stories/"
        if self.train:
            self.fname = os.path.join(self.data_dir, "train.pkl")
        else:
            self.fname = os.path.join(self.data_dir, "valid.pkl")

    def _process_data(self):
        self.processed_data = []
        doc_id = 0
        min_length = np.inf
        split_pattern = ". "
        data = pickle.load(open(self.fname, "rb"))
        dataset_size = 4000 if self.train else 1000
        for i, example in enumerate(data):
            if i >= dataset_size:
                break
            story = example[0]
            title, text = story.split('\n')
            text = text.split(split_pattern)
            text = [t + split_pattern for t in text if t[-1] != "."]
            story = [title] + text
            if len(story) <= 3:
                continue
            for sentence_counter, sentence in enumerate(story):
                sentence_info = {
                    "sentence": sentence,
                    "sentence_id": sentence_counter,
                    "doc_id": doc_id,
                    "total_doc_sentences": len(story)
                }
                self.processed_data.append(sentence_info)
            doc_id += 1

    def __len__(self):
        # NOTE temporary for fast iteration
        # return len(self.processed_data)
        return int(len(self.processed_data)/10)

class ROCStoriesRandomT(ROCStories):

    def __init__(
            self,
            filepath,
            train,
            config,
            all_dataset=None,
            data_dim=0,
            n_segments=0,
            n_obs_seg=0,
            tokenizer_name='GPT2',
            unit='sentence',
            seed=1,
            one_hot_labels=False,
    ):
        super().__init__(
            filepath=filepath,
            train=train,
            config=config,
            data_dim=data_dim,
            n_segments=n_segments,
            n_obs_seg=n_obs_seg,
            tokenizer_name=tokenizer_name,
            unit=unit,
            seed=seed,
            one_hot_labels=one_hot_labels
        )
        self.k = self.config.k

    def __getitem__(self, index):
        utterance = self.processed_data[index]
        sentence_num = utterance['sentence_id']

        # Check if index is start of a seq. If so -> +2
        if sentence_num == 0:
            index += 2
        if sentence_num == 1:
            index += 1

        # Update
        utterance = self.processed_data[index]
        sentence_num = utterance['sentence_id']

        # TRIAL 2: Sample all random points, t, t', t''
        T = sentence_num
        # t is a random point in between
        nums = list(range(T))
        t1 = random.choice(nums)
        nums.remove(t1)
        t2 = random.choice(nums)
        if t2 < t1:
            t = t2
            t2 = t1
            t1 = t

        assert t1 < t2 and t2 < T
        y_0 = self.processed_data[index - T + t1]['sentence']
        y_t = self.processed_data[index - T + t2]['sentence']
        y_T = self.processed_data[index]['sentence']

        t_ = t1
        t = t2

        total_doc = utterance['total_doc_sentences']
        result = {
            'y_0': y_0,
            'y_t': y_t,
            'y_T': y_T,
            't_': t_,
            't': t,
            'T': T,
            'total_t': total_doc,
        }
        return result


class ROCStoriesDiscourse(ROCStories):

    def __init__(
            self,
            filepath,
            train,
            config,
            all_dataset=None,
            data_dim=0,
            n_segments=0,
            n_obs_seg=0,
            tokenizer_name='GPT2',
            unit='sentence',
            seed=1,
            one_hot_labels=False,
    ):
        super().__init__(
            filepath=filepath,
            train=train,
            config=config,
            data_dim=data_dim,
            n_segments=n_segments,
            n_obs_seg=n_obs_seg,
            tokenizer_name=tokenizer_name,
            unit=unit,
            seed=seed,
            one_hot_labels=one_hot_labels
        )
        self.k = self.config.k

        print("examples")
        for _ in range(10):
            idx = np.random.randint(self.__len__())
            print(self.__getitem__(idx))

    def __getitem__(self, index):
        label = random.randint(0, 1) # either in- or out-of-order

        # Setup 2: sample t+k utterance
        utterance = self.processed_data[index]
        tp1 = min(utterance['total_doc_sentences']-1, utterance['sentence_id']+self.k)
        # t = max(0, tp1-self.k)
        t = max(1, tp1-self.k) # don't sample the title

        y_t = self.processed_data[index + (t - utterance['sentence_id'])]
        y_tp1 = self.processed_data[index + (tp1 - utterance['sentence_id'])]

        assert y_t['doc_id'] == y_tp1['doc_id']

        y_t = y_t['sentence']
        y_tp1 = y_tp1['sentence']

        if label: # in order
            pass # do nothing
        else:
            # # METHOD 1: FLIP ORDER
            # tmp = y_tp1
            # y_tp1 = y_t
            # y_t = tmp

            # METHOD 2: RANDOM
            random_idx = np.random.randint(len(self.processed_data))
            y_tp1 = self.processed_data[random_idx]['sentence']

        if self.one_hot_labels:
            labels = torch.zeros(2)
            labels[label] = 1.0
            label = labels

        result = {
            'y_t': y_t,
            'y_tp1': y_tp1,
            't': t,
            'tp1': tp1,
            'label': label,
            'idx': index
        }
        return result


class ROCStoriesTPKData(ROCStoriesDiscourse):
    # t, t + k
    # and t-1, t-2, t-3

    def __init__(
            self,
            filepath,
            train,
            config,
            all_dataset=None,
            data_dim=0,
            n_segments=0,
            n_obs_seg=0,
            tokenizer_name='GPT2',
            unit='sentence',
            seed=1,
            one_hot_labels=False,
    ):
        super().__init__(
            filepath=filepath,
            train=train,
            config=config,
            data_dim=data_dim,
            n_segments=n_segments,
            n_obs_seg=n_obs_seg,
            tokenizer_name=tokenizer_name,
            unit=unit,
            seed=seed,
            one_hot_labels=one_hot_labels
        )

    def __getitem__(self, index):
        if self.config.k == 1:
            if self.processed_data[index]['doc_id'] != self.processed_data[index+1]['doc_id']:
                index -= 1

            y_t = self.processed_data[index]['sentence']
            y_tp1 = self.processed_data[index+1]['sentence']
            t = self.processed_data[index]['sentence_id']/self.processed_data[index]['total_doc_sentences']
            dt = 1./self.processed_data[index]['total_doc_sentences']
        else:
            assert self.config.k > 1
            # k sampling
            utterance = self.processed_data[index]
            tp1 = min(utterance['total_doc_sentences']-1,
                      utterance['sentence_id']+self.config.k)
            t = max(0, tp1-self.config.k)

            dt = (tp1 - t)/utterance['total_doc_sentences']
            y_t = self.processed_data[index + (t - utterance['sentence_id'])]['sentence']
            y_tp1 = self.processed_data[index + (tp1 - utterance['sentence_id'])]['sentence']
            t = self.processed_data[index + (t - utterance['sentence_id'])]['sentence_id']/utterance['total_doc_sentences']

        y_tm1 = (self.processed_data[index] if (index - 1 < 0 or self.processed_data[index]['doc_id'] != self.processed_data[index-1]['doc_id']) else self.processed_data[index-1])
        y_tm1 = y_tm1['sentence']
        y_tm2 = (self.processed_data[index] if (index - 2 < 0 or self.processed_data[index]['doc_id'] != self.processed_data[index-2]['doc_id']) else self.processed_data[index-2])
        y_tm2 = y_tm2['sentence']
        y_tm3 = (self.processed_data[index] if (index - 3 < 0 or self.processed_data[index]['doc_id'] != self.processed_data[index-3]['doc_id']) else self.processed_data[index-3])
        y_tm3 = y_tm3['sentence']


        result = {
            'y_t': y_t,
            'y_tm1': y_tm1,
            'y_tm2': y_tm2,
            'y_tm3': y_tm3,
            'y_tpk': y_tp1,
        }
        return result

