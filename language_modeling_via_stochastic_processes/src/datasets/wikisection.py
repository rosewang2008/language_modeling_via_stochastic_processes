import torch
import random
import os
import json

from language_modeling_via_stochastic_processes.src.datasets import encoder
from language_modeling_via_stochastic_processes.src import constants

class WikisectionTriplet(encoder.BaseDataset):

    def __init__(
            self,
            train,
            seed,
            config,
            all_dataset=None,
            tokenizer_name="GPT2",
            ):

        dir_path = constants.PATH2WIKISECTION
        if train:
            self.filepath = os.path.join(dir_path, "HGD_en_city_train.json")
        else:
            self.filepath = os.path.join(dir_path, "HGD_en_city_test.json")

        super().__init__(
            train=train,
            tokenizer_name=tokenizer_name,
            all_dataset=all_dataset,
            seed=seed,
            config=config
        )

    def _load_data(self):
        with open(self.filepath, 'rb') as f:
            self.data = json.load(f)

    def _set_section_names(self):
        self.section_names = ['abstract', 'History', 'Geography', 'Demographics']
        self.section_ids = ['[ {} ]'.format(name.upper()) for name in self.section_names]

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

    def __len__(self):
        return len(self.processed_data) - 1

class WikisectionDiscourse(WikisectionTriplet):

    def __init__(
            self,
            train,
            config,
            all_dataset=None,
            tokenizer_name='GPT2',
            seed=1,
    ):
        super(WikisectionDiscourse, self).__init__(
            train=train,
            all_dataset=all_dataset,
            config=config,
            tokenizer_name=tokenizer_name,
            seed=seed,
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
        k = self.config.data_params.k

        utterance = self.processed_data[index]
        tp1 = min(utterance['total_doc_sentences']-1, utterance['sentence_id']+k)
        t = max(0, tp1-k)

        y_t = self.processed_data[index + (t - utterance['sentence_id'])]
        y_tp1 = self.processed_data[index + (tp1 - utterance['sentence_id'])]

        assert y_t['doc_id'] == y_tp1['doc_id']

        y_t = y_t['sentence']
        y_tp1 = y_tp1['sentence']

        if not label:
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

class WikisectionTPK(WikisectionTriplet):
    # t, t + k
    # and t-1, t-2, t-3
    def __init__(
            self,
            config,
            train,
            all_dataset,
            seed,
            tokenizer_name='GPT2',
            ):

        super().__init__(
            train=train,
            tokenizer_name=tokenizer_name,
            seed=seed,
            all_dataset=all_dataset,
            config=config
        )

    def __getitem__(self, index):
        k = self.config.data_params.k

        if k == 1:
            if self.processed_data[index]['doc_id'] != self.processed_data[index+1]['doc_id']:
                index -= 1

            y_t = self.processed_data[index]['sentence']
            y_tp1 = self.processed_data[index+1]['sentence']
            t = self.processed_data[index]['sentence_id']/self.processed_data[index]['total_doc_sentences']
        else:
            assert k > 1
            # k sampling
            utterance = self.processed_data[index]
            tp1 = min(utterance['total_doc_sentences']-1,
                      utterance['sentence_id']+k)
            t = max(0, tp1-k)

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
