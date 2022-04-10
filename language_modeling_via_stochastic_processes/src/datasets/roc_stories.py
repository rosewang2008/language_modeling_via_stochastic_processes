import sys
import os
import random
import pickle
import tqdm
import torch
import numpy as np
from language_modeling_via_stochastic_processes.src import constants

from language_modeling_via_stochastic_processes.src.datasets import encoder

class ROCStoriesDataset(encoder.BaseDataset):

    def __init__(
            self,
            train,
            tokenizer_name,
            seed,
            config,
            all_dataset=None):
        super().__init__(
            train=train,
            all_dataset=all_dataset,
            config=config,
            tokenizer_name=tokenizer_name,
            seed=seed,
        )

    def _set_section_names(self):
        self.section_names = []
        self.section_ids = []

        self.data_dir = constants.PATH2ROCSTORIES
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
        for example in tqdm.tqdm(data):
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
        return len(self.processed_data)

class ROCStoriesTriplet(ROCStoriesDataset):

    def __init__(
            self,
            train,
            config,
            all_dataset=None,
            tokenizer_name='GPT2',
            seed=1,
    ):
        super().__init__(
            train=train,
            all_dataset=all_dataset,
            config=config,
            tokenizer_name=tokenizer_name,
            seed=seed,
        )
        self.k = self.config.data_params.k

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


class ROCStoriesDiscourse(ROCStoriesDataset):

    def __init__(
            self,
            train,
            config,
            all_dataset=None,
            tokenizer_name='GPT2',
            seed=1,
    ):
        super().__init__(
            train=train,
            all_dataset=all_dataset,
            config=config,
            tokenizer_name=tokenizer_name,
            seed=seed,
        )
        self.k = self.config.data_params.k

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

class ROCStoriesTPK(ROCStoriesDataset):

    def __init__(
            self,
            train,
            all_dataset,
            config,
            tokenizer_name="GPT2",
            seed=1,
    ):
        """
        """
        super(ROCStoriesTPK, self).__init__(
            train=train,
            all_dataset=all_dataset,
            config=config,
            tokenizer_name=tokenizer_name,
            seed=seed,

        )

    def __getitem__(self, index):
        if self.config.data_params.k == 1:
            if self.processed_data[index]['doc_id'] != self.processed_data[index+1]['doc_id']:
                index -= 1

            y_t = self.processed_data[index]['sentence']
            y_tp1 = self.processed_data[index+1]['sentence']
            t = self.processed_data[index]['sentence_id']/self.processed_data[index]['total_doc_sentences']
        else:
            # k sampling
            utterance = self.processed_data[index]
            tp1 = min(utterance['total_doc_sentences']-1,
                      utterance['sentence_id']+self.config.data_params.k)
            t = max(0, tp1-self.config.data_params.k)

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
