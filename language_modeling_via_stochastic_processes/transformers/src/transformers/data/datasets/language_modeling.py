# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional
import copy

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock
import datasets
from tqdm import tqdm

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
from collections import defaultdict

from transformers import (
    GPT2Tokenizer,
)

from language_modeling_via_stochastic_processes.src import constants

logger = logging.get_logger(__name__)


DEPRECATION_WARNING = (
    "This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: {0}"
)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            f"cached_lm_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should look for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info(f"Creating features from dataset file at {file_path}")

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class LineByLineWithRefDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, ref_path: str):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm_wwm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        assert os.path.isfile(ref_path), f"Ref file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info(f"Creating features from dataset file at {file_path}")
        logger.info(f"Use ref segment results at {ref_path}")
        with open(file_path, encoding="utf-8") as f:
            data = f.readlines()  # use this method to avoid delimiter '\u2029' to split a line
        data = [line.strip() for line in data if len(line) > 0 and not line.isspace()]
        # Get ref inf from file
        with open(ref_path, encoding="utf-8") as f:
            ref = [json.loads(line) for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        assert len(data) == len(ref)

        batch_encoding = tokenizer(data, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

        n = len(self.examples)
        for i in range(n):
            self.examples[i]["chinese_ref"] = torch.tensor(ref[i], dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class LineByLineWithSOPTextDataset(Dataset):
    """
    Dataset for sentence order prediction task, prepare sentence pairs for SOP task
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_dir: str, block_size: int):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isdir(file_dir)
        logger.info(f"Creating features from dataset file folder at {file_dir}")
        self.examples = []
        # TODO: randomness could apply a random seed, ex. rng = random.Random(random_seed)
        # file path looks like ./dataset/wiki_1, ./dataset/wiki_2
        for file_name in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file_name)
            assert os.path.isfile(file_path)
            article_open = False
            with open(file_path, encoding="utf-8") as f:
                original_lines = f.readlines()
                article_lines = []
                for line in original_lines:
                    if "<doc id=" in line:
                        article_open = True
                    elif "</doc>" in line:
                        article_open = False
                        document = [
                            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
                            for line in article_lines[1:]
                            if (len(line) > 0 and not line.isspace())
                        ]

                        examples = self.create_examples_from_document(document, block_size, tokenizer)
                        self.examples.extend(examples)
                        article_lines = []
                    else:
                        if article_open:
                            article_lines.append(line)

        logger.info("Dataset parse finished.")

    def create_examples_from_document(self, document, block_size, tokenizer, short_seq_prob=0.1):
        """Creates examples for a single document."""

        # Account for special tokens
        max_num_tokens = block_size - tokenizer.num_special_tokens_to_add(pair=True)

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pretraining and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.
        target_seq_length = max_num_tokens
        if random.random() < short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)

        # We DON'T just concatenate all of the tokens from a document into a long
        # sequence and choose an arbitrary split point because this would make the
        # next sentence prediction task too easy. Instead, we split the input into
        # segments "A" and "B" based on the actual "sentences" provided by the user
        # input.
        examples = []
        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]  # get a segment
            if not segment:
                i += 1
                continue
            current_chunk.append(segment)  # add a segment to current chunk
            current_length += len(segment)  # overall token length
            # if current length goes to the target length or reaches the end of file, start building token a and b
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A` (first) sentence.
                    a_end = 1
                    # if current chunk has more than 2 sentences, pick part of it `A` (first) sentence
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    # token a
                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    # token b
                    tokens_b = []
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                    if len(tokens_a) == 0 or len(tokens_b) == 0:
                        continue

                    # switch tokens_a and tokens_b randomly
                    if random.random() < 0.5:
                        is_next = False
                        tokens_a, tokens_b = tokens_b, tokens_a
                    else:
                        is_next = True

                    def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
                        """Truncates a pair of sequences to a maximum sequence length."""
                        while True:
                            total_length = len(tokens_a) + len(tokens_b)
                            if total_length <= max_num_tokens:
                                break
                            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                            assert len(trunc_tokens) >= 1
                            # We want to sometimes truncate from the front and sometimes from the
                            # back to add more randomness and avoid biases.
                            if random.random() < 0.5:
                                del trunc_tokens[0]
                            else:
                                trunc_tokens.pop()

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    # add special tokens
                    input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                        "sentence_order_label": torch.tensor(0 if is_next else 1, dtype=torch.long),
                    }
                    examples.append(example)
                current_chunk = []  # clear current chunk
                current_length = 0  # reset current text length
            i += 1  # go to next line
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class TextDatasetForNextSentencePrediction(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        short_seq_probability=0.1,
        nsp_probability=0.5,
    ):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.short_seq_probability = short_seq_probability
        self.nsp_probability = nsp_probability

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            f"cached_nsp_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        self.tokenizer = tokenizer

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        #
        # Example:
        # I am very happy.
        # Here is the second sentence.
        #
        # A new document.

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.documents = [[]]
                with open(file_path, encoding="utf-8") as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()

                        # Empty lines are used as document delimiters
                        if not line and len(self.documents[-1]) != 0:
                            self.documents.append([])
                        tokens = tokenizer.tokenize(line)
                        tokens = tokenizer.convert_tokens_to_ids(tokens)
                        if tokens:
                            self.documents[-1].append(tokens)

                logger.info(f"Creating examples from {len(self.documents)} documents.")
                self.examples = []
                for doc_index, document in enumerate(self.documents):
                    self.create_examples_from_document(document, doc_index, block_size)

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def create_examples_from_document(self, document: List[List[int]], doc_index: int, block_size: int):
        """Creates examples for a single document."""

        max_num_tokens = block_size - self.tokenizer.num_special_tokens_to_add(pair=True)

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pretraining and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_probability:
            target_seq_length = random.randint(2, max_num_tokens)

        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0

        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []

                    if len(current_chunk) == 1 or random.random() < self.nsp_probability:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(10):
                            random_document_index = random.randint(0, len(self.documents) - 1)
                            if random_document_index != doc_index:
                                break

                        random_document = self.documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    # Actual next
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    # add special tokens
                    input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                        "next_sentence_label": torch.tensor(1 if is_random_next else 0, dtype=torch.long),
                    }

                    self.examples.append(example)

                current_chunk = []
                current_length = 0

            i += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

class WikisectionDataset(TextDataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self,
                 cl_model,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int,
                 special_words: list,
                 use_section_null: bool,
                 overwrite_cache=False,
                 cache_dir: Optional[str] = None,
                 ):
        super(WikisectionDataset, self).__init__(
                 tokenizer=tokenizer,
                 file_path=file_path,
                 block_size=block_size,
                 overwrite_cache=overwrite_cache,
                 cache_dir=cache_dir,
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cl_model = cl_model
        self.use_section_null = use_section_null
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.examples = []
        self.raw_texts = []
        self.cl_texts = []
        self.section_ids = []
        self.cl_embeddings = []
        self.n_big = 0
        self.n_small = 0
        self.cpu_device = torch.device('cpu')
        self.cl_offset = 0
        self.lengths = defaultdict(lambda: [])
        self.section_idx_offset = 1
        self.special_words = special_words

        if 'long' in file_path:
            assert len(self.special_words) > 5
        else:
            assert len(self.special_words) == 5

        # string form of id's
        self.section_names = self.special_words[:-1]
        self.cl_eos_str = self.special_words[-1]
        assert self.cl_eos_str == ' . '
        # id token
        section_tokens = self.tokenizer(self.section_names)['input_ids']
        self.section_tokens = [tok[0] for tok in section_tokens]
        self.cl_eos_id = self.tokenizer(self.cl_eos_str)['input_ids'][0]
        assert self.cl_eos_id > 50000 # just checking its a new token

        self.set_cl_tokenizer()
        start = time.time()
        self.process_dataset()
        end = time.time()
        print("Processing dataset took {}".format(end-start))

    def process_dataset(self):
        with open(self.file_path, encoding="utf-8") as f:
            for idx, row in enumerate(f.readlines()):
                if row.strip():
                    # Text used for CL embeddings.
                    cl_text = self._clean2cltext(row)
                    # Text for GPT2
                    row = row.strip() # NOTE: remove break line
                    row = row.replace(". ", " . ") # NOTE marking end of sentence
                    row = f"{self.tokenizer.bos_token} {row} {self.tokenizer.eos_token}"
                    tokenized_text = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(row))

                    last_section_id = 0
                    if len(tokenized_text) >= self.block_size:
                        pass
                        # skip / filter large exmaples
                    else:
                        self.n_small += 1
                        example = self.tokenizer.build_inputs_with_special_tokens(tokenized_text)
                        self.examples.append(example)
                        section_ids, _ = self._determine_section_ids(example, last_section_id)
                        self.section_ids.append(section_ids)
                        self.raw_texts.append(self.tokenizer.decode(self.examples[-1]))
                        self.cl_texts.append(cl_text)
                        self._get_cl_embeddings(tokenized_example=example, gpt2_text=row,
                                                raw_text=self.raw_texts[-1],
                                                cl_text=self.cl_texts[-1])
                        if len(self.examples) > 1422:
                            break # same length as toy wikisection

        self.labels = copy.deepcopy(self.examples)
        print(f"big: {self.n_big} vs small: {self.n_small}")
        for k, v in self.lengths.items():
            print("[ {} ] {}+-{}".format(k, np.mean(v), np.std(v)/np.sqrt(len(v) )))


    def set_cl_tokenizer(self):
        self.cl_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.cl_tokenizer.pad_token = self.cl_tokenizer.eos_token
        self.cl_tokenizer.add_tokens(self.section_names)

    def cl_tokenize(self, text, device):
        output = self.cl_tokenizer(
            text,
            padding=True,
            return_tensors='pt',
        )
        input_ids = output['input_ids'].squeeze(0)
        attention_mask = output['attention_mask'].squeeze(0)
        eos_input_ids = torch.tensor([[self.cl_tokenizer.eos_token_id]*input_ids.shape[0]])
        eos_attention = torch.tensor([[0]*input_ids.shape[0]])
        input_ids = torch.cat((input_ids, eos_input_ids.T), dim=1)
        attention_mask = torch.cat((attention_mask, eos_attention.T), dim=1)
        return input_ids.to(device), attention_mask.to(device)


    def _full_section_ids(self, tokenized_text, last_section_id):
        """output an array \in [0, 3]"""
        section_ids = np.zeros(len(tokenized_text))
        section_tokens = self.tokenizer(self.section_names)['input_ids']
        section_tokens = [tok[0] for tok in section_tokens]

        # Getting the first section token
        start_tok = None
        for tok in section_tokens:
            if tok in tokenized_text:
                start_tok = tok
                break

        if start_tok is None:
            start_tok = section_tokens[last_section_id]
            start_token_idx = 0
            pause = True
        else:
            start_token_idx = tokenized_text.index(start_tok)
            pause = False

        # Handle off-by-one (feed in new section before the section actually starts)
        start_token_idx -= self.section_idx_offset
        start_token_idx = max(0, start_token_idx)

        if start_tok != section_tokens[0]: # doesn't start with abstract
            section_ids[:start_token_idx] = section_tokens.index(start_tok - 1) # \in [0, 3]

        for next_tok in section_tokens[section_tokens.index(start_tok)+1:]:
            # if next_tok is not in text, then the rest is for start_tok
            if next_tok not in tokenized_text:
                section_ids[start_token_idx:] = section_tokens.index(start_tok) # \in [0, 3]
                break
            else:
                # Handle off-by-one
                next_tok_idx = tokenized_text.index(next_tok) - self.section_idx_offset
                next_tok_idx = max(0, next_tok_idx)
                section_ids[start_token_idx:next_tok_idx] = section_tokens.index(start_tok) # \in [0, 3]

            self.lengths[self.section_names[section_tokens.index(start_tok)]].append(
                next_tok_idx + 1 - start_token_idx)
            start_tok = next_tok
            start_token_idx = next_tok_idx

        if start_tok == section_tokens[-1]: # end section, rest of text is this token
            section_ids[start_token_idx:] = section_tokens.index(start_tok) # \in [0, 3]
            self.lengths[self.section_names[section_tokens.index(start_tok)]].append(
                len(section_ids) - start_token_idx)

        last_section_id = int(section_ids[-1])
        return section_ids, last_section_id

    def _null_section_id(self, tokenized_text, last_section_id):
        """output an array \in [0, 4] where 4 = null"""
        section_tokens = self.tokenizer(self.section_names)['input_ids']
        section_tokens = [tok[0] for tok in section_tokens]
        NULL_ID = len(section_tokens)
        section_ids = np.ones(len(tokenized_text)) * NULL_ID

        for section_id, section_tok in enumerate(section_tokens):
            if section_tok in tokenized_text:
                tok_idx = tokenized_text.index(section_tok) - self.section_idx_offset
                tok_idx = max(0, tok_idx)
                section_ids[tok_idx] = section_id
                last_section_id = section_id

        return section_ids, last_section_id


    def _determine_section_ids(self, tokenized_text, last_section_id):
        if self.use_section_null:
            section_ids, last_section_id = self._null_section_id(tokenized_text, last_section_id)
        else:
            section_ids, last_section_id = self._full_section_ids(tokenized_text, last_section_id)
        return section_ids, last_section_id

    def _clean2cltext(self, row):
        # Remove section tokens from text.
        for tok in self.section_names:
            row = row.replace(tok, "")
        cl_text = row.replace(".\n", ". ")
        return cl_text

    def _get_cl_embeddings(self, tokenized_example, gpt2_text, raw_text, cl_text):
        return self.get_cl_embeddings(tokenized_example, gpt2_text, raw_text, cl_text)

    def get_end_points(self, tokenized_example):
        eos_idxs = [i-1 for i, x in enumerate(tokenized_example) if x == self.cl_eos_id]
        eos_idxs += [len(tokenized_example)]
        return eos_idxs

    def get_cl_embeddings(self, tokenized_example, gpt2_text, raw_text, cl_text):
        split_pattern = " . "
        cl_embeddings = []
        eos_idxs = self.get_end_points(tokenized_example)
        split_sentences = gpt2_text.split(split_pattern)
        split_sentences = [_ + split_pattern for _ in split_sentences[:-1]] + [split_sentences[-1]]
        assert len(eos_idxs) == len(split_sentences)
        cl_input_ids, cl_attention_mask = self.cl_tokenize(split_sentences, self.device)
        cl_feats = self.cl_model.forward(input_ids=cl_input_ids, attention_mask=cl_attention_mask) # 1, feat_size
        # Align feats to the sentence length
        last_idx = 0
        for eos_idx, feat in zip(eos_idxs, cl_feats):
            cl_embeddings += [feat] * (eos_idx - last_idx)
            last_idx = eos_idx

        assert len(cl_embeddings) == len(tokenized_example)

        if self.cl_offset:
            cl_embeddings = cl_embeddings[self.cl_offset:] + [cl_embeddings[-1]] * self.cl_offset
        self.cl_embeddings.append(cl_embeddings)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.section_ids[i], dtype=torch.long),
                torch.stack(self.cl_embeddings[i]).to(self.cpu_device),
                self.cl_texts[i]
                )

class StoriesDataset(TextDataset):
    """
    ROC STORIES
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int,
                 special_words: list,
                 cl_model,
                 overwrite_cache=False,
                 cache_dir: Optional[str] = None,
                 ):
        from language_modeling_via_stochastic_processes.src import constants
        fpath = os.path.join(constants.PATH2WIKISECTION, "wikisection_withSections.train.txt")
        super(StoriesDataset, self).__init__(
                 tokenizer=tokenizer,
                 file_path=fpath,
                 block_size=block_size,
                 overwrite_cache=overwrite_cache,
                 cache_dir=cache_dir,
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cpu_device = torch.device('cpu')
        self.cl_model = cl_model
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.examples = []
        self.raw_texts = []
        self.cl_texts = []
        self.section_ids = []
        self.cl_embeddings = []
        self.special_words = special_words
        self.cl_offset=0

        import sys
        sys.path.append("/nlp/scr/rewang/ilm")
        import ilm
        self.cl_eos_str = self.special_words[-1]
        assert self.cl_eos_str == ' . '
        self.cl_eos_id = self.tokenizer(self.cl_eos_str)['input_ids'][0]
        assert self.cl_eos_id > 50000 # just checking its a new token
        self.set_cl_tokenizer()
        start = time.time()
        self.process_dataset()
        end = time.time()
        print("Processing dataset took {}".format(end-start))

    def set_cl_tokenizer(self):
        self.cl_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.cl_tokenizer.pad_token = self.cl_tokenizer.eos_token

        self.cl_tokenizer.add_tokens(self.special_words)

    def cl_tokenize(self, text, device):
        output = self.cl_tokenizer(
            text,
            padding=True,
            return_tensors='pt',
        )
        input_ids = output['input_ids'].squeeze(0)
        attention_mask = output['attention_mask'].squeeze(0)
        eos_input_ids = torch.tensor([[self.cl_tokenizer.eos_token_id]*input_ids.shape[0]])
        eos_attention = torch.tensor([[0]*input_ids.shape[0]])
        input_ids = torch.cat((input_ids, eos_input_ids.T), dim=1)
        attention_mask = torch.cat((attention_mask, eos_attention.T), dim=1)
        return input_ids.to(device), attention_mask.to(device)

    def process_dataset(self):
        data = pickle.load(open(self.file_path, "rb"))
        split_pattern = ". "
        for example in tqdm(data):
            story = example[0]
            title, text = story.split('\n')
            text = text.split(split_pattern)
            story = [title] + text
            if len(story) <= 3:
                continue
            row  = self.cl_eos_str.join(story)
            row = f"{self.tokenizer.bos_token} {row} {self.tokenizer.eos_token}"
            tokenized_text = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(row))

            if len(tokenized_text) >= self.block_size:
                pass
                # skip / filter large exmaples
            else:
                example = self.tokenizer.build_inputs_with_special_tokens(tokenized_text)
                self.examples.append(example)
                self.section_ids.append([0])
                self.raw_texts.append(self.tokenizer.decode(self.examples[-1]))
                self.cl_texts.append(row)
                self.get_cl_embeddings(tokenized_example=example,
                                        gpt2_text=row,
                                        raw_text=self.raw_texts[-1],
                                        cl_text=self.cl_texts[-1])

        self.labels = copy.deepcopy(self.examples)
        print("examples")
        print(self.cl_texts[0])
        print(self.cl_texts[-1])

    def get_end_points(self, tokenized_example):
        eos_idxs = [i-1 for i, x in enumerate(tokenized_example) if x == self.cl_eos_id]
        eos_idxs += [len(tokenized_example)]
        return eos_idxs

    def get_cl_embeddings(self, tokenized_example, gpt2_text, raw_text, cl_text):
        split_pattern = self.cl_eos_str
        cl_embeddings = []
        eos_idxs = self.get_end_points(tokenized_example)
        split_sentences = gpt2_text.split(split_pattern)
        split_sentences = [_ + split_pattern for _ in split_sentences[:-1]] + [split_sentences[-1]]
        assert len(eos_idxs) == len(split_sentences)
        cl_input_ids, cl_attention_mask = self.cl_tokenize(split_sentences, self.device)
        cl_feats = self.cl_model.forward(input_ids=cl_input_ids, attention_mask=cl_attention_mask) # 1, feat_size
        # Align feats to the sentence length
        last_idx = 0
        for eos_idx, feat in zip(eos_idxs, cl_feats):
            cl_embeddings += [feat] * (eos_idx - last_idx)
            last_idx = eos_idx
        assert len(cl_embeddings) == len(tokenized_example)

        if self.cl_offset:
            cl_embeddings = cl_embeddings[self.cl_offset:] + [cl_embeddings[-1]] * self.cl_offset

        self.cl_embeddings.append(cl_embeddings)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.section_ids[i], dtype=torch.long),
                torch.stack(self.cl_embeddings[i]).to(self.cpu_device),
                self.cl_texts[i]
                )

class RecipeDataset(TextDataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self,
                 cl_model,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int,
                 use_section_null: bool,
                 special_words:list,
                 data_dir=os.path.join(constants.PATH2RECIPENLG, 'dataset'),
                 overwrite_cache=False,
                 cache_dir: Optional[str] = None,
                 name: str = 'recipe'
                 ):
        super(RecipeDataset, self).__init__(
                 tokenizer=tokenizer,
                 file_path=file_path,
                 block_size=block_size,
                 overwrite_cache=overwrite_cache,
                 cache_dir=cache_dir,
        )

        self.cpu_device = torch.device('cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cl_model = cl_model
        self.lengths = defaultdict(lambda: [])
        self.special_words= special_words
        assert self.special_words # should not be emtpy
        self.special_tokens = [_[0] for _ in tokenizer(self.special_words)['input_ids']]
        self.file_path = file_path
        self.data_dir = data_dir
        self.train = 'train' in self.file_path
        self.block_size = block_size
        self.cl_offset = 0
        self._set_indices()
        self.set_cl_tokenizer()

        self.use_section_null = use_section_null
        self.tokenizer = tokenizer
        self.examples = []
        self.cl_texts = []
        self.cl_embeddings = []
        self.section_ids = []
        self.raw_texts = []

        # string form of id's
        self.special_words = special_words
        self.section_names = self.special_words[:-1]
        self.cl_eos_str = self.special_words[-1]
        assert self.cl_eos_str == ' . '
        # id token
        section_tokens = self.tokenizer(self.section_names)['input_ids']
        self.section_tokens = [tok[0] for tok in section_tokens]
        self.cl_eos_id = self.tokenizer(self.cl_eos_str)['input_ids'][0]
        assert self.cl_eos_id > 50000 # just checking its a new token

        self._process_dataset()

    def set_cl_tokenizer(self):
        self.cl_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.cl_tokenizer.pad_token = self.cl_tokenizer.eos_token

        self.cl_tokenizer.add_tokens(self.special_words)

    def cl_tokenize(self, text, device):
        output = self.cl_tokenizer(
            text,
            padding=True,
            return_tensors='pt',
        )
        input_ids = output['input_ids'].squeeze(0)
        attention_mask = output['attention_mask'].squeeze(0)
        eos_input_ids = torch.tensor([[self.cl_tokenizer.eos_token_id]*input_ids.shape[0]])
        eos_attention = torch.tensor([[0]*input_ids.shape[0]])
        input_ids = torch.cat((input_ids, eos_input_ids.T), dim=1)
        attention_mask = torch.cat((attention_mask, eos_attention.T), dim=1)
        return input_ids.to(device), attention_mask.to(device)

    def _set_indices(self):
        # SEE REPO <NONSTATIONARITY> FOR INDEXES
        if self.train:
            self.start_idx, self.end_idx = 0, 4_000
        else:
            self.start_idx, self.end_idx = 500_000, 501_000

    def _process_dataset(self):
        self.all_dataset = datasets.load_dataset("recipe_nlg", data_dir=self.data_dir)['train']
        num_filtered = 0
        for doc_id in tqdm(range(self.start_idx, self.end_idx)):
            doc = self.all_dataset[doc_id]
            # Put all the document sentences together.
            title = [doc['title'] + " . "]
            ingredients = [(', '.join(doc['ner']) + " . ").capitalize()]
            directions = [d[:-1] + " . " for d in doc['directions']]
            # Text used for CL embeddings
            all_sentences = title + ingredients + directions
            all_sentences = [s for s in all_sentences if s] # make sure sentences are not empty
            cl_text = "".join(all_sentences)
            # Text for GPT2
            gpt2_text = ([self.special_words[0] + " "] + title
                         + [self.special_words[1] + " "]  + ingredients
                         + [self.special_words[2] + " "]  + directions)
            gpt2_text = [s for s in gpt2_text if s]
            gpt2_text = "".join(gpt2_text)

            row = f"{self.tokenizer.bos_token} {gpt2_text} {self.tokenizer.eos_token}"
            tokenized_text = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(row))

            if len(tokenized_text) >= self.block_size:
                num_filtered+=1
            else:
                self.examples.append(self.tokenizer.build_inputs_with_special_tokens(tokenized_text))
                self.cl_texts.append(cl_text)
                self.get_cl_embeddings(
                    tokenized_example=tokenized_text,
                    raw_text=row,
                    cl_text=row,
                    gpt2_text=row)
                # section_ids, _ = self._determine_section_ids(tokenized_text, last_section_id=None)
                section_ids = [0]
                self.section_ids.append(section_ids)
                self.raw_texts.append(row)

        self.labels = copy.deepcopy(self.examples)
        print("num examples {}".format(len(self.examples)))
        print(f"num filtered {num_filtered}")
        print("Lengths")
        for k, v in self.lengths.items():
            print("[ {} ] {}+-{}".format(k, np.mean(v), np.std(v) ))

    def get_end_points(self, tokenized_example):
        eos_idxs = [i-1 for i, x in enumerate(tokenized_example) if x == self.cl_eos_id]
        eos_idxs += [len(tokenized_example)]
        return eos_idxs

    def get_cl_embeddings(self, tokenized_example, raw_text, cl_text, gpt2_text):
        split_pattern = " . "
        cl_embeddings = []
        eos_idxs = self.get_end_points(tokenized_example)
        split_sentences = gpt2_text.split(split_pattern)
        split_sentences = [_ + split_pattern for _ in split_sentences[:-1]] + [split_sentences[-1]]
        assert len(eos_idxs) == len(split_sentences)
        cl_input_ids, cl_attention_mask = self.cl_tokenize(split_sentences, self.device)
        cl_feats = self.cl_model.forward(input_ids=cl_input_ids, attention_mask=cl_attention_mask) # 1, feat_size
        # Align feats to the sentence length
        last_idx = 0
        for eos_idx, feat in zip(eos_idxs, cl_feats):
            cl_embeddings += [feat] * (eos_idx - last_idx)
            last_idx = eos_idx

        assert len(cl_embeddings) == len(tokenized_example)

        if self.cl_offset:
            cl_embeddings = cl_embeddings[self.cl_offset:] + [cl_embeddings[-1]] * self.cl_offset
        self.cl_embeddings.append(cl_embeddings)

    def set_cl_tokenizer(self):
        self.cl_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.cl_tokenizer.pad_token = self.cl_tokenizer.eos_token
        self.cl_tokenizer.add_tokens(self.special_words)

    def __len__(self):
        return len(self.examples)


    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.section_ids[i], dtype=torch.long),
                torch.stack(self.cl_embeddings[i]).to(self.cpu_device),
                self.cl_texts[i]
                )

    def _full_section_ids(self, tokenized_text, last_section_id):
        """output an array \in [0, 3]"""
        section_ids = np.zeros(len(tokenized_text))
        section_tokens = self.special_tokens

        start_idx = 0
        cur_tok = section_tokens[0]
        for tok in section_tokens[1:]: # skip abstract - always first and present
            if tok in tokenized_text:
                end_idx = tokenized_text.index(tok) - 1
                section_ids[start_idx:end_idx] = section_tokens.index(cur_tok)
                # Track length
                self.lengths[self.special_words[section_tokens.index(cur_tok)]].append(
                    end_idx + 1 - start_idx)
                # Update to next token
                cur_tok = tok
                start_idx = end_idx
        section_ids[start_idx:] = section_tokens.index(cur_tok)
        # Track length
        self.lengths[self.special_words[section_tokens.index(cur_tok)]].append(
            len(tokenized_text) - start_idx)
        last_section_id = cur_tok
        return section_ids, last_section_id

    def _null_section_id(self, tokenized_text, last_section_id):
        """output an array \in [0, 4] where 4 = null"""
        section_tokens = self.special_tokens
        NULL_ID = len(section_tokens)
        section_ids = np.ones(len(tokenized_text)) * NULL_ID

        for section_id, section_tok in enumerate(section_tokens):
            if section_tok in tokenized_text:
                tok_idx = tokenized_text.index(section_tok) - 1
                section_ids[tok_idx] = section_id
                last_section_id = section_id
        return section_ids, last_section_id


    def _determine_section_ids(self, tokenized_text, last_section_id):
        if self.use_section_null:
            section_ids, last_section_id = self._null_section_id(tokenized_text, last_section_id)
        else:
            section_ids, last_section_id = self._full_section_ids(tokenized_text, last_section_id)
        return section_ids, last_section_id


class TaskmasterDataset(RecipeDataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self,
                 cl_model,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int,
                 use_section_null: bool,
                 special_words:list,
                 data_dir=os.path.join(constants.PATH2RECIPENLG, 'dataset'),
                 overwrite_cache=False,
                 cache_dir: Optional[str] = None,
                 name: str = 'wikihow'
                 ):
        self.name = name
        self.train = True if 'train' in file_path else False
        super(TaskmasterDataset, self).__init__(
                cl_model=cl_model,
                 tokenizer=tokenizer,
                 file_path=file_path,
                 block_size=block_size,
                 use_section_null=use_section_null,
                 special_words=special_words,
                 data_dir=os.path.join(constants.PATH2RECIPENLG, 'dataset'),
                 overwrite_cache=False,
                 cache_dir=cache_dir,
                name=name
        )

    def _set_indices(self):
        if "tm2" in self.name:
            print('LOADING RESTAURANT TM')
            self.data_dir = constants.PATH2TM2
            if self.train:
                self.data_files = ['restaurant-search.json'] # 3276 conversations
                self.start_conversation = 0
                self.end_conversation = 2000
            else:
                self.data_files = ['restaurant-search.json']
                self.start_conversation = 2000
                self.end_conversation = 3276
        else:
            print('LOADING MOVIE TM')
            self.data_dir = constants.PATH2TICKETTALK
            if self.train:
                self.data_files = ['data_0{}.json'.format(i) for i in range(0, 3)]
            else:
                self.data_files = ['data_{}.json'.format(i) for i in range(13, 14)]

    def _process_dataset(self):
        num_filtered = 0

        self.processed_data = []
        split_pattern = ".  "
        doc_counter = 0
        # self.lengths = defaultdict(lambda: [])
        for fname in self.data_files:
            data = json.load(open(os.path.join(self.data_dir, fname), 'rb'))
            if "restaurant" in self.name:
                data = data[self.start_conversation:self.end_conversation]
            for conversation in data:
                full_text = ""
                cl_text = []
                for sentence_counter, utterance in enumerate(conversation['utterances']):
                    text = "[ {} ] {}".format(utterance['speaker'].upper(), utterance['text'])
                    full_text += text + " "
                    cl_text.append(text)
                row = f"{self.tokenizer.bos_token} {full_text} {self.tokenizer.eos_token}"
                tokenized_text = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(row))
                if len(tokenized_text) >= self.block_size:
                    num_filtered+=1
                else:
                    example = self.tokenizer.build_inputs_with_special_tokens(tokenized_text)
                    self.examples.append(example)
                    self.cl_texts.append(full_text)
                    section_ids = [0]
                    self.get_cl_embeddings(example, full_text, cl_text, gpt2_text=row)
                    self.section_ids.append(section_ids)
                    self.raw_texts.append(row)
            if len(self.examples) > 1240:
                break

        self.labels = copy.deepcopy(self.examples)
        print("num examples {}".format(len(self.examples)))
        print(f"num filtered {num_filtered}")
        print("Lengths")
        for k, v in self.lengths.items():
            print("[ {} ] {}+-{}".format(k, np.mean(v), np.std(v)/np.sqrt(len(v))))

        print("examples")
        print(self.raw_texts[0])
        print(self.raw_texts[-1])

    def get_end_points(self, tokenized_example):
        eos_idxs = []
        for tok in self.special_tokens[:2]:
            eos_idxs += [i-1 for i, x in enumerate(tokenized_example) if x == tok]
        eos_idxs += [len(tokenized_example)]
        eos_idxs.sort()
        eos_idxs = eos_idxs[1:]
        return eos_idxs

    def get_cl_embeddings(self, tokenized_example, raw_text, cl_text, gpt2_text):

        cl_embeddings = []
        eos_idxs = self.get_end_points(tokenized_example)

        assert len(eos_idxs) == len(cl_text)

        cl_input_ids, cl_attention_mask = self.cl_tokenize(cl_text, self.device)
        cl_feats = self.cl_model.forward(
            input_ids=cl_input_ids, attention_mask=cl_attention_mask) # 1, feat_size
        # Align feats to the sentence length
        last_idx = 0
        for eos_idx, feat in zip(eos_idxs, cl_feats):
            cl_embeddings += [feat] * (eos_idx - last_idx)
            last_idx = eos_idx

        assert len(cl_embeddings) == len(tokenized_example)

        if self.cl_offset:
            cl_embeddings = cl_embeddings[self.cl_offset:] + [cl_embeddings[-1]] * self.cl_offset
        self.cl_embeddings.append(cl_embeddings)


class WikihowDataset(RecipeDataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self,
                 cl_model,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int,
                 use_section_null: bool,
                 special_words:list,
                 data_dir=os.path.join(constants.PATH2RECIPENLG, 'dataset'),
                 overwrite_cache=False,
                 cache_dir: Optional[str] = None,
                 name: str = 'wikihow'
                 ):
        super(WikihowDataset, self).__init__(
            cl_model,
                 tokenizer=tokenizer,
                 file_path=file_path,
                 block_size=block_size,
                 use_section_null=use_section_null,
                 special_words=special_words,
                 data_dir=os.path.join(constants.PATH2RECIPENLG, 'dataset'),
                 overwrite_cache=False,
                 cache_dir=cache_dir,
                name=name
        )

    def _set_indices(self):
        # SEE REPO <NONSTATIONARITY> FOR INDEXES
        if self.train:
            self.start_idx, self.end_idx = 0, 700
        else:
            self.start_idx, self.end_idx = 40_000, 40_100

    def _process_dataset(self):
        self.data_name ="/nlp/scr/rewang/data/wiki_how_data.pkl"
        with open(self.data_name, 'rb') as f:
            self.all_dataset = pickle.load(f)

        num_filtered = 0

        self.processed_data = []
        split_pattern = ".  "
        doc_counter = 0
        for doc_id in tqdm(range(self.start_idx, self.end_idx)):
            doc = self.all_dataset[doc_id]
            method2steps = defaultdict(list)
            # Wikihow has k different methods
            for _, v in doc['steps'].items():
                # section = one of the how-to methods
                method2steps[v['section']].append(v)

            for method_name, steps in method2steps.items():
                doc_info = []
                sentence_counter = 0
                # Put all the document sentences together.
                gpt2_text = [self.special_words[0] + " " + doc['title'] + " . "]
                gpt2_text += [self.special_words[1] + " " +  method_name + " . "]
                for step_num, step in enumerate(steps):
                    gpt2_directions = [ f"{self.special_words[2]} {step_num} "
                                  + step['summary'][:-1] + " . "]
                    sentences = [_ + " . " for _ in step['text'].split(split_pattern)]
                    if sentences[-1].endswith(". . "):
                        sentences[-1] = sentences[-1].replace('. . ', ' . ')
                    gpt2_text += gpt2_directions + sentences

                # removing any empty sentences
                gpt2_list = [s for s in gpt2_text if s]
                gpt2_text = "".join(gpt2_list)

                row = f"{self.tokenizer.bos_token} {gpt2_text} {self.tokenizer.eos_token}"
                tokenized_text = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(row))
                if len(tokenized_text) >= self.block_size:
                    num_filtered+=1
                else:
                    example = self.tokenizer.build_inputs_with_special_tokens(tokenized_text)
                    self.examples.append(example)
                    self.cl_texts.append(gpt2_text)
                    section_ids, _ = self._determine_section_ids(tokenized_text, last_section_id=None)
                    self.lengths['per [ STEP ]'].append(
                        self.lengths['[ STEP ]'][-1]/(tokenized_text.count(50259)))
                    self.get_cl_embeddings(tokenized_example=example, raw_text=gpt2_text, cl_text=gpt2_list, gpt2_text=row)

                    self.section_ids.append(section_ids)
                    self.raw_texts.append(row)

        self.labels = copy.deepcopy(self.examples)
        print("num examples {}".format(len(self.examples)))
        print(f"num filtered {num_filtered}")
        print("Lengths")
        for k, v in self.lengths.items():
            print("[ {} ] {}+-{}".format(k, np.mean(v), np.std(v) ))

    def get_end_points(self, tokenized_example):
        eos_idxs = [i-1 for i, x in enumerate(tokenized_example) if x == self.cl_eos_id]
        eos_idxs += [len(tokenized_example)]
        return eos_idxs

    def get_cl_embeddings(self, tokenized_example, raw_text, cl_text, gpt2_text):
        split_pattern = " . "
        cl_embeddings = []
        eos_idxs = self.get_end_points(tokenized_example)
        split_sentences = gpt2_text.split(split_pattern)
        split_sentences = [_ + split_pattern for _ in split_sentences[:-1]] + [split_sentences[-1]]
        assert len(eos_idxs) == len(split_sentences)
        cl_input_ids, cl_attention_mask = self.cl_tokenize(split_sentences, self.device)
        cl_feats = self.cl_model.forward(input_ids=cl_input_ids, attention_mask=cl_attention_mask) # 1, feat_size
        # Align feats to the sentence length
        last_idx = 0
        for eos_idx, feat in zip(eos_idxs, cl_feats):
            cl_embeddings += [feat] * (eos_idx - last_idx)
            last_idx = eos_idx

        assert len(cl_embeddings) == len(tokenized_example)

        if self.cl_offset:
            cl_embeddings = cl_embeddings[self.cl_offset:] + [cl_embeddings[-1]] * self.cl_offset
        self.cl_embeddings.append(cl_embeddings)

