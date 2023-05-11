#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:

https://huggingface.co/models?filter=causal-lm


# Stories
python run_clm.py --model_name_or_path gpt2  --dataset_name stories --do_train --do_eval --output_dir story_runs/ --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=10 --load_best_model_at_end=True


python run_clm.py --model_name_or_path gpt2  --dataset_name wikisection_section --do_train --do_eval --output_dir test/ --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=10 --load_best_model_at_end=True --overwrite_output_dir

python run_clm.py --model_name_or_path gpt2  --dataset_name wikisection --do_train --do_eval --output_dir test/ --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=10 --load_best_model_at_end=True --overwrite_output_dir

"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import wandb
import torch

import getpass
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    Trainer_Time,
    TrainingArguments,
    default_data_collator,
    set_seed,
    PreTrainedTokenizer,
    DataCollatorForWikiSectionSplitBySection,
    # DataCollatorForWikiSectionFullDoc,
    Wikisection_BySectionDataset,
    # Wikisection_FullDocDataset,
    WikisectionDataset,
    WikisectionFilterLargeDataset,
    StoriesDataset,
    RecipeDataset,
    TaskmasterDataset,
    WikihowDataset,
    StoriesDataset_SplitSentence,
    GPT2TimeLMHeadModel,
)

from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.8.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default="/nlp/scr/rewang/huggingface/",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    dryrun: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    debug_ids: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_contrastive_embeddings: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_section_ids: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_noisy_embeddings: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    embedding_type: str = field(
        default="entireSection",
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    cl_fpath: str = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    latent_dim: int = field(
        default=3,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    use_normalized_loss_cl: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    project: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    tag: Optional[str] = field(
        default="", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    use_bos: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_section_null: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    label:str = field(
        default="",
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def get_data_paths(data_args: DataTrainingArguments):
    if data_args.dataset_name == "wikisection_filter":
        train_path = "/nlp/scr/rewang/public_language_modeling_via_stochastic_processes/encoder/code/scratch/wikisection_withSections.train.txt"
        val_path = "/nlp/scr/rewang/public_language_modeling_via_stochastic_processes/encoder/code/scratch/wikisection_withSections.val.txt"
        test_path = "/nlp/scr/rewang/public_language_modeling_via_stochastic_processes/encoder/code/scratch/wikisection_withSections.test.txt"
    elif "stories" in data_args.dataset_name:
        train_path = "/nlp/scr/rewang/ilm/data/char_masks/roc_stories/train.pkl"
        val_path = "/nlp/scr/rewang/ilm/data/char_masks/roc_stories/valid.pkl"
        test_path = "/nlp/scr/rewang/ilm/data/char_masks/roc_stories/test.pkl"
    elif 'recipe' in data_args.dataset_name:
        train_path = "/nlp/scr/rewang/public_language_modeling_via_stochastic_processes/encoder/code/scratch/wikisection_withSections.train.txt"
        val_path = "/nlp/scr/rewang/public_language_modeling_via_stochastic_processes/encoder/code/scratch/wikisection_withSections.val.txt"
        test_path = "/nlp/scr/rewang/public_language_modeling_via_stochastic_processes/encoder/code/scratch/wikisection_withSections.test.txt"
    elif 'wikihow' in data_args.dataset_name:
        train_path = "/nlp/scr/rewang/public_language_modeling_via_stochastic_processes/encoder/code/scratch/wikisection_withSections.train.txt"
        val_path = "/nlp/scr/rewang/public_language_modeling_via_stochastic_processes/encoder/code/scratch/wikisection_withSections.val.txt"
        test_path = "/nlp/scr/rewang/public_language_modeling_via_stochastic_processes/encoder/code/scratch/wikisection_withSections.test.txt"
    elif 'taskmaster' in data_args.dataset_name:
        train_path = "/nlp/scr/rewang/public_language_modeling_via_stochastic_processes/encoder/code/scratch/wikisection_withSections.train.txt"
        val_path = "/nlp/scr/rewang/public_language_modeling_via_stochastic_processes/encoder/code/scratch/wikisection_withSections.val.txt"
        test_path = "/nlp/scr/rewang/public_language_modeling_via_stochastic_processes/encoder/code/scratch/wikisection_withSections.test.txt"
    else:
        raise ValueError()

    return train_path, val_path, test_path

def get_dataset(
    cl_model,
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    file_path: str,
    special_words: list,
    cache_dir: Optional[str] = None,
    training_args: TrainingArguments = None,
):
    if "wikisection" in args.dataset_name:
        dataset = WikisectionFilterLargeDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            use_section_null=args.use_section_null,
            special_words=special_words,
            block_size=args.block_size,
            cl_model=cl_model
        )
    elif args.dataset_name == 'roc_stories':
        dataset = StoriesDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size,
            special_words=special_words,
            cl_model=cl_model
        )
    elif args.dataset_name == 'wikihow':
        dataset = WikihowDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            use_section_null=args.use_section_null,
            special_words=special_words,
            block_size=args.block_size,
            cl_model=cl_model
        )
    elif args.dataset_name == 'recipe':
        dataset = RecipeDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            use_section_null=args.use_section_null,
            special_words=special_words,
            block_size=args.block_size,
            cl_model=cl_model
        )
    elif 'taskmaster' in args.dataset_name:
        dataset = TaskmasterDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            use_section_null=args.use_section_null,
            special_words=special_words,
            block_size=args.block_size,
            cl_model=cl_model,
            name=args.dataset_name
        )
    return dataset

def get_data_collator(
        args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer):
    # if args.dataset_name == "wikisection_splitsection":
    #     data_collator = DataCollatorForWikiSectionSplitBySection(
    #         tokenizer=tokenizer)
    # elif args.dataset_name == "wikisection":
    #     data_collator = DataCollatorForWikiSectionSplitBySection(
    #         tokenizer=tokenizer)
    # elif args.dataset_name == "wikisection_filter":
    #     data_collator = DataCollatorForWikiSectionSplitBySection(
    #         tokenizer=tokenizer)
    # elif args.dataset_name == "stories_split":
    #     data_collator = DataCollatorForWikiSectionSplitBySection(
    #         tokenizer=tokenizer)
    # elif args.dataset_name == "wikisection_fulldoc":
    #     data_collator = DataCollatorForWikiSectionFullDoc(
    #         tokenizer=tokenizer)


    # current code
    # if "wikisection" in args.dataset_name:
    #     data_collator = DataCollatorForWikiSectionSplitBySection(
    #         tokenizer=tokenizer)
    # elif args.dataset_name == "stories":
    #     data_collator = DataCollatorForWikiSectionSplitBySection(
    #         tokenizer=tokenizer)
    # else:
    #     raise ValueError()
    data_collator = DataCollatorForWikiSectionSplitBySection(
        tokenizer=tokenizer)
    return data_collator


HIDDEN_DIM = 128

def load_cl_model(filepath, latent_dim,base_model, use_section_ids,
                  token_size):
    import sys
    sys.path.append('/nlp/scr/rewang/public_language_modeling_via_stochastic_processes/encoder/code/')
    from src.models import language
    if filepath is None:
        model = language.GPT2OUEncoder(
             hidden_dim=HIDDEN_DIM,
             latent_dim=latent_dim,
             finetune_gpt2=False)
    elif 'bert' in filepath or 'bert' in base_model.lower():
        model = language.BERTOUEncoder(
             hidden_dim=HIDDEN_DIM,
             latent_dim=latent_dim,
             finetune=False)
    elif "gpt2" in base_model.lower():
        model = language.GPT2OUEncoder(
             hidden_dim=HIDDEN_DIM,
             latent_dim=latent_dim,
             finetune_gpt2=False)
    if use_section_ids:
        model.model.resize_token_embeddings(token_size)

    if filepath is not None:
        state_dict = torch.load(filepath)
        new_dict = {}
        for k, v in state_dict['state_dict'].items():
            if 'infonce' in filepath:
                if 'g_ar' in k:
                    continue
                if 'W_k' in k:
                    continue

            if 'vae' in filepath:
                if k.startswith('model.fc'):
                    continue

            if k.startswith('model.'):
                new_dict[k[len('model.'):]] = v
            elif k.startswith('time_model.'):
                continue
            else:
                new_dict[k] = v

        model.load_state_dict(new_dict)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model

def get_checkpoint(dataset_name, latent_dim, base_model="gpt2",
                   sec_id=False, token_size=None,
                   filepath=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if base_model.lower() == "bert":
    #     if sec_id:
    #         ckpt_dict = BERT_SECID_TOY_WIKISECTION_CKPTS
    #     else:
    #         ckpt_dict = BERT_TOY_WIKISECTION_CKPTS
    # else:
    #     if dataset_name == "long_wikisection":
    #         ckpt_dict = LONG_WIKISECTION_CKPTS
    #     elif dataset_name == "wikisection_filter" or 'wikisection' in dataset_name:
    #         print("in the right branch")
    #         ckpt_dict = WIKISECTION_CKPTS
    #     elif 'stories' in dataset_name:
    #         ckpt_dict = STORY_CKPTS
    #     elif 'recipe' in dataset_name:
    #         ckpt_dict = RECIPE_CKPTS
    #     elif 'wikihow' in dataset_name:
    #         ckpt_dict = WIKIHOW_CKPTS

    # fpath = (filepath if filepath is not None
    #          else ckpt_dict['latent{}_norm{}'.format(latent_dim, with_norm)])
    model = load_cl_model(filepath,
                          latent_dim,
                          base_model,
                          use_section_ids=sec_id,
                          token_size=token_size
                          )
    model.to(device)
    model = model.eval()
    return model


def get_special_tokens(dataset_name, tokenizer, add_tokens=True):
    # if dataset_name == "stories":
    #     SECTION_IDS = ['[ WP ]', '[ RESPONSE ]']
    if "wikisection_filter" == dataset_name:
        SECTION_IDS = ['[ ABSTRACT ]', '[ HISTORY ]', '[ GEOGRAPHY ]', '[ DEMOGRAPHICS ]']
    if "long_wikisection" == dataset_name:
        SECTION_IDS = [
            '[ ABSTRACT ]',
            '[ HISTORY ]',
            '[ GEOGRAPHY ]',
            '[ GEOGRAPHY | CLIMATE ]',
            '[ CLIMATE ]',
            '[ DEMOGRAPHICS ]',
            '[ ECONOMY ]',
            '[ TRANSPORTATION ]',
            '[ GOVERNMENT ]',
            '[ SPORTS ]',
        ]
    if 'recipe' in dataset_name:
        SECTION_IDS = [
            '[ TITLE ]',
            '[ INGREDIENTS ]',
            '[ DIRECTIONS ]'
        ]
    if 'taskmaster' in dataset_name:
        SECTION_IDS = [
            '[ USER ]',
            '[ ASSISTANT ]',
        ]
    if 'wikihow' in dataset_name:
        SECTION_IDS = [
            '[ TITLE ]',
            '[ METHOD ]',
            '[ STEP ]'
        ]
    if dataset_name == 'roc_stories':
        SECTION_IDS = []
    SECTION_IDS += [' . ']
    if add_tokens:
        # NOTE loading previous tokenizer sometimes already includes the new tokens
        eos = tokenizer(' . ')['input_ids']
        print("Old tokenizer size: ", len(tokenizer))
        if len(eos) == 1 and eos[0] == 50256 + len(SECTION_IDS):
            print("Not adding because it's already contained")
            pass # don't add cause it's already contained
        else:
            print("Adding tokens, ", SECTION_IDS)
            tokenizer.add_tokens(SECTION_IDS)
        print("New tokenizer size: ", len(tokenizer))
    SPECIAL_TOKENS = [_[0] for _ in tokenizer(SECTION_IDS)['input_ids']]
    return SECTION_IDS, SPECIAL_TOKENS, tokenizer

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    if model_args.dryrun:
        print("Running in dryrun mode")
        os.environ['WANDB_MODE'] = 'dryrun'

    os.environ['WANDB_CONSOLE']='wrap'

    wandb_run = wandb.init(
        project=data_args.project,
        entity=getpass.getuser(),
        name=training_args.output_dir,
        config=training_args,
    )

    if data_args.tag:
        wandb_run.tags = wandb_run.tags + (data_args.tag,)


    if data_args.project is not None:
        os.environ['WANDB_PROJECT'] = data_args.project

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)sfilter_ -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    #     # Downloading and loading a dataset from the hub.
    #     if data_args.dataset_name == "wikisection":
    #         datasets = load_dataset('./wikisection_lm.py')
    #     elif data_args.dataset_name == "wikisection_section":
    #         datasets = load_dataset('./wikisection_lm_sections.py')
    #     elif data_args.dataset_name == "stories":
    #         datasets = load_dataset('./stories_lm.py')
    #     else:
    #         datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    #         if "validation" not in datasets.keys():
    #             datasets["validation"] = load_dataset(
    #                 data_args.dataset_name,
    #                 data_args.dataset_config_name,
    #                 split=f"train[:{data_args.validation_split_percentage}%]",
    #                 cache_dir=model_args.cache_dir,
    #             )
    #             datasets["train"] = load_dataset(
    #                 data_args.dataset_name,
    #                 data_args.dataset_config_name,
    #                 split=f"train[{data_args.validation_split_percentage}%:]",
    #                 cache_dir=model_args.cache_dir,
    #             )
    # else:
    #     data_files = {}
    #     if data_args.train_file is not None:
    #         data_files["train"] = data_args.train_file
    #     if data_args.validation_file is not None:
    #         data_files["validation"] = data_args.validation_file
    #     extension = (
    #         data_args.train_file.split(".")[-1]
    #         if data_args.train_file is not None
    #         else data_args.validation_file.split(".")[-1]
    #     )
    #     if extension == "txt":
    #         extension = "text"
    #     datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)


    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)


    config.use_contrastive_embeddings = model_args.use_contrastive_embeddings
    config.debug_ids = model_args.debug_ids
    config.embedding_type = model_args.embedding_type
    config.use_section_ids = model_args.use_section_ids
    config.use_section_null = data_args.use_section_null
    config.use_noisy_embeddings = model_args.use_noisy_embeddings
    config.dataset_name = data_args.dataset_name
    config.cl_latent_dim = model_args.latent_dim

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if True:
        tokenizer = AutoTokenizer.from_pretrained('gpt2', **tokenizer_kwargs)
        tokenizer.pad_token = tokenizer.eos_token
    # if model_args.tokenizer_name:
    #     tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    # elif model_args.model_name_or_path:
    #     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    SECTION_IDS, SPECIAL_TOKENS, tokenizer = get_special_tokens(
        dataset_name=data_args.dataset_name, tokenizer=tokenizer)
    # -1 because of the added " . "
    config.max_num_sections = len(SECTION_IDS) - 1

    if model_args.model_name_or_path:
        model = GPT2TimeLMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_args.model_name_or_path,
        #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #     config=config,
        #     cache_dir=model_args.cache_dir,
        #     revision=model_args.model_revision,
        #     use_auth_token=True if model_args.use_auth_token else None,
        # )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))
    model.transformer.special_tokens = SPECIAL_TOKENS
    print("Resized model to {}".format(len(tokenizer)))
    print("Added special tokens, ", SPECIAL_TOKENS)
    print("Added special tokens, ", SECTION_IDS)

    # Getting checkpoint dict:
    cpu_device = torch.device('cpu')
    # base_model = 'bert' if 'bert' in model_args.cl_fpath else 'gpt2'
    base_model = 'gpt2'
    # token_size = 29000 if base_model=='bert' else 50261
    CL_MODEL = get_checkpoint(
        dataset_name=data_args.dataset_name,
        latent_dim=model_args.latent_dim,
        # with_norm=model_args.use_normalized_loss_cl,
        sec_id=True, # Aug 14: from today, always use section ids
        token_size= len(tokenizer),
        base_model=base_model,
        filepath=model_args.cl_fpath
    )# .to(cpu_device)

    # # Preprocessing the datasets.
    # # First we tokenize all the texts.
    # if training_args.do_train:
    #     column_names = datasets["train"].column_names
    # else:
    #     column_names = datasets["validation"].column_names
    # text_column_name = "text" if "text" in column_names else column_names[0]

    # # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    # tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    # def tokenize_function(examples):
    #     with CaptureLogger(tok_logger) as cl:
    #         output = tokenizer(examples[text_column_name])
    #     # clm input could be much much longer than block_size
    #     if "Token indices sequence length is longer than the" in cl.out:
    #         tok_logger.warning(
    #             "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
    #         )
    #     return output

    # tokenized_datasets = datasets.map(
    #     tokenize_function,
    #     batched=True,
    #     num_proc=data_args.preprocessing_num_workers,
    #     remove_columns=column_names,
    #     load_from_cache_file=not data_args.overwrite_cache,
    #     desc="Running tokenizer on dataset",
    # )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    data_args.block_size = block_size
    ### Data
    train_path, val_path, eval_path = get_data_paths(data_args)

    train_dataset = get_dataset(
        args=data_args, tokenizer=tokenizer,
        file_path=train_path,
        cache_dir=model_args.cache_dir,
        special_words=SECTION_IDS,
        training_args=training_args,
        cl_model=CL_MODEL
    )
    # val_dataset = get_dataset(
    #     args=data_args, tokenizer=tokenizer,
    #     file_path=val_path,
    #     special_words=SECTION_IDS,
    #     cache_dir=model_args.cache_dir,
    #     training_args=training_args,
    #     cl_model=CL_MODEL
    # )
    eval_dataset = get_dataset(
        args=data_args, tokenizer=tokenizer,
        file_path=eval_path,
        special_words=SECTION_IDS,
        cache_dir=model_args.cache_dir,
        training_args=training_args,
        cl_model=CL_MODEL
    )

    data_collator = get_data_collator(
        args=data_args,
        tokenizer=tokenizer)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    # lm_datasets = tokenized_datasets.map(
    #     group_texts,
    #     batched=True,
    #     num_proc=data_args.preprocessing_num_workers,
    #     load_from_cache_file=not data_args.overwrite_cache,
    #     desc=f"Grouping texts in chunks of {block_size}",
    # )


    # if training_args.do_train:
    #     # if "train" not in tokenized_datasets:
    #     #     raise ValueError("--do_train requires a train dataset")
    #     # train_dataset = lm_datasets["train"]
    #     if data_args.max_train_samples is not None:
    #         train_dataset = train_dataset.select(range(data_args.max_train_samples))

    # if training_args.do_eval:
    #     # if "validation" not in tokenized_datasets:
    #     #     raise ValueError("--do_eval requires a validation dataset")
    #     # eval_dataset = lm_datasets["validation"]
    #     if data_args.max_eval_samples is not None:
    #         eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Initialize our Trainer
    if True:
        trainer = Trainer_Time(
            model=model,
            special_tokens=SPECIAL_TOKENS,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            # data_collator=default_data_collator,
            data_collator=data_collator
        )

    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=default_data_collator,
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            perplexity = math.exp(metrics["train_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
            wandb.log({"ppl": perplexity})
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
