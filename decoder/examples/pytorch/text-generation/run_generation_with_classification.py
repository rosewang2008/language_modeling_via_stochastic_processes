#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)

"""


import argparse
import logging

import wandb
import os
import numpy as np
import torch
import tqdm

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2TimeLMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

from generation_metrics import GenerationMetrics
from run_bridge_generation import get_data_paths, get_dataset
from run_cl_transition_fulldoc_generation_da import get_checkpoint
from run_time_clm import get_special_tokens


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    # "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2": (GPT2TimeLMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--classification_model",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--cl_fpath", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--latent_dim", type=int, default=0)
    parser.add_argument("--project", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--no_eos", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--dryrun", action="store_true", default=False, help="Text added prior to input.")
    parser.add_argument("--suppress_eos", action="store_true", default=False, help="Text added prior to input.")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--use_dataset", action="store_true", default=False, help="Text added prior to input.")
    parser.add_argument("--use_bos", action="store_true", default=True, help="Text added prior to input.")
    parser.add_argument("--dataset_name", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--tag", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--use_section_null", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--method", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--label", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--num_intervals", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()
    # set classifier
    if "long" in args.dataset_name:
        args.classification_model = "/nlp/scr/rewang/transformers/examples/pytorch/text-classification/long_c1"
    elif "wiki" in args.dataset_name:
        args.classification_model = "/nlp/scr/rewang/transformers/examples/pytorch/text-classification/c1_"
    elif "stories" in args.dataset_name:
        args.classification_model = "/nlp/scr/rewang/transformers/examples/pytorch/text-classification/long_c1"
    elif "recipe" in args.dataset_name:
        args.classification_model = "/nlp/scr/rewang/transformers/examples/pytorch/text-classification/recipe_c1_short"
    elif "wikihow" in args.dataset_name:
        args.classification_model = "/nlp/scr/rewang/transformers/examples/pytorch/text-classification/wikihow_mini_c1"
    elif "taskmaster" in args.dataset_name:
        # TODO need to train
        args.classification_model = "/nlp/scr/rewang/transformers/examples/pytorch/text-classification/wikihow_mini_c1"
    else:
        raise ValueError()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    if args.project is not None:
        os.environ['WANDB_PROJECT'] = args.project

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    if args.dryrun:
        print("Running in dryrun mode")
        os.environ['WANDB_MODE'] = 'dryrun'

    os.environ['WANDB_CONSOLE']='wrap'
    wandb_run = wandb.init(
        project=args.project)
    wandb.config.update(args)
    if args.tag:
        wandb_run.tags = wandb_run.tags + (args.tag,)

    fname = args.model_name_or_path.split('/')[-2] + "_" + args.method
    tracker = GenerationMetrics(model=model, device=args.device,
                                tokenizer=tokenizer, dataset_name=args.dataset_name,
                                model_args=args,
                                fname=fname)

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    prompt_text = args.prompt if args.prompt else "" # input("Model prompt >>> ")

    # Different models need different input formatting and/or extra arguments
    requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
    if requires_preprocessing:
        prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
        preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

        if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
            tokenizer_kwargs = {"add_space_before_punct_symbol": True}
        else:
            tokenizer_kwargs = {}

        encoded_prompt = tokenizer.encode(
            preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
        )
    else:
        prefix = args.prefix if args.prefix else args.padding_text
        encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(args.device)

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt

    SECTION_IDS, SPECIAL_TOKENS, tokenizer = get_special_tokens(
        dataset_name=args.dataset_name, tokenizer=tokenizer)
    base_model = 'gpt2'
    try:
        CL_MODEL = get_checkpoint(
            dataset_name=args.dataset_name,
            latent_dim=args.latent_dim,
            sec_id=True, # Aug 14: from today, always use section ids
            token_size= len(tokenizer)-1,
            base_model=base_model,
            filepath=args.cl_fpath
        )# .to(cpu_device)
    except:
        CL_MODEL = get_checkpoint(
            dataset_name=args.dataset_name,
            latent_dim=args.latent_dim,
            sec_id=True, # Aug 14: from today, always use section ids
            token_size= len(tokenizer),
            base_model=base_model,
            filepath=args.cl_fpath
        )# .to(cpu_device)
    CL_MODEL.to(args.device)
    CL_MODEL.eval()

    train_path, val_path, eval_path = get_data_paths(args)
    eval_dataset = get_dataset(
        args=args, tokenizer=tokenizer,
        file_path=eval_path,
        special_words=SECTION_IDS,
        cache_dir="/nlp/scr/rewang/huggingface",
        cl_model=CL_MODEL
    )

    if "stories" in args.model_name_or_path:
        SECTION_IDS = ['[ WP ]']
        bad_words_ids = tokenizer(SECTION_IDS)['input_ids']
    else:
        bad_words_ids = None

    if args.suppress_eos:
        bad_words_ids = [[tokenizer.eos_token_id]]

    if args.no_eos:
        max_length=1024
        min_length = 1023
    else:
        max_length=1024
        min_length= 10 # default value


    # # num_intervals = args.num_return_sequences // 5
    # if args.method != "sample":
    #     num_intervals = len(eval_dataset)
    # else:
    #     num_intervals = args.num_intervals

    num_intervals = len(eval_dataset)

    for _ in tqdm.tqdm(range(num_intervals)):
        # First check whether we should use input ids from eval dataset
        if args.use_dataset or args.method == "greedy" or args.method == "beam":
            if 'wikisection' in args.dataset_name:
                k = 3
            else:
                k = 5
            example = eval_dataset.examples[_][:k]
            encoded_prompt = torch.tensor([example]).to(args.device)
            input_ids = encoded_prompt
            prompt_text = tokenizer.decode(example, skip_special_tokens=True)
            print("Using eval prompt: {}".format(prompt_text))
            assert args.num_return_sequences == 1

        # if "stories" in args.dataset_name:
        #     row = eval_dataset.cl_texts[_]
        #     row = row.replace('<newline>', '')
        #     row = row.replace(' , ', ', ')
        #     row = row.strip() # NOTE: remove break line
        #     row = ' '.join(row.split()) # remove multiple spaces
        #     split_pattern = " . "
        #     split_text = row.split(split_pattern)[:-1]
        #     split_text = [ _ + split_pattern for _ in split_text ]
        #     story_prompt = row[:row.index('[ RESPONSE ]')]
        #     prompt_text = f"<|endoftext|> {story_prompt}"
        #     prefix = args.prefix if args.prefix else args.padding_text
        #     encoded_prompt = tokenizer.encode(
        #         prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
        #     encoded_prompt = encoded_prompt.to(args.device)
        #     if encoded_prompt.size()[-1] == 0:
        #         input_ids = None
        #     else:
        #         input_ids = encoded_prompt

        if 'filter' in args.dataset_name:
            length = 1024
        elif 'beam' in args.method or 'greedy' in args.method:
            length = 1024
        else:
            length = 1024
            # length = len(eval_dataset.examples[_])

        # max_length = min(length + len(encoded_prompt[0]), 1024)

        if args.method == "sample":
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
                bad_words_ids=bad_words_ids,
                min_length=min_length-50
            )
            # from torch import nn
            # n_prob = 0.0
            # import pdb; pdb.set_trace()
            # scores = output_sequences.scores
            # output_sequences = output_sequences[0]
            # for score, token in zip(scores, output_sequences[0]):
            #     p = nn.functional.softmax(score)[0]
            #     n_prob += -torch.log(p)[token]
            # ppl = torch.exp(n_prob)
            # print(ppl)
            # loss = model(output_sequences[0], lm_labels=output_sequences[0])
            # ppl = torch.exp(loss.loss)
            # import pdb; pdb.set_trace()

        # # NOTE GREEDY
        elif args.method == "greedy":
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=min(length + len(encoded_prompt[0]), 1024),
                num_return_sequences=args.num_return_sequences,
                bad_words_ids=bad_words_ids,
            )

        # # NOTE Beam search
        elif args.method == "beam":
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=min(length + len(encoded_prompt[0]), 1024),
                num_beams=5,
                early_stopping=True,
                num_return_sequences=args.num_return_sequences,
                # no_repeat_ngram_size=2, # To avoid repetition
                bad_words_ids=bad_words_ids,
            )

        else:
            raise ValueError("need to specify --method")

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            # print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")

            original = torch.clone(generated_sequence)
            generated_sequence = generated_sequence.tolist()

            # Decode text
            # text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            text = tokenizer.decode(generated_sequence, skip_special_tokens=True)

            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                # prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
                prompt_text + text[len(tokenizer.decode(encoded_prompt[0], skip_special_tokens=True)) :]
            )
            try:
                gt_raw_seq = eval_dataset.raw_texts[_]
            except:
                gt_raw_seq = eval_dataset.raw_texts[-1]

            tracker.calculate(input_ids=original, raw_seq=total_sequence,
                                  gt_raw_seq=gt_raw_seq
                              )

            generated_sequences.append(total_sequence)
            print('[ ACTUAL ] {}'.format(tokenizer.decode(eval_dataset.examples[_])))
            print('[ GENERATED, length {} ] {}'.format(len(generated_sequence), total_sequence))
            # print(total_sequence)

    tracker.print_results()
    return generated_sequences


if __name__ == "__main__":
    main()
