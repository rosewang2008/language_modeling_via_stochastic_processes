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

python run_time_generation.py --model_type=gpt2 --model_name_or_path="/nlp/scr/rewang/transformers/examples/pytorch/language-modeling/wikisection_splitsection" --prompt="Boston" --length=100 --dataset_name=wikisection_splitsection --fixed_prompt

python run_time_generation.py --model_type=gpt2 --model_name_or_path="/nlp/scr/rewang/transformers/examples/pytorch/language-modeling/wikisection_splitsection" --prompt="Boston" --length=100 --dataset_name=wikisection_splitsection

python run_generation.py --model_type=gpt2 --model_name_or_path="/nlp/scr/rewang/transformers/examples/pytorch/language-modeling/checkpoints" --prompt="Boston, officially the City of Boston, is the capital and most populous city of the Commonwealth of Massachusetts in the United States and 21st most populous city in the country." --length=500

python run_generation.py --model_type=gpt2 --model_name_or_path="/nlp/scr/rewang/transformers/examples/pytorch/language-modeling/stories_runs" --prompt="[ WP ] There was a person stuck in a castle. [ RESPONSE ] " --length=500
"""


import argparse
import logging

import numpy as np
import torch

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


from transformers import (
    # Wikisection_FullDocDataset,
    StoriesDataset,
    Wikisection_BySectionDataset,
    WikisectionDataset,
)

import sys
sys.path.append("../language-modeling")
from run_time_clm import get_data_paths, get_dataset


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



def simulate_brownian_bridge(B_0, B_T, num_samples, dt=0.05, mu=0.0, sigma=1.0):
    """Run bridge forward pinned at B_0 and B_T"""
    if isinstance(B_0, torch.Tensor):
        B_0 = B_0.cpu().detach().numpy()
    if isinstance(B_T, torch.Tensor):
        B_T = B_T.cpu().detach().numpy()

    bridge = [B_0]
    x_t = np.copy(B_0)
    for step in range(num_samples - 2):
        dim = B_0.shape[-1]
        noise = np.sqrt(dt)*sigma*np.random.normal(mu, sigma, dim)
        t = step/num_samples
        x_tp1 = x_t * (1- dt/(1-t)) + (dt/(1-t))*B_T + noise
        bridge.append(x_tp1)
        x_tp1 = x_t

    bridge.append(B_T)

    assert len(bridge) == num_samples
    return bridge

def get_cl_feats(raw_text,gpt2):
    cl_input_ids, cl_attention_mask = gpt2.cl_tokenize_text(raw_text)
    cl_feats = gpt2.cl_model.forward(input_ids=cl_input_ids, attention_mask=cl_attention_mask)
    return cl_feats, {"seq_len": cl_input_ids.shape[-1]-1}

def split_text(raw_text):
    split_pattern = ". "
    split_raw_text = [_ + split_pattern for _ in raw_text.split(split_pattern)]
    split_raw_text[-1] = split_raw_text[-1].rstrip(split_pattern)
    return split_raw_text

def get_feat_as_sequence(raw_text_split, model):
    feats = [get_cl_feats(raw_text=text, gpt2=model.transformer)[0] for text in raw_text_split]
    # Making feats as long as sequence length
    feat_seq = [[feat]*get_cl_feats(raw_text=text, gpt2=model.transformer)[1]['seq_len']
                 for feat, text in zip(feats,raw_text_split)]
    return feat_seq

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

    parser.add_argument("--dataset_name", type=str, default="", help="Text added prior to input.")

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--fixed_prompt", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")

    parser.add_argument("--method", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--first_sentence", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--full_section", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--autoregressive", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

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

    if "stories" in args.model_name_or_path:
        SECTION_IDS = ['[ WP ]']
        bad_words_ids = tokenizer(SECTION_IDS)['input_ids']
    elif 'wikisection' in args.model_name_or_path:
        SECTION_IDS = ['[ ABSTRACT ]', '[ HISTORY ]', '[ GEOGRAPHY ]', '[ DEMOGRAPHICS ]']
        bad_words_ids = tokenizer(SECTION_IDS)['input_ids']
    else:
        bad_words_ids = None

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    model.transformer._config.use_noisy_embeddings = False
    logger.info(args)

    prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")

    # Data
    train_path, eval_path = get_data_paths(args)
    eval_dataset = get_dataset(
        args, tokenizer=tokenizer,
        file_path=eval_path,
        cache_dir="/nlp/scr/rewang/huggingface")

    for example_id in range(0, len(eval_dataset), 4):
        WIKI_ORDER = ['[ ABSTRACT ]', '[ HISTORY ]', '[ GEOGRAPHY ]', '[ DEMOGRAPHICS ]']
        print("[ NOTE ]: Wiki order {}".format(WIKI_ORDER))
        # if example_id > 2*4:
        if example_id > 0:
            return

        raw_text_0 = eval_dataset.examples[example_id]
        raw_text_1 = eval_dataset.examples[example_id+1]
        raw_text_2 = eval_dataset.examples[example_id+2]
        raw_text_T = eval_dataset.examples[example_id+3]
        raw_text = [eval_dataset.examples[example_id+idx] for idx in range(4)]

        # Prompt: get the first two words
        # prompt_text = " ".join(raw_text_0.split(" ")[:2])
        prompt_text = ""# "<|endoftext|>"
        print("\n")
        print("[ Prompt ]: {}".format(prompt_text))


        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

            if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}

            # encoded_prompt = tokenizer.encode(
            #     preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
            # )
            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=True, return_tensors="pt", **tokenizer_kwargs
            )
        else:
            prefix = args.prefix if args.prefix else args.padding_text
            # encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
            encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=True, return_tensors="pt")

        encoded_prompt = encoded_prompt.to(args.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt
        print("input ids: {}".format(input_ids))
        # NOTE for input ids to be None
        # input_ids = None

        RAW_TEXT = [raw_text_0, raw_text_1, raw_text_2, raw_text_T]
        for section_idx, raw_text_0 in enumerate(RAW_TEXT[:-1]):
            if args.full_section:
                # NOTE full supervision, don't need to change the start section (gen full document)
                # No interpolation
                if section_idx != 0:
                    break


            print("===================")
            print("[ START SECTION ] {}".format(WIKI_ORDER[section_idx]))
            if args.first_sentence:
                # NOTE only taking the first sentence
                print("Splitting raw text to first sentence")
                raw_text_0_ = [split_text(raw_text_0)[0]]
                raw_text_T_ = [split_text(raw_text_T)[0]]
                print("[ Raw text 0 ] {}".format(raw_text_0_))
                print("[ Raw text T ] {}".format(raw_text_T_))
                section_0_feats = [[get_cl_feats(raw_text=text, gpt2=model.transformer)[0] for text in raw_text_0_]]
                # get_feat_as_sequence(raw_text_split=raw_text_0, model=model)
                section_T_feats = [[get_cl_feats(raw_text=text, gpt2=model.transformer)[0] for text in raw_text_T_]]
                # get_feat_as_sequence(raw_text_split=raw_text_T, model=model)

                ## Building bridge
                bridge = section_0_feats

                # NOTE: force bridge to be of 4 values ~ 4 sections
                section_length = 3
                section_bridge = simulate_brownian_bridge(
                    B_0=pin_start,
                    B_T=pin_end,
                    num_samples=(2-section_idx)*section_length+2)[1:-1] # ignore first and last elements

                if section_bridge:
                    for start_idx in range(2-section_idx):
                        bridge += [section_bridge[start_idx*section_length: (start_idx+1)*section_length]]

                bridge += section_T_feats

            if args.full_section:
                raw_text_0_ = split_text(raw_text_0)
                section_0_feats = [[get_cl_feats(raw_text=text, gpt2=model.transformer)[0] for text in raw_text_0_]]
                raw_text_T_ = split_text(raw_text_T)
                section_T_feats = [[get_cl_feats(raw_text=text, gpt2=model.transformer)[0] for text in raw_text_T_]]

                bridge = section_0_feats

                if section_idx != 2:
                    # NOTE: full supervision with latents
                    for raw_text in RAW_TEXT[section_idx+1:-1]:
                        # section_feats = get_feat_as_sequence(raw_text_split=split_text(raw_text), model=model)
                        section_feats = [[get_cl_feats(raw_text=text, gpt2=model.transformer)[0]
                                          for text in split_text(raw_text)]]
                        bridge += section_feats

                bridge += section_T_feats

            # NOTE bridge = [[section 1 feats], [section 2 feats], ...]
            # list of lists
            full_split_raw_text = [raw_text_0_, split_text(raw_text_1), split_text(raw_text_2), raw_text_T_]

            for t, cl_feat in enumerate(bridge):
                section_idx_true = section_idx + t
                input_ids = None # NOTE new section
                prompt_text = ""
                encoded_prompt = tokenizer.encode("", add_special_tokens=True, return_tensors="pt").to(args.device)

                for feat_num, feat in enumerate(cl_feat):
                    # print("[ ACTUAL @ {}% ]: {}".format(WIKI_ORDER[section_idx_true], RAW_TEXT[section_idx_true]))
                    print("[ ACTUAL @ {}% ]: {}".format(
                        WIKI_ORDER[section_idx_true], full_split_raw_text[section_idx_true][feat_num]))

                    if isinstance(feat, np.ndarray):
                        feat = torch.Tensor(feat).to(args.device)

                    if args.autoregressive:
                        if input_ids is None:
                            old_feat = feat
                            # feat = None
                        else:
                            new_feat = torch.clone(feat)
                            feat = old_feat
                            old_feat = new_feat


                    print("cl feat", feat)

                    if args.method == "sample":
                        output_sequences = model.generate(
                            input_ids=input_ids,
                            section_ids=None,
                            raw_text=None, # full_example,
                            # cl_feats=torch.cat(cl_feat).to(args.device),
                            # cl_feats=torch.Tensor(feat)# .to(args.device),
                            cl_feats=feat, # .to(args.device),
                            max_length=args.length + len(encoded_prompt[0]),
                            temperature=args.temperature,
                            top_k=args.k,
                            top_p=args.p,
                            repetition_penalty=args.repetition_penalty,
                            do_sample=True,
                            num_return_sequences=args.num_return_sequences,
                            bad_words_ids=bad_words_ids,
                        )

                    # # NOTE GREEDY
                    elif args.method == "greedy":
                        output_sequences = model.generate(
                            input_ids=input_ids,
                            # cl_feats=torch.Tensor(cl_feat).to(args.device),
                            cl_feats=feat,
                            max_length=args.length + len(encoded_prompt[0]),
                            num_return_sequences=args.num_return_sequences,
                            bad_words_ids=bad_words_ids,
                        )

                    # # NOTE Beam search
                    elif args.method == "beam":
                        output_sequences = model.generate(
                            input_ids=input_ids,
                            # cl_feats=torch.Tensor(cl_feat).to(args.device),
                            cl_feats=feat,
                            max_length=args.length + len(encoded_prompt[0]),
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
                        if args.autoregressive:
                            if input_ids is None:
                                input_ids = generated_sequence.unsqueeze(0)
                            else:
                                input_ids = generated_sequence[input_ids.size()[-1]:].unsqueeze(0)


                        generated_sequence = generated_sequence.tolist()

                        # Decode text
                        # text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                        text = tokenizer.decode(generated_sequence, skip_special_tokens=True)

                        # Remove all text after the stop token
                        text = text[: text.find(args.stop_token) if args.stop_token else None]

                        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
                        # total_sequence = (
                        #     prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
                        # )
                        total_sequence = (
                            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], skip_special_tokens=True)) :]
                        )

                        if args.autoregressive:
                            prompt_text = text[len(tokenizer.decode(encoded_prompt[0], skip_special_tokens=True)) :]
                            encoded_prompt = input_ids

                        generated_sequences.append(total_sequence)
                        print("[ GENERATED @ {} ]: {}".format(WIKI_ORDER[section_idx_true], total_sequence))


    return generated_sequences


if __name__ == "__main__":
    main()
