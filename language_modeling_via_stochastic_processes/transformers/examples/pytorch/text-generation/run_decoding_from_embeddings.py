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

import os
import wandb
import numpy as np
import torch
import tqdm

from transformers import (
    GPT2TimeLMHeadModel,
    GPT2Tokenizer,
)


from generation_metrics import GenerationMetrics
import sys
sys.path.append('../language-modeling')
from run_time_clm import (
    get_checkpoint,
    get_special_tokens,
    get_data_paths,
    get_dataset)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2TimeLMHeadModel, GPT2Tokenizer),
}

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def simulate_brownian_bridge(B_0, B_T, num_samples, sentence_lengths, dt=0.05, mu=0.0, sigma=1.0):
    """Run bridge forward pinned at B_0 and B_T"""
    if isinstance(B_0, torch.Tensor):
        B_0 = B_0.cpu().detach().numpy()
    if isinstance(B_T, torch.Tensor):
        B_T = B_T.cpu().detach().numpy()

    bridge = [B_0]
    x_t = np.copy(B_0)
    for step in range(num_samples - 2): # number of sentences
        dim = B_0.shape[-1]
        noise = np.sqrt(dt)*sigma*np.random.normal(mu, sigma, dim)
        t = step/num_samples
        x_tp1 = x_t * (1- dt/(1-t)) + (dt/(1-t))*B_T + noise
        length_idx = step % len(sentence_lengths)
        bridge += [x_tp1] * sentence_lengths[length_idx]
        x_tp1 = x_t

    length_idx = step % len(sentence_lengths)
    bridge += [B_T] * sentence_lengths[length_idx]

    return bridge

def split_text(raw_text):
    split_pattern = ". "
    split_raw_text = [_ + split_pattern for _ in raw_text.split(split_pattern)]
    split_raw_text[-1] = split_raw_text[-1].rstrip(split_pattern)
    return split_raw_text

def get_density(dataset, lm, cl_model):
    """Estimate density of last latent"""
    first_latents = []
    last_latents = []
    length = len(dataset)
    for text_i in range(length):
        first_latents.append(dataset.cl_embeddings[text_i][0].detach().cpu().numpy())
        last_latents.append(dataset.cl_embeddings[text_i][-1].detach().cpu().numpy())
    first_latents = np.array(first_latents)
    last_latents = np.array(last_latents)
    return first_latents.mean(0), first_latents.std(0), last_latents.mean(0), last_latents.std(0)

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
    parser.add_argument("--num-sentences", type=int, default=0)
    parser.add_argument("--split-sentences", type=int, default=1)
    parser.add_argument("--multiply-sentences", type=int, default=1)
    parser.add_argument("--p", type=float, default=0.99)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")

    parser.add_argument("--no_eos", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--dryrun", action="store_true", default=False, help="Text added prior to input.")
    parser.add_argument("--suppress_eos", action="store_true", default=False, help="Text added prior to input.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--dataset_name", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--fixed_prompt", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--num_intervals", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--use_dataset", action="store_true", default=False, help="Text added prior to input.")
    parser.add_argument("--project", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--encoder_filepath", type=str, required=True,default="", help="Text added prior to input.")
    parser.add_argument("--latent_dim", type=int, default=3, help="random seed for initialization")
    parser.add_argument("--use_random_embs", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--use_true_end_latent", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--label", type=str, default="", help="Text added prior to input.")

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
    args.use_section_null = 0

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

    model.transformer._config.use_contrastive_embeddings = True

    if args.suppress_eos:
        bad_words_ids = [[tokenizer.eos_token_id]]
    else:
        bad_words_ids = None

    if args.no_eos:
        min_length = 1023
    else:
        min_length= 10 # default value

    SECTION_IDS, SPECIAL_TOKENS, tokenizer = get_special_tokens(
        dataset_name=args.dataset_name, tokenizer=tokenizer)

    model.transformer.special_tokens = SPECIAL_TOKENS
    base_model = 'gpt2'
    CL_MODEL = get_checkpoint(
        dataset_name=args.dataset_name,
        latent_dim=args.latent_dim,
        sec_id=True,
        token_size=len(tokenizer),
        base_model=base_model,
        filepath=args.encoder_filepath
    )# .to(cpu_device)
    CL_MODEL.to(args.device)
    CL_MODEL.eval()
    # model.transformer.CL_MODEL = CL_MODEL

    fname = args.model_name_or_path.split('/')[-2]
    args.encoder_type = 'contrastt'

    gt_cl_tracker = GenerationMetrics(model=model, device=args.device,
                                tokenizer=tokenizer, dataset_name=args.dataset_name,
                                fname=fname+"_trueCLEmbs_" + args.method,
                                model_args=args,
                                subclass="GT")
    random_cl_tracker = GenerationMetrics(model=model, device=args.device,
                                tokenizer=tokenizer, dataset_name=args.dataset_name,
                            model_args=args,
                                fname=fname+"_randomCLEmbs_"+args.method,
                                subclass="RANDOM")
    bridge_cl_tracker = GenerationMetrics(model=model, device=args.device,
                                tokenizer=tokenizer, dataset_name=args.dataset_name,
                                fname=fname+"_bridgeCLEmbs_"+args.method,
                            model_args=args,
                                subclass="BRIDGE")

    if args.fp16:
        model.half()

    if args.dryrun:
        print("Running in dryrun mode")
        os.environ['WANDB_MODE'] = 'dryrun'

    os.environ['WANDB_CONSOLE']='wrap'
    wandb.init(project=args.project)
    wandb.config.update(args)

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    model.transformer._config.use_noisy_embeddings = False
    logger.info(args)

    prompt_text = args.prompt if args.prompt else "" # input("Model prompt >>> ")

    # Data
    assert args.dataset_name
    from language_modeling_via_stochastic_processes.src import constants
    print(f'Args: {args}')
    train_path, _, eval_path = get_data_paths(args)
    train_dataset = get_dataset(
        args=args,
        tokenizer=tokenizer,
        file_path=train_path,
        special_words=SECTION_IDS,
        cache_dir=constants.PATH2HUGGINGFACE,
        cl_model=CL_MODEL,
    )
    eval_dataset = get_dataset(
        args=args,
        tokenizer=tokenizer,
        file_path=eval_path,
        special_words=SECTION_IDS,
        cache_dir=constants.PATH2HUGGINGFACE,
        cl_model=CL_MODEL,
    )


    # Estimate dnesity for last sentence
    first_latent_mu, first_latent_std, last_latent_mu, last_latent_std = get_density(dataset=train_dataset, lm=model, cl_model=CL_MODEL)

    print("last latent mu", last_latent_mu)
    print("last latent std", last_latent_std)

    num_intervals = len(eval_dataset)

    print("Checking example embeddings: {}".format(eval_dataset.cl_embeddings[0][0]))
    print("Checking example embeddings: {}".format(eval_dataset.cl_embeddings[0][-1]))
    print("Checking example embeddings: {}".format(eval_dataset.cl_embeddings[-1][0]))
    print("Checking example embeddings: {}".format(eval_dataset.cl_embeddings[-1][-1]))

    for  num_example in tqdm.tqdm(range(num_intervals)):
        if 'wiki' in args.dataset_name:
            split_text = eval_dataset.cl_texts[num_example].split('. ')[:-1]
        if args.use_dataset or args.method == "greedy" or args.method == "beam":
            if 'wikisection' in args.dataset_name:
                k = 3
            else:
                k = 5
            example = eval_dataset.examples[num_example][:k]
            encoded_prompt = torch.tensor([example]).to(args.device)
            input_ids = encoded_prompt
            prompt_text = tokenizer.decode(example, skip_special_tokens=True)
            print("Using eval prompt: {}".format(prompt_text))
        else: # stories
            row = eval_dataset.cl_texts[num_example]
            row = row.replace('<newline>', '')
            row = row.replace(' , ', ', ')
            row = row.strip() # NOTE: remove break line
            row = ' '.join(row.split()) # remove multiple spaces
            split_pattern = " . "
            split_text = row.split(split_pattern)[:-1]
            split_text = [ _ + split_pattern for _ in split_text ]


        print('[ ACTUAL ] {}'.format(eval_dataset.raw_texts[num_example]))

        # Get all the CL feats
        true_cl_feats = torch.stack(eval_dataset.cl_embeddings[num_example])
        true_cl_feats = true_cl_feats[::args.split_sentences]
        # true_cl_feats = eval_dataset.__getitem__(_)[-2]

        LABELS = ['TRUE CL', 'BRIDGE CL (DE)',
                  # 'RANDOM CL'
                  ]
        # INTERPOLATION - BRIDGE
        print(f"DENSITY ESTIMATE: {last_latent_mu}")
        print(f"DENSITY ESTIMATE STD: {last_latent_std}")
        B_T = np.random.normal(loc=last_latent_mu, scale=last_latent_std)


        num_sentences = len(true_cl_feats) if not args.split_sentences else int(len(true_cl_feats)/float(args.split_sentences))
        num_sentences *= args.multiply_sentences

        try:
            actual_inputs = eval_dataset.examples[num_example]
        except:
            actual_inputs = eval_dataset.examples[-1]

        end = eval_dataset.get_end_points(actual_inputs)
        if min_length > 1020:
            actual_num_sentences = len(end)
            ratio = (min_length+1)/(len(actual_inputs))
            num_sentences = int(ratio*actual_num_sentences)
            # num_sentences = min_length
        else:
            ratio = 1.0

        print("Original num sentences: {}".format(len(end)))
        print("Target num sentences: {}".format(num_sentences))
        print("min length", min_length)
        end_lengths = [end[i] if i == 0 else end[i+1] - end[i] for i in range(len(end)-1)]
        end_lengths = (np.array(end_lengths)*(num_sentences/len(end)))
        end_lengths = np.ones(end_lengths.shape)
        end_lengths = end_lengths.astype(np.int)

        if 'tc' in args.encoder_filepath:
            bridge_feats = simulate_brownian_bridge(
                B_0=true_cl_feats[0], B_T=B_T, num_samples=num_sentences,
                sentence_lengths=end_lengths
            )
        else:
            bridge_feats = [true_cl_feats[0].detach().cpu().numpy()]
            for _ in range(num_sentences):
                feat = (1 - _/num_sentences) * bridge_feats[0] + _/num_sentences * B_T
                bridge_feats.append(feat)

        bridge_feats = torch.tensor(
            bridge_feats, dtype=true_cl_feats.dtype).to(args.device)
        # RANDOM
        random_feats = torch.rand(true_cl_feats.shape).to(args.device)
        feats = [true_cl_feats, bridge_feats, random_feats]

        # wandb.log({"diff_feats": (bridge_feats-true_cl_feats).sum()/bridge_feats.shape[0]})

        trackers = [gt_cl_tracker, bridge_cl_tracker,
                    # random_cl_tracker
                    ]

        for seq_i, (seq_cl_feats, tracker) in enumerate(zip(feats, trackers)):
            cl_feats = seq_cl_feats[0] # Get the first sentence feat
            prefix = args.prefix if args.prefix else args.padding_text
            encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=True, return_tensors="pt")
            encoded_prompt = encoded_prompt.to(args.device)

            if encoded_prompt.size()[-1] == 0:
                input_ids = None
            else:
                input_ids = encoded_prompt

            if 'filter' in args.dataset_name:
                length = 1024
            else:
                length = 1024 # len(eval_dataset.examples[_])

            # RESET THE CL INDEX
            model.transformer._cur_cl_idx = 0
            model.transformer._has_reset = False

            max_length = min(length + len(encoded_prompt[0]), 1024)
            if args.no_eos:
                max_length = 1024

            if args.method == "sample":
                output_sequences = model.generate(
                    input_ids=input_ids,
                    section_ids=None,
                    cl_feats=cl_feats, # .to(args.device),
                    seq_cl_feats=seq_cl_feats,
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

            # # NOTE GREEDY
            elif args.method == "greedy":
                output_sequences = model.generate(
                    input_ids=input_ids,
                    section_ids=None,
                    cl_feats=cl_feats, # .to(args.device),
                    seq_cl_feats=seq_cl_feats,
                    max_length=min(length + len(encoded_prompt[0]), 1024),
                    num_return_sequences=args.num_return_sequences,
                )

            # # NOTE Beam search
            elif args.method == "beam":
                output_sequences = model.generate(
                    input_ids=input_ids,
                    section_ids=None,
                    cl_feats=cl_feats, # .to(args.device),
                    seq_cl_feats=seq_cl_feats,
                    max_length=min(length + len(encoded_prompt[0]), 1024),
                    num_beams=5,
                    early_stopping=True,
                    num_return_sequences=args.num_return_sequences,
                    # no_repeat_ngram_size=2, # To avoid repetition
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

                print("Generated length: {}".format(len(generated_sequence)))

                # Decode text
                # text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                text = tokenizer.decode(generated_sequence, skip_special_tokens=True)

                # Remove all text after the stop token
                text = text[: text.find(args.stop_token) if args.stop_token else None]

                # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
                total_sequence = (
                    prompt_text + text[len(tokenizer.decode(encoded_prompt[0], skip_special_tokens=True)) :]
                )

                gt_raw_seq = eval_dataset.raw_texts[num_example]
                tracker.calculate(input_ids=original, raw_seq=total_sequence,
                                  cl_feats=cl_feats,
                                  gt_raw_seq=gt_raw_seq
                                  )
                generated_sequences.append(total_sequence)
                print("[ GENERATED FOR {} ]: {}".format(LABELS[seq_i], total_sequence))

    for tracker in trackers:
        tracker.print_results()
    return generated_sequences


if __name__ == "__main__":
    main()
