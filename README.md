# Language modeling via stochastic processes
[[Paper]](https://arxiv.org/pdf/2203.11370.pdf) [[Open Review]](https://openreview.net/forum?id=pMQwKL1yctf) [[Long Video]](https://www.youtube.com/watch?v=AwnoASlxeIs)

**ICLR Oral 2022**

**Rose E Wang, Esin Durmus, Noah Goodman, Tatsunori Hashimoto**

![](assets/encoder.png)

## Introduction

**Abstract:** 
Modern language models can generate high-quality short texts. However, they often meander or are incoherent when generating longer texts. 
These issues arise from the next-token-only language modeling objective.
Recent work in self-supervised learning suggests that models can learn good latent representations via contrastive learning, which can be effective for discriminative tasks.
Our work analyzes the application of contrastive representations for generative tasks, like long text generation.
We propose one approach for leveraging constrastive representations, which we call Time Control (TC).  TC first learns a contrastive representation of the target text domain, then generates text by decoding from these representations.
Compared to domain-specific methods and fine-tuning GPT2 across a variety of text domains, TC performs competitively to methods specific for learning sentence representations on discourse coherence. On long text generation settings, TC preserves the text structure both in terms of ordering (up to +15% better) and text length consistency (up to +90% better).

Contents:
- [Installation](#installation)
- [Datasets](#datasets)
- [Encoder](#encoder)
- [Decoder](#decoder) 
- [Generation](#generation)
- [Analysis](#analysis)

## Installation

0. Follow the commands in `setup.sh`
1. Make sure you are in the virtual environment: `conda activate language_modeling_via_stochastic_processes`
2. Install the decoder's version of the transformers library: 
```
cd decoder # enter the decoder repo
pip install -e . # Installing transformers locally; I modified their GPT2 module to take in our learned embeddings for decoding.
```
3. Make sure you have a [wandb](https://wandb.ai/) account!


## Datasets

**This repo contains all but two datasets (Wikihow and Recipe NLG)**. Instructions are below.

The other four datasets are already in this repo.

### Wikihow

The Wikihow dataset needs to be downloaded from [this link](https://drive.google.com/file/d/13slZcWrVUQ1RCkkxwf2QrPoTsH-vJl_3/view?usp=sharing). It's a pkl file that should go under as `path/2/repo/data/wikihow/wiki_how_data.pkl`. 

### Wikisection

The Wikisection dataset used in this paper is already included. 

It came from [this prior work](https://github.com/sebastianarnold/WikiSection) -- specifically, we used the English city wikipedia articles.

### Recipe NLG

The Recipe NLG dataset needs to be downloaded.
Download the [Recipe NLG dataset](https://recipenlg.cs.put.poznan.pl/dataset) and put the data under `encoder/data/recipe_nlg`.

### TM2

The TM2 dataset used in this paper is already included. 
It came from the [TM2 Restaurant Search dataset](https://github.com/google-research-datasets/Taskmaster/blob/master/TM-2-2020/data/restaurant-search.json).

### TicketTalk

The TicketTalk dataset used in this paper is already included.  
It can be found as the [TicketTalk dataset (all the json files)](https://github.com/google-research-datasets/Taskmaster/tree/master/TM-3-2020/data). 


## Encoder

Before running experiments, `cd encoder/code; source init_env.sh`

In `encoder/code/scripts/run_ou.py`, set the variable name `ckpt_dir` to your checkpoint directory.

The script for training the encoders (TC, VAE, Brownian, InfoNCE) can be found at `encoder/code/scripts/train_encoders.sh`.

## Encoder experiments 

Before running experiments, `cd encoder/code; source init_env.sh`

In `encoder/code/scripts/run_discourse.py` and `encoder/code/src/systems/discourse_system.py`, set the correct paths to your data directory and repo.

The script for running the discourse coherence experiments can be found at `encoder/code/scripts/discourse.sh`. 

## Decoder

For training the decoder, you'll need to be in directory `decoder/examples/pytorch/language-modeling/`.

The script for training the decoder can be found at `decoder/examples/pytorch/language-modeling/train_encoders.sh`. Make sure to change the `path2repo` variable.

You'll need to change the directories to your data directory as appropriate in `run_time_clm.py`


## Generation
![](assets/generation.png)

For generating texts, you'll need to be in directory `decoder/transformers/examples/pytorch/text-generation/`.

The script for generating text and measuring per-section length mismatches can be found at `decoder/transformers/examples/pytorch/text-generation/toy_wikisection_generation.sh`.

The script for generating long texts can be found at `decoder/transformers/examples/pytorch/text-generation/long_generation.sh`.


## Analysis

To collect all the metrics, check out `analysis/run_analysis.sh`. You can run all the evaluations with `source analysis/run_analysis.sh`.

Remember to change the wandb username and project name as what you listed in the encoder and decoder experiments.