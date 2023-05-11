import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import logging
import math
import os
import sys, json, torch
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
from rnn_lm import RNNLM

os.environ['HF_HOME'] = "/u/scr/xlisali/struct-inf/cache_data"
os.environ['TRANSFORMERS_CACHE'] = '/u/scr/xlisali/struct-inf/cache_data'

def main():
    model_name = 'trained_models/retrain_lm/rnn_from_data_lr=0.001_e=10'
    total_num = 1000000 #200 #1000K
    config = AutoConfig.from_pretrained(model_name, cache_dir=os.environ['TRANSFORMERS_CACHE'])

    config.rnn_dropout = 0.3
    config.embed_dim = 400
    config.rnn_type = 'LSTM'
    config.num_layers = 4
    config.hidden_dim = 400


    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.environ['TRANSFORMERS_CACHE'])
    tokenizer.pad_token = tokenizer.eos_token
    model = RNNLM.from_pretrained(model_name, config=config, cache_dir=os.environ['TRANSFORMERS_CACHE']).cuda()
    model.eval()
    # print([x for x in list(model.named_parameters())[:2]])
    # fout = open(out_path, 'w')
    # print(model)
    bsz = 1000
    model_name = 'rnn'
    out_path = f'gpt2_samples/{model_name}_samples_{total_num}.json'

    full_lst = []
    full_lst_idx = []
    for i in tqdm(range(total_num // bsz)):
        outputs = model.generate(max_length=150, do_sample=True, num_return_sequences=bsz, top_k=len(tokenizer))
        # print(outputs)
        # print(outputs.shape)
        out_sents = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(i * len(out_sents))
        full_lst += out_sents
        full_lst_idx += outputs.cpu().tolist()
        # print(out_sents[:2])
        # out_sents_ = [tokenizer.bos_token + line for line in out_sents]
        # out_sents_ = [line for line in out_sents]
        # print(out_sents_[:2])

        # temp_test = tokenizer(out_sents_, padding=True,
        #           truncation=True,
        #           max_length=150, )['input_ids']
        # temp_test =  torch.tensor(temp_test).long()
        # print(temp_test.shape)
        # print(temp_test)
        # torch.all(torch.eq(temp_test.cuda(), outputs))

    print(len(full_lst))
    print(len(full_lst_idx))
    # print(full_lst_idx[:10])
    with open(out_path, 'w') as f:
        json.dump({'data': full_lst, "idx":full_lst_idx}, f)

if __name__ == '__main__':
    set_seed(101)
    with torch.no_grad():
        main()
