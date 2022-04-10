#!/bin/bash
nseed=10

domains=("recipe" "roc_stories" "tm2" "tickettalk" "wikihow" "wikisection")
path2repo="path/2/repo"

for domain in ${domains[@]}; do
    for seed in $(seq 1 1 $nseed); do
        for latent_dim in {8,16,32}; do
            python run_decoding_from_embeddings.py --model_type=gpt2 --model_name_or_path=${path2repo}/language_modeling_via_stochastic_processes/transformers/examples/pytorch/language-modeling/LM_${domain}_${latent_dim}/ --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=${domain} --encoder_filepath=${path2repo}/language_modeling_via_stochastic_processes/models/${domain}/tc${latent_dim}/epoch=99-step=75299.ckpt --latent_dim=${latent_dim} --project=LM_${domain} --no_eos --label=LM_${domain}_${latent_dim} --seed=${seed}
        done
    done
done
