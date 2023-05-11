#!/bin/bash

##### This is the script for running the TOY WIKISECTION EXPERIMENTS #####
##### Specifically, this is looking at GENERATION QUALITY #####

# NUM_SEEDS: 5
# METHODS: GPT2 finetuning, section null, section ID, CL embeddings (3-512), random embeddings (3-512)
nseed=3
ndecoderseed=3
project="ICLR2022_v3_text_evaluation"
env_name="language_modeling_via_stochastic_processes"
path2repo='/nlp/scr/rewang/public_language_modeling_via_stochastic_processes'
tag="randDecoder"
priority='low'


for decoder_seed in $(seq 1 1 $ndecoderseed); do
    ##### < METHOD: GPT2 FINETUNING > #####
    # Sample with p = 0.95, 0.99
    for seed in $(seq 1 1 $nseed); do
        # p = 0.99
        nlprun -n toy_gpt2 -g 1 'python run_generation_with_classification.py --model_type=gpt2 --model_name_or_path="'${path2repo}'/decoder/examples/pytorch/language-modeling/LM_toyWiki_gpt2_seed'${decoder_seed}'/" --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection_filter --project='${project}' --tag='${tag}' --p=0.99 --label=gpt --cl_fpath="'${path2repo}'/encoder/code/encoder_models/wikisection8_tc/checkpoints/epoch=99-step=127199.ckpt" --latent_dim=8  --seed='${seed} -a ${env_name} -p low


        ##### < METHOD: SECTION NULL > ##### ### SPARSE
        # p = 0.99
        nlprun -n toy_secNull -g 1 'python run_section_id_generation.py --model_type=gpt2 --model_name_or_path="'${path2repo}'/decoder/examples/pytorch/language-modeling/LM_toyWiki_gpt2_secSparse_seed'${decoder_seed}'/" --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection_filter --project='${project}' --p=0.99 --label=secSparse --use_section_null --cl_fpath="'${path2repo}'/encoder/code/encoder_models/wikisection8_tc/checkpoints/epoch=99-step=127199.ckpt" --latent_dim=8  --tag='${tag}' --seed='${seed} -a ${env_name} -p low

        ##### < METHOD: SECTION ID > #####
        # p = 0.99, 0.95
        nlprun -n toy_secID -g 1 'python run_section_id_generation.py --model_type=gpt2 --model_name_or_path="'${path2repo}'/decoder/examples/pytorch/language-modeling/LM_toyWiki_gpt2_secDense_seed'${decoder_seed}'/" --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection_filter --project='${project}' --p=0.99 --label=secDense --cl_fpath="'${path2repo}'/encoder/code/encoder_models/wikisection8_tc/checkpoints/epoch=99-step=127199.ckpt" --latent_dim=8  --tag='${tag}' --seed='${seed} -a ${env_name} -p low


        ##### < METHOD: CL EMBEDDINGS > #####
        for latent_dim in {8,16,32}; do
            method='infonce'
            nlprun -n toy_cl -g 1 'python run_cl_transition_fulldoc_generation_da_bb.py --model_type=gpt2 --model_name_or_path="'${path2repo}'/decoder/examples/pytorch/language-modeling/LM_toyWiki_'${method}''${latent_dim}'_seed'${decoder_seed}'/" --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection_filter --project='${project}' --p=0.99 --seed='${seed}' --label='${method}''${latent_dim}'_bb  --cl_fpath='${path2repo}'/encoder/code/encoder_models/wikisection'${latent_dim}'_'${method}'/checkpoints/epoch=99-step=127199.ckpt --tag='${tag}' --latent_dim='${latent_dim} -a ${env_name} -p low
            nlprun -n toy_cl -g 1 'python run_cl_transition_fulldoc_generation_da.py --model_type=gpt2 --model_name_or_path="'${path2repo}'/decoder/examples/pytorch/language-modeling/LM_toyWiki_'${method}''${latent_dim}'_seed'${decoder_seed}'/" --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection_filter --project='${project}' --p=0.99 --seed='${seed}' --label='${method}''${latent_dim}'  --cl_fpath='${path2repo}'/encoder/code/encoder_models/wikisection'${latent_dim}'_'${method}'/checkpoints/epoch=99-step=127199.ckpt --tag='${tag}' --latent_dim='${latent_dim} -a ${env_name} -p low

            method='brownian'
            nlprun -n toy_cl -g 1 'python run_cl_transition_fulldoc_generation_da_bm.py --model_type=gpt2 --model_name_or_path="'${path2repo}'/decoder/examples/pytorch/language-modeling/LM_toyWiki_'${method}''${latent_dim}'_seed'${decoder_seed}'/" --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection_filter --project='${project}' --p=0.99 --seed='${seed}' --label='${method}''${latent_dim}'_bm --cl_fpath='${path2repo}'/encoder/code/encoder_models/wikisection'${latent_dim}'_'${method}'/checkpoints/epoch=99-step=127199.ckpt  --tag='${tag}' --latent_dim='${latent_dim} -a ${env_name} -p low
            nlprun -n toy_cl -g 1 'python run_cl_transition_fulldoc_generation_da.py --model_type=gpt2 --model_name_or_path="'${path2repo}'/decoder/examples/pytorch/language-modeling/LM_toyWiki_'${method}''${latent_dim}'_seed'${decoder_seed}'/" --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection_filter --project='${project}' --p=0.99 --seed='${seed}' --label='${method}''${latent_dim}' --cl_fpath='${path2repo}'/encoder/code/encoder_models/wikisection'${latent_dim}'_'${method}'/checkpoints/epoch=99-step=127199.ckpt  --tag='${tag}' --latent_dim='${latent_dim} -a ${env_name} -p low

            # p = 0.99
            method='tc'
            nlprun -n toy_cl -g 1 'python run_cl_transition_fulldoc_generation_da_bb.py --model_type=gpt2 --model_name_or_path="'${path2repo}'/decoder/examples/pytorch/language-modeling/LM_toyWiki_'${method}''${latent_dim}'_seed'${decoder_seed}'/" --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection_filter --project='${project}' --p=0.99 --seed='${seed}' --label='${method}''${latent_dim}'_bb --cl_fpath='${path2repo}'/encoder/code/encoder_models/wikisection'${latent_dim}'_'${method}'/checkpoints/epoch=99-step=127199.ckpt --tag='${tag}' --latent_dim='${latent_dim} -a ${env_name} -p low
            nlprun -n toy_cl -g 1 'python run_cl_transition_fulldoc_generation_da.py --model_type=gpt2 --model_name_or_path="'${path2repo}'/decoder/examples/pytorch/language-modeling/LM_toyWiki_'${method}''${latent_dim}'_seed'${decoder_seed}'/" --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection_filter --project='${project}' --p=0.99 --seed='${seed}' --label='${method}''${latent_dim}' --cl_fpath='${path2repo}'/encoder/code/encoder_models/wikisection'${latent_dim}'_'${method}'/checkpoints/epoch=99-step=127199.ckpt --tag='${tag}' --latent_dim='${latent_dim} -a ${env_name} -p low

            method='vae'
            nlprun -n toy_cl -g 1 'python run_cl_transition_fulldoc_generation_da_bb.py --model_type=gpt2 --model_name_or_path="'${path2repo}'/decoder/examples/pytorch/language-modeling/LM_toyWiki_'${method}''${latent_dim}'_seed'${decoder_seed}'/" --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection_filter --project='${project}' --p=0.99 --seed='${seed}' --label='${method}''${latent_dim}'_bb --tag='${tag}' --cl_fpath='${path2repo}'/encoder/code/encoder_models/wikisection'${latent_dim}'_'${method}'/checkpoints/epoch=99-step=127199.ckpt --latent_dim='${latent_dim} -a ${env_name} -p low
            nlprun -n toy_cl -g 1 'python run_cl_transition_fulldoc_generation_da.py --model_type=gpt2 --model_name_or_path="'${path2repo}'/decoder/examples/pytorch/language-modeling/LM_toyWiki_'${method}''${latent_dim}'_seed'${decoder_seed}'/" --prompt="<|endoftext|>" --num_return_sequences=1 --num_intervals=1000 --method=sample --stop_token="<|endoftext|>" --dataset_name=wikisection_filter --project='${project}' --p=0.99 --seed='${seed}' --label='${method}''${latent_dim}' --tag='${tag}' --cl_fpath='${path2repo}'/encoder/code/encoder_models/wikisection'${latent_dim}'_'${method}'/checkpoints/epoch=99-step=127199.ckpt --latent_dim='${latent_dim} -a ${env_name} -p low
        done
    done
done

