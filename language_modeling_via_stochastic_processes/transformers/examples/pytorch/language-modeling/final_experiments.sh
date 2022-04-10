#!/bin/bash

domains=("recipe" "roc_stories" "tm2" "tickettalk" "wikihow" "wikisection")
path2repo='/path/2/repo'

for domain in ${domains[@]}; do
    for latent_dim in {8,16,32}; do
        # Time control
        python run_time_clm.py --model_name_or_path gpt2 --dataset_name ${domain} --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --seed=1 --encoder_filepath=${path2repo}/language_modeling_via_stochastic_processes/models/${domain}/tc${latent_dim}/epoch=99-step=21999.ckpt --latent_dim=${latent_dim} --output_dir LM_${domain}_${latent_dim} --evaluation_strategy=steps --eval_steps=1000 --use_contrastive_embeddings
        # VAE
        python run_time_clm.py --model_name_or_path gpt2 --dataset_name ${domain} --do_train --do_eval --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --save_total_limit=1 --load_best_model_at_end=True --overwrite_output_dir --num_train_epochs=10 --seed=1 --encoder_filepath=${path2repo}/language_modeling_via_stochastic_processes/models/${domain}/vae${latent_dim}/epoch=99-step=21999.ckpt --latent_dim=${latent_dim} --output_dir LM_${domain}_${latent_dim} --evaluation_strategy=steps --eval_steps=1000 --use_contrastive_embeddings
        # ...etc
    done
done

