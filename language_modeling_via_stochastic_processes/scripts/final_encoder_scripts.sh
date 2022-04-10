#!/bin/bash
"""Training the encoder

Loss: brownian_bridge, brownian, vae, infonce
Dim: 8, 16, 32
Dataset: wikisection, wikihow, recipe, roc_stories, tm2, ticketalk

"""
dims=(8 16 32)
losses=('brownian_bridge' 'brownian' 'vae' 'infonce')
datasets=('wikisection' 'wikihow' 'recipe' 'roc_stories' 'tm2' 'tickettalk')
path2repo='path/2/repo'

for dim in ${dims[@]}; do
    for loss in ${losses[@]}; do
        for dataset in ${datasets[@]}; do
            expt=${loss}${dim}'_'${dataset}
            cd ${path2repo}; python scripts/train_encoder.py --config-name=${loss} wandb_settings.exp_name=${expt} wandb_settings.exp_dir=${expt} data_params.name=${dataset} model_params.latent_dim=${dim}
        done
    done
done
