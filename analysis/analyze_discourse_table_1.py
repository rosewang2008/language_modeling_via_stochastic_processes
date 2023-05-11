# Pull data from wandb and analyze discourse
"""
Table 1
"""

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from collections import defaultdict


project_name = 'ICLR2022_v2_discourse'

# Get the run data
api = wandb.Api(timeout=69)
runs = api.runs(f'rewang/{project_name}')

# Get the data from the runs
"""
I want to track 
- dataset
- method
- seed
- k
"""
data = []
modeldatasetkdim2seed = defaultdict(list)
for wandb_run in runs:

    # Check the run status is finished
    if wandb_run.state != 'finished':
        continue

    # Get run data
    config = wandb_run.config
    history = wandb_run.history()

    # Check if history is empty
    if history.empty:
        continue

    history = wandb_run.scan_history(['_step', 'epoch', 'test_acc'])
    history = pd.DataFrame(history)
    
    max_epoch = history['epoch'].max()
    model = config['exp_name'].split('_')[-1]

    if config['dataset'] == 'taskmaster':
        dataset = '{}_{}'.format(
            config['dataset'], 
            config['data_params']['tm_type'])
    else:
        dataset = config['dataset']

    model_dataset_k = '{}_{}_{}'.format(model, dataset, config['k'])

    if model == 'infonce' or model == 'brownian' or model == 'vae' or model == 'tc':
        has_latent = True
        model_dataset_k = '{}_{}'.format(model_dataset_k, config['data_params']['latent_dim'])
    else:
        has_latent = False

    history = history[history['epoch'] == max_epoch]

    if config['seed'] in modeldatasetkdim2seed[model_dataset_k]:
        print('Repeated seed, skipping')
        continue
    else:
        modeldatasetkdim2seed[model_dataset_k].append(config['seed'])

    if has_latent:
        data.append({
            'dataset': dataset,
            'k': config['k'],
            'seed': config['seed'],
            'model': model,
            'accuracy': history['test_acc'].mean(),
            'latent_dim': config['data_params']['latent_dim'],
        })
    else:
        data.append({
            'dataset': dataset,
            'k': config['k'],
            'seed': config['seed'],
            'model': model,
            'accuracy': history['test_acc'].mean(),
        })


# Check how many unique seeds per model
df = pd.DataFrame(data)
# Save the dataframe
df.to_csv('discourse_table_1.csv', index=False)
print('Saved dataframe to discourse_table_1.csv')

for model in df['model'].unique():
    print(model, len(df[(df['model'] == model)]['seed'].unique()))

model_ordering = [
    'gpt2',
    'bert',
    'albert',
    'sbert',
    'simcse',
    'vae',
    'infonce',
    'brownian',
    'tc',
    ]

# Print accuracy results: mean and std
for dataset in df['dataset'].unique():
    print('----------------')
    print('dataset: {}'.format(dataset))
    # for model in df['model'].unique():
    for model in model_ordering:
        print()
        print('>>>> model: {}'.format(model))
        model_df = df[(df['model'] == model) & (df['dataset'] == dataset)]
        # Sorted latent dim and check for Nones 
        if 'latent_dim' in model_df.columns:
            model_df = model_df.sort_values('latent_dim')

        for latent_dim in model_df['latent_dim'].unique():
            print()
            for k in [5, 10]:
                if model == 'vae' or model == 'tc' or model == 'brownian' or model == 'infonce':
                    df_temp = df[(df['dataset'] == dataset) & (df['k'] == k) & (df['model'] == model) & (df['latent_dim'] == latent_dim)]

                    print('num seeds', len(df_temp['seed'].unique()))
                    print('k={}, latent_dim: {}, accuracy: {:.3f} +- {:.3f}'.format(
                        k, latent_dim, df_temp['accuracy'].mean(), df_temp['accuracy'].std()))
                    # print('dataset: {}, k: {}, model: {}, latent_acc: {:.2f} +- {:.2f}'.format(dataset, k, model, df_temp['accuracy'].mean(), df_temp['accuracy'].std()))
                else:
                    df_temp = df[
                        (df['dataset'] == dataset) & 
                        (df['k'] == k) & 
                        (df['model'] == model)]
                    
                    print('num seeds', len(df_temp['seed'].unique()))
                    print('k={}, accuracy: {:.3f} +- {:.3f}'.format(
                        k, df_temp['accuracy'].mean(), df_temp['accuracy'].std()))


