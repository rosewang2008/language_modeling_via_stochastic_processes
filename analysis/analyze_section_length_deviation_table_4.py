# Pull data from wandb and analyze discourse
"""
Table 5
"""

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from collections import defaultdict


project_name = 'ICLR2022_long_text_evaluation'

# Get the run data
api = wandb.Api(timeout=69)
runs = api.runs(f'rewang/{project_name}')

# Get the data from the runs
"""
I want to track 
- split with vs. w/o noBB
- dataset
- method
- seed


Take BRIDGE/TITLE length abs diff and the normal length -> calculate ratio

Wikihow:
    most_recent['TITLE GT'] = 10.4
    most_recent['METHOD GT'] = 8.7
    most_recent['STEP GT'] = 101.02 # 480.2
    sec_names = ['TITLE', 'METHOD', 'STEP']

Taskmaster: 
    most_recent['USER GT'] = 11.79
    most_recent['ASSISTANT GT'] = 17.96
    sec_names = ['USER', 'ASSISTANT']

Wikisection Wikihow TicketTalk Recipe
"""

dataset2step = {
    'wikihow': 720,
    'restaurant_taskmaster': 3800,
    'taskmaster': 3500,
}

gpt2_step2dataset = {
   'recipe': 998,
    'taskmaster': 1186,
    'restaurant_taskmaster': 1186,
    'wikihow': 242,
    'wikisection_filter': 430,
}


dataset2keys = {
    'wikihow': ['TITLE', 'METHOD', 'STEP'],
    'restaurant_taskmaster': ['USER', 'ASSISTANT'],
    'taskmaster': ['USER', 'ASSISTANT'],
}

key2length = { # Calculate the length abs diff from ground truth lenght -> report mismatch
    'TITLE': 10.4,
    'METHOD': 8.7,
    'STEP': 101.02,
    'USER': 11.79,
    'ASSISTANT': 17.96,
}

data = []
modeldatasetdim2seed = defaultdict(list)
for wandb_run in runs:
    if not wandb_run.config: # Empty config
        continue
    
    dataset = wandb_run.config['dataset_name']

    if 'gpt2' in wandb_run.config['model_name_or_path']:
        latent_dim = 1
        model = 'gpt2'
    else:
        latent_dim = wandb_run.config['latent_dim']
        model = wandb_run.config['label'].split('_')[1].split(str(latent_dim))[0]

    if '_bb' in wandb_run.config['label']:
        with_buggy_decoding = False
    else:
        with_buggy_decoding = True


    if '_bm' in wandb_run.config['label']:
        with_buggy_decoding = False # non buggy decoding for brownian motion    
    elif '_bb' in wandb_run.config['label'] and 'brownian' in wandb_run.config['label']:
        # Don't use this
        continue

    if dataset not in dataset2step:
        print('Dataset {} not in dataset2step'.format(dataset))
        continue

    keys = dataset2keys[dataset]
    if 'gpt2' in wandb_run.config['model_name_or_path']:
        wandb_keys = ['[ {} ] length recent'.format(key) for key in keys]
    else:
        wandb_keys = ['BRIDGE/[ {} ] length recent'.format(key) for key in keys]

    # Check history is not empty
    history = wandb_run.history()
    if history.empty:
        print('Run {} has empty history'.format(wandb_run.id))
        continue

    history = wandb_run.scan_history(['_step', ] + wandb_keys)
    history = pd.DataFrame(history)

    # Skip the CHECK runs 
    if 'CHECK' in wandb_run.config['label']:
        print('Run {} is a CHECK run'.format(wandb_run.id))  
        continue

    if 'gpt2' in wandb_run.config['model_name_or_path']:
        step_min = gpt2_step2dataset[dataset]
        if step_min > history['_step'].max():
            print('Run {} has step {} < {}'.format(wandb_run.id, history['_step'].max(), step_min))
            continue
    elif not ('gpt2' in wandb_run.config['model_name_or_path']):
        step_min = dataset2step[dataset]
        if (step_min > history['_step'].max()):
            print('Run {} has step {} < {}'.format(wandb_run.id, history['_step'].max(), step_min))
            continue

    # # Check if number of steps is correct
    # if not (dataset2step[dataset] <= wandb_run.history()['_step'].max()):
    #     continue
    config = wandb_run.config
    model_dataset = '{}_{}_{}_buggyDecoding{}'.format(model, latent_dim, dataset, with_buggy_decoding)

    if config['seed'] in modeldatasetdim2seed[model_dataset]:
        print('Run {} has duplicate seed'.format(wandb_run.id))
        continue
    else:
        modeldatasetdim2seed[model_dataset].append(config['seed'])

    metrics = {
        'dataset': dataset,
        'seed': config['seed'],
        'model': model,
        'latent_dim': latent_dim,
        'with_buggy_decoding': with_buggy_decoding,
    }

    # Calculating metrics

    avg_dev = 0.0
    for key, wandb_key in zip(keys, wandb_keys):
        results = history[wandb_key].values
        # Remove nans
        results = results[~np.isnan(results)]
        # Get the last value
        last_value = results[-1]
        # Calculate the abs diff
        abs_diff = abs(last_value - key2length[key])
        # Calculate the ratio
        ratio = abs_diff / key2length[key]
        avg_dev += ratio
    
    metrics['average_section_length_deviation'] = avg_dev / len(keys)
    data.append(metrics)

# Check how many unique seeds per model
df = pd.DataFrame(data)

for model in df['model'].unique():
    print(model, len(df[(df['model'] == model)]['seed'].unique()))
    

# Print accuracy results: mean and std

# Save to csv

results  = []

for with_buggy_decoding in [True, False]:
    print('----------------- with_buggy_decoding', with_buggy_decoding, '-----------------')
    for dataset in df['dataset'].unique():
        print('------------------')
        print('\nDataset: {}'.format(dataset))

        for model in df['model'].unique():
            print()
            print('\n>>>> model: {}'.format(model))
            model_df = df[(df['model'] == model) & (df['dataset'] == dataset)]
            # Sorted latent dim and check for Nones 
            if 'latent_dim' in model_df.columns:
                model_df = model_df.sort_values('latent_dim')
            # if model == 'vae' or model == 'tc' or model == 'brownian' or model == 'infonce':
            for latent_dim in model_df['latent_dim'].unique():
                df_temp = df[(df['dataset'] == dataset) & (df['model'] == model) & (df['latent_dim'] == latent_dim) & (df['with_buggy_decoding'] == with_buggy_decoding)]
                # Check number of seeds
                print('num seeds', len(df_temp['seed'].unique()), df_temp['seed'].unique())
                print('latent_dim: {}, avg_deviation: {:.3f} +- {:.3f}'.format(
                    latent_dim, df_temp['average_section_length_deviation'].mean()*100, df_temp['average_section_length_deviation'].std()*100))

                results.append({
                    'name': f"{model} ({latent_dim})",
                    'dataset': dataset,
                    "with_buggy_decoding": with_buggy_decoding,
                    "score": f"{df_temp['average_section_length_deviation'].mean()*100:.3f} +- {df_temp['average_section_length_deviation'].std()*100:.3f}"
                })

# Save 
df = pd.DataFrame(results)
df.to_csv('section_length_deviation_table_4.csv')

