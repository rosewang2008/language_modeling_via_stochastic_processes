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

Wikisection Wikihow TicketTalk Recipe
"""

dataset2step = {
    'wikisection_filter': 1291, 
    # 'restaurant_taskmaster': 3800,
    'taskmaster': 3500,
    'recipe': 1000,
    'wikihow': 720,
}

gpt2_step2dataset = {
   'recipe': 998,
    'taskmaster': 1186,
    'wikihow': 242,
    'wikisection_filter': 430,
}

key = 'BRIDGE/ordering recent'

data = []
modeldatasetdim2seed = defaultdict(list)
for wandb_run in runs:

    if not wandb_run.config: # Empty config
        continue
    
    dataset = wandb_run.config['dataset_name']

    if dataset not in dataset2step:
        continue

    # Check history is not empty
    history = wandb_run.history()
    if history.empty:
        continue

    if 'gpt2' in wandb_run.config['model_name_or_path']:
        history = wandb_run.scan_history(['_step', 'ordering recent'])
    else:
        history = wandb_run.scan_history(['_step', 'BRIDGE/ordering recent'])

    history = pd.DataFrame(history)

    if 'gpt2' in wandb_run.config['model_name_or_path']:
        step_min = gpt2_step2dataset[dataset]
        if step_min > history['_step'].max():
            continue
    elif not ('gpt2' in wandb_run.config['model_name_or_path']):
        step_min = dataset2step[dataset]
        if (step_min > history['_step'].max()):
            continue

    # Get run data
    config = wandb_run.config
    
    if '_bb' in wandb_run.config['label']:
        with_buggy_decoding = False
    else:
        with_buggy_decoding = True
    
    if '_bm' in wandb_run.config['label']:
        with_buggy_decoding = False # non buggy decoding for brownian motion    
    elif '_bb' in wandb_run.config['label'] and 'brownian' in wandb_run.config['label']:
        # Don't use this
        continue
    
    if 'gpt2' in wandb_run.config['model_name_or_path']:
        latent_dim = 1
        model = 'gpt2'
    else:
        latent_dim = wandb_run.config['latent_dim']
        model = wandb_run.config['label'].split('_')[1].split(str(latent_dim))[0]
        if model == 'tcbb':
            model = 'tc'

    model_dataset = '{}_{}_{}_buggyDecoding{}'.format(model, latent_dim, dataset, with_buggy_decoding)

    if config['seed'] in modeldatasetdim2seed[model_dataset]:
        continue
    else:
        modeldatasetdim2seed[model_dataset].append(config['seed'])

    if 'gpt2' in wandb_run.config['model_name_or_path']:
        results = history['ordering recent']
    else:
        results = history['BRIDGE/ordering recent']

    # Remove nans
    results = results[~np.isnan(results)]
    # Get last element
    ordering_acc = results.iloc[-1]

    data.append({
        'dataset': dataset,
        'seed': config['seed'],
        'model': model,
        'accuracy': ordering_acc,
        'latent_dim': latent_dim,
        'with_buggy_decoding': with_buggy_decoding,
    })


# Check how many unique seeds per model
df = pd.DataFrame(data)

for model in df['model'].unique():
    print(model, len(df[(df['model'] == model)]['seed'].unique()))



# Print accuracy results: mean and std
# Save to csv

results = []

for with_buggy_decoding in [True, False]:
    print('----------------- with_buggy_decoding', with_buggy_decoding, '-----------------')
    for dataset in df['dataset'].unique():
        print('\n>>>> dataset = ', dataset)
        for model in df['model'].unique():
            print('\n> model = ', model)
            model_df = df[(df['model'] == model) & (df['dataset'] == dataset)]
            # Sorted latent dim and check for Nones 
            if 'latent_dim' in model_df.columns:
                model_df = model_df.sort_values('latent_dim')
            # if model == 'vae' or model == 'tc' or model == 'brownian' or model == 'infonce':
            for latent_dim in model_df['latent_dim'].unique():
                df_temp = df[(df['dataset'] == dataset) & (df['model'] == model) & (df['latent_dim'] == latent_dim) & (df['with_buggy_decoding'] == with_buggy_decoding)]
                # Check number of seeds
                print('num seeds', len(df_temp['seed'].unique()), df_temp['seed'].unique())
                print('latent_dim: {}, accuracy: {:.3f} +- {:.3f}'.format(
                    latent_dim, df_temp['accuracy'].mean()*100, df_temp['accuracy'].std()*100))

                
                results.append({
                    'name': f"{model} ({latent_dim})",
                    'dataset': dataset,
                    'with_buggy_decoding': with_buggy_decoding,
                    'score': f"{df_temp['accuracy'].mean()*100:.3f} +- {df_temp['accuracy'].std()*100:.3f}",
                })

# Sve to csv
df = pd.DataFrame(results)
df.to_csv('ordering_table_5.csv', index=False)
        