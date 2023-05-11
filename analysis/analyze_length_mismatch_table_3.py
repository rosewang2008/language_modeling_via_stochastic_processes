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


project_name = 'ICLR2022_text_evaluation'

# Get the run data
api = wandb.Api(timeout=69)
runs = api.runs(f'rewang/{project_name}')

SECTION_NAMES = [
    'ABSTRACT',
    'GEOGRAPHY',
    'HISTORY',
    'DEMOGRAPHICS',
]

gpt2_step2dataset = {
    'wikisection_filter': 430,
}
dataset2step = {
    'wikisection_filter': 1291, 
}

data = []
modeldatasetdim2seed = defaultdict(list)
for wandb_run in runs:
    if not wandb_run.config: # Empty config
        continue

    print('looking at:', wandb_run)
    
    dataset = wandb_run.config['dataset_name']

    if dataset != 'wikisection_filter':
        print('dataset not wikisection_filter', dataset)
        continue

    source_keys = ['[ {} ] length recent'.format(key) for key in SECTION_NAMES]
    target_keys = ['{} GT'.format(key) for key in SECTION_NAMES]

    # Check history is not empty
    history = wandb_run.history()
    if history.empty:
        print(wandb_run.config['label'], 'history empty')
        continue

    model = wandb_run.config['label']

    if ('tc' in model) or ('brownian' in model) or ('infonce' in model) or ('vae' in model):
        source_keys = ['BRIDGE/[ {} ] length recent'.format(key) for key in SECTION_NAMES]

    history = wandb_run.scan_history(['_step', ] + source_keys + target_keys) 
    history = pd.DataFrame(history)

    config = wandb_run.config

    # Check that state of run is finished
    if wandb_run.state != 'finished':
        print(wandb_run, wandb_run.config['label'], 'status not finished: ', wandb_run.state)
        continue

    if '_bb' in wandb_run.config['label']:
        with_buggy_decoding = False
    else:
        with_buggy_decoding = True

    if '_bm' in wandb_run.config['label']:
        with_buggy_decoding = False # non buggy decoding for brownian motion    
    elif '_bb' in wandb_run.config['label'] and 'brownian' in wandb_run.config['label']:
        # Don't use this
        continue

    if 'gpt' in model:
        latent_dim = 1
    elif 'secDense' in model:
        latent_dim = 1
    elif 'secSparse' in model:
        latent_dim = 1
    else:
        latent_dim = wandb_run.config['latent_dim']
        if 'tc' in model:
            model = 'tc'
        elif 'brownian' in model:
            model = 'brownian'
        elif 'vae' in model:
            model = 'vae'
        elif 'infonce' in model:
            model = 'infonce'
        else:
            raise ValueError('model not found')

    model_dataset = '{}_{}_{}_buggyDecoding{}'.format(model, latent_dim, dataset, with_buggy_decoding)

    if config['seed'] in modeldatasetdim2seed[model_dataset] or (config['seed'] not in [1,2,3]):
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

    try: 
        avg_dev = 0.0
        for source_key, target_key in zip(source_keys, target_keys):
            source_result = history[source_key].values[-1]
            target_result = history[target_key].values[-1]
            # Calculate the abs diff
            abs_diff = abs(source_result - target_result)
            # Calculate the ratio
            ratio = abs_diff / target_result
            avg_dev += ratio
    except:
        print('error', model)
        continue
    
    metrics['average_length_mismatch'] = avg_dev / len(source_keys)
    data.append(metrics)

# Check how many unique seeds per model
df = pd.DataFrame(data)

for model in df['model'].unique():
    print(model, len(df[(df['model'] == model)]['seed'].unique()))

# Print accuracy results: mean and std
# Create csv

results = []

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
                    latent_dim, df_temp['average_length_mismatch'].mean()*100, df_temp['average_length_mismatch'].std()*100))

                results.append({
                    'name': f"{model} ({latent_dim})",
                    'dataset': dataset,
                    'with_buggy_decoding': with_buggy_decoding,
                    'score': f"{df_temp['average_length_mismatch'].mean()*100:.3f} +- {df_temp['average_length_mismatch'].std()*100:.3f}",
                })

# Save results csv
df_results = pd.DataFrame(results)
df_results.to_csv('length_mismatch_table_3.csv')

