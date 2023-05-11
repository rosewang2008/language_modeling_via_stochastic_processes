import os
import json
import shutil
import torch
import numpy as np
from collections import Counter, OrderedDict
from dotmap import DotMap
from pytorch_lightning.callbacks.progress import ProgressBar
from collections import defaultdict
import scipy.interpolate as interp
import tqdm
import matplotlib.pyplot as plt
import wandb

import urllib
import os
import json

WEBHOOK_URL = "https://hooks.slack.com/services/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" # os.environ.get('SLACK_WEBHOOK_URL')

if WEBHOOK_URL is None:
    print('Environment variable SLACK_WEBHOOK_URL not set: Slack messages will not be sent.')

def send_message(text):
    if WEBHOOK_URL is not None:
        r = urllib.request.Request(WEBHOOK_URL,
                                   data=json.dumps({'text': text}).encode('utf-8'),
                                   headers={
                                       'Content-Type': 'application/json'
                                   },
                                   method='POST')
        with urllib.request.urlopen(r) as f:
            status = str(f.status)
    else:
        status = 'not sent - no webhook URL'

    print('Slack message: {} (status: {})'.format(text, status))

def load_json(f_path):
    with open(f_path, 'r') as f:
        return json.load(f)


def save_json(obj, f_path):
    with open(f_path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)



class RoseProgressBar(ProgressBar):

    def init_validation_tqdm(self):
        bar = tqdm(
            position=0,
            leave=True
        )
        return bar


# Aug 17 Zero shot expt: H1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_feats(model, dataset, x, device):
    input_ids, attention_mask = dataset.tokenize_caption(
         x, device=device)
    if 'BERT' in type(model).__name__:
        input_ids = input_ids[:, :512]
        attention_mask = attention_mask[:, :512]
    torch.cuda.empty_cache()
    feats = model.forward(input_ids=input_ids, attention_mask=attention_mask)
    del input_ids
    del attention_mask
    torch.cuda.empty_cache()
    return feats


def calculate_recipe_embeddings(model, dataset, batch_size):
    """
    Calculates z, 1-z, and 1-abs(z)
    for recipe because its in a pd frame
    """
    index = 0
    max_num_sentences = 0

    embeddings = defaultdict(lambda: []) # list of embeddings over sequence
    diff_embeddings = defaultdict(lambda: []) # list of embeddings over sequence
    diff_abs_embeddings = defaultdict(lambda: []) # list of embeddings over sequence

    skipped = 0

    doc_key = 'story_id' if 'stories' in type(dataset).__name__.lower() else 'doc_id'
    total_key = 'total_sentences' if 'stories' in type(dataset).__name__.lower() else 'total_doc_sentences'

    for i in tqdm.tqdm(range(dataset.processed_data.iloc[-1][doc_key]-10)):
        if i > 0:
            index = index + doc_length

        try:
            doc_length = dataset.processed_data.iloc[index][total_key]
        except:
            break
        max_num_sentences = max(max_num_sentences, doc_length)

        x = [dataset.processed_data.iloc[i]['sentence'] for i in range(index, index + doc_length)]

        split_size = batch_size
        split_x = [x[i:i+split_size] for i in range(0, len(x), split_size)]
        feats = []
        for split in split_x:
            if len(split) == 1:
                sent_feats = get_feats(
                    model=model, dataset=dataset, x=split*2, device=device).detach().cpu().numpy()
                sent_feats = sent_feats[:1,]
            else:
                sent_feats = get_feats(
                    model=model, dataset=dataset, x=split, device=device).detach().cpu().numpy()

            torch.cuda.empty_cache()
            feats.append(sent_feats)

        sent_feats = np.concatenate(feats)
        for _ in range(sent_feats.shape[-1]):
            embeddings[_].append(sent_feats[:, _])
            diff_embeddings[_].append(1-sent_feats[:, _])
            diff_abs_embeddings[_].append(1-abs(sent_feats[:, _]))

    info = {'max_num_sentences': max_num_sentences}
    return embeddings, diff_embeddings, diff_abs_embeddings, info


def calculate_embeddings(model, dataset, batch_size):
    """
    Calculates z, 1-z, and 1-abs(z)
    """
    index = 0
    max_num_sentences = 0

    embeddings = defaultdict(lambda: []) # list of embeddings over sequence
    diff_embeddings = defaultdict(lambda: []) # list of embeddings over sequence
    diff_abs_embeddings = defaultdict(lambda: []) # list of embeddings over sequence

    skipped = 0

    # doc_key = 'story_id' if 'stories' in type(dataset).__name__.lower() else 'doc_id'
    doc_key = 'doc_id'
    # total_key = 'total_sentences' if 'stories' in type(dataset).__name__.lower() else 'total_doc_sentences'
    total_key = 'total_doc_sentences'

    for i in tqdm.tqdm(range(dataset.processed_data[-1][doc_key]-10)):
        if i > 3000: # dont evaluate everything
            break
        if i > 0:
            index = index + doc_length

        try:
            doc_length = dataset.processed_data[index][total_key]
        except:
            break
        max_num_sentences = max(max_num_sentences, doc_length)

        x = [dataset.processed_data[i]['sentence'] for i in range(index, index + doc_length)]

        split_size = batch_size
        split_x = [x[i:i+split_size] for i in range(0, len(x), split_size)]
        feats = []
        for split in split_x:
            if len(split) == 1:
                sent_feats = get_feats(
                    model=model, dataset=dataset, x=split*2, device=device).detach().cpu().numpy()
                sent_feats = sent_feats[:1,]
            else:
                sent_feats = get_feats(
                    model=model, dataset=dataset, x=split, device=device).detach().cpu().numpy()

            torch.cuda.empty_cache()
            feats.append(sent_feats)

        sent_feats = np.concatenate(feats)
        for _ in range(sent_feats.shape[-1]):
            embeddings[_].append(sent_feats[:, _])
            diff_embeddings[_].append(1-sent_feats[:, _])
            diff_abs_embeddings[_].append(1-abs(sent_feats[:, _]))

    info = {'max_num_sentences': max_num_sentences}
    return embeddings, diff_embeddings, diff_abs_embeddings, info

def create_plot(embeddings, info, title_name, disable_per_dim=True):

    ref = np.arange(info['max_num_sentences'])

    interpolated_nn_dist_mean = []
    interpolated_nn_dist_std = []

    all_mean = []
    all_std = []

    for dim in range(len(embeddings)):

        doc_nn_dist_mean = embeddings[dim]

        for i in range(len(embeddings[dim])):
            # Interpolate the means
            arr2 = doc_nn_dist_mean[i]
            arr2_interp = interp.interp1d(np.arange(arr2.size),arr2[:])
            arr2_stretch = arr2_interp(np.linspace(0,arr2.size-1,ref.size))
            arr2_stretch = np.expand_dims(arr2_stretch, 1)
            interpolated_nn_dist_mean.append(arr2_stretch)

        mean = np.concatenate(interpolated_nn_dist_mean, axis=1).mean(1)
        std = np.concatenate(interpolated_nn_dist_mean, axis=1).std(1)

        all_mean.append(mean)
        all_std.append(std)

        if not disable_per_dim:
            plt.errorbar(t, mean,
                         yerr=std,
                         fmt='-o')

            plt.title(f'{title_name}[dim {dim}]')
            plt.savefig(fname)
            plt.clf()

    t = np.arange(len(mean))
    plt.errorbar(t, np.array(all_mean).mean(0),
                 yerr=np.array(all_std).mean(0),
                 fmt='-o')
    plt.hlines(y=0, xmin=0, xmax=len(t), colors= 'r', linestyles='--')
    plt.title(f'{title_name}')
    # wandb.log({f"{title_name}": plt})
    plt.savefig("temp.jpg")
    wandb.log({f"{title_name}_image": wandb.Image("temp.jpg")})
    plt.clf()


def calculate_zero_shot(model, dataset, batch_size, notes):
    # if 'recipe' in type(dataset).__name__.lower():
    #     embeddings, diff_embeddings, diff_abs_embeddings, info = calculate_recipe_embeddings(
    #         model=model, dataset=dataset, batch_size=batch_size)
    # else:
    embeddings, diff_embeddings, diff_abs_embeddings, info = calculate_embeddings(
        model=model, dataset=dataset, batch_size=batch_size)

    create_plot(embeddings, info, title_name=f'{notes}/z')
    create_plot(diff_embeddings, info, title_name=f'{notes}/1-z')
    create_plot(diff_abs_embeddings, info, title_name=f'{notes}/1-abs(z)')
