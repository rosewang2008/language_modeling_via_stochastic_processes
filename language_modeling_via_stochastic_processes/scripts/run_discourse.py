import os
import wandb
import hydra
import getpass
from copy import deepcopy
import random
import torch
import numpy
from language_modeling_via_stochastic_processes.src.systems import discourse_system
import pytorch_lightning as pl

torch.backends.cudnn.benchmark = True

SYSTEM = {
    discourse_system.DiscourseSystem
}

@hydra.main(config_path="../config/local_coherence_modelling", config_name="discourse_coherence")
def run(config):
    print(config)
    if config.wandb_settings.dryrun:
        print("Running in dryrun mode")
        os.environ['WANDB_MODE'] = 'dryrun'
    os.environ['WANDB_CONSOLE']='wrap'
    seed_everything(
        config.experiment_params.seed,
        use_cuda=config.experiment_params.cuda)

    wandb.init(
        project=config.wandb_settings.project,
        group=config.wandb_settings.group,
        entity=getpass.getuser(),
        name=config.wandb_settings.exp_name,
        config=config,
    )

    SystemClass = discourse_system.DiscourseSystem
    system = SystemClass(config)

    trainer = pl.Trainer(
        default_root_dir=config.wandb_settings.exp_dir,
        gpus=1,
        max_epochs=int(config.experiment_params.num_epochs),
        min_epochs=int(config.experiment_params.num_epochs),
    )

    trainer.fit(system)
    trainer.test(system)

def seed_everything(seed, use_cuda=True):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda: torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    run()
