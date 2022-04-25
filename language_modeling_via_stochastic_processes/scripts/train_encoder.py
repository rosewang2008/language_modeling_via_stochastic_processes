import os
import wandb
import getpass
import random, torch, numpy
from language_modeling_via_stochastic_processes.src.systems import (
    brownian_bridge_system,
    brownian_system,
    vae_system,
    infonce_system
    )

import pytorch_lightning as pl
from language_modeling_via_stochastic_processes.src.evaluation import recovery
import hydra

torch.backends.cudnn.benchmark = True

SYSTEM = {
    'brownian_bridge': brownian_bridge_system.BrownianBridgeSystem,
    'brownian': brownian_system.BrownianSystem,
    'vae': vae_system.VAESystem,
    'infonce': infonce_system.InfoNCESystem
}

@hydra.main(config_path="../config/encoder", config_name="brownian_bridge")
def run(config):
    if config.wandb_settings.dryrun:
        print("Running in dryrun mode")
        os.environ['WANDB_MODE'] = 'dryrun'
    os.environ['WANDB_CONSOLE']='wrap'

    seed_everything(
        config.experiment_params.seed,
        use_cuda=config.experiment_params.cuda)

    wandb.init(
        project=config.wandb_settings.project,
        entity=getpass.getuser(),
        name=config.wandb_settings.exp_name,
        group=config.wandb_settings.group,
        config=config,
    )

    print("CKPT AT {}".format(
        os.path.join(config.wandb_settings.exp_name,
        'checkpoints')))
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(config.wandb_settings.exp_name, 'checkpoints'),
        save_top_k=-1,
        every_n_epochs=config.experiment_params.checkpoint_epochs,
    )

    SystemClass = SYSTEM[config.loss_params.loss]
    system = SystemClass(config)

    trainer = pl.Trainer(
        default_root_dir=config.wandb_settings.exp_dir,
        gpus=1,
        checkpoint_callback=ckpt_callback,
        max_epochs=int(config.experiment_params.num_epochs),
        min_epochs=int(config.experiment_params.num_epochs),
    )

    trainer.fit(system)

    ## Save the model
    system.save(directory=wandb.run.dir)

    ## Evaluation:
    trainer.test(system)

def seed_everything(seed, use_cuda=True):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda: torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    run()



