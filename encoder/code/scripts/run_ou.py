"""
source init_env.sh
python scripts/run.py --dryrun

or with interactive

WANDB_CONSOLE="wrap" python scripts/run_ou.py --dryrun
"""

import os
import platform
import wandb
import getpass
from copy import deepcopy
import random, torch, numpy
from src.systems import system, ou_system, vae_system, cl_system
from src.utils import load_json, RoseProgressBar
from src.setup import process_config
# from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from src.evaluation import recovery
from src import utils

torch.backends.cudnn.benchmark = True

SYSTEM = {
    'TCLSystem': system.TCLSystem,
    'WikiTCLSystem': system.WikiTCLSystem,
    'WikiBrownianBridgeSystem': system.WikiBrownianBridgeSystem,
    'WikiInfoNCESystem': system.WikiInfoNCESystem,
    'WikiVAESystem': system.WikiVAESystem,
    'WikiOUSystem': system.WikiOUSystem,
    'StoriesBrownianBridgeSystem': system.StoriesBrownianBridgeSystem,
    'WikihowBrownianBridgeSystem': system.WikihowBrownianBridgeSystem,
    'RecipeBrownianBridgeSystem': system.RecipeBrownianBridgeSystem,
    'BinaryWikiSystem': system.BinaryWikiSystem,
    'DeltaBinaryWikiSystem': system.BinaryDeltaWikiSystem,
    'OUSystem': ou_system.OUSystem,
    'NoisyOUSystem': ou_system.NoisyOUSystem,
    'OUSystem_Classification': ou_system.OUSystem_Classification,
    'OUSystem_LearnMu': ou_system.OUSystem_LearnMu,
    'BrownianBridgeSystem': ou_system.BrownianBridgeSystem,
    'BrownianBridgeEverythingSystem': ou_system.BrownianBridgeEverythingSystem,
    # 'VAESystem': vae_system.VAESystem,
    # 'VAEBridgeSystem': vae_system.VAEBridgeSystem,
    'CLSystem': cl_system.CLSystem,
    'CLBridgeSystem': cl_system.CLBridgeSystem,
}

JUICE_DIR = "/juice/u/rewang/experiments"

stepDict = {1: [int(5e3), int(5e3)],
            2: [int(1e4), int(1e4)],
            3: [int(1e4), int(1e4)],
            4: [int(1e4), int(1e4)],
            5: [int(1e4), int(1e4)]}

def run(args):
    if args.dryrun:
        print("Running in dryrun mode")
        os.environ['WANDB_MODE'] = 'dryrun'
    os.environ['WANDB_CONSOLE']='wrap'

    config_path = args.config
    config = process_config(config_path, args=args)

    if args.project:
        config.project=args.project

    if args.loss:
        config.loss_params.loss =args.loss

    if args.dataset:
        config.data_params.dataset_loader =args.dataset
        config.data_params.name = args.dataset

    if args.seed:
        config.seed = args.seed
        config.data_params.data_seed = args.seed
    seed_everything(config.seed, use_cuda=config.cuda)
    print("seed", config.seed)

    ### Modifying config with flags ###
    if args.p is not None:
        config.data_params.p = float(args.p)

    config.k = args.k
    config.contrast = args.contrast

    config.data_params.tm_type = args.tm_type
    config.data_params.dt = args.dt
    config.use_section_ids = args.use_section_ids
    if args.batch_size:
        config.optim_params.batch_size = args.batch_size

    if args.data_dim:
        config.data_params.data_dim = args.data_dim
    if args.latent_dim:
        config.data_params.latent_dim = args.latent_dim
    # make sure hidden has the same dim as data
    config.data_params.hidden_dim = config.data_params.data_dim

    if args.exp_name:
        config.exp_name = args.exp_name

    if args.num_epochs:
        config.num_epochs = args.num_epochs

    if args.checkpoint_steps:
        config.checkpoint_step = args.checkpoint_steps

    if args.interval:
        config.data_params.sampling_interval = args.interval

    ####################################

    assert config.data_params.data_dim > 1

    utils.send_message(f'Starting run {config.exp_name}')
    wandb_run = wandb.init(
        project=config.project,
        entity=getpass.getuser(),
        name=config.exp_name,
        config=config,
    )

    if args.tag:
        wandb_run.tags = wandb_run.tags + (args.tag,)
        print(f'[ Wandb tags ] {wandb_run.tags}')

    config.exp_dir = wandb.run.dir
    ckpt_dir = os.path.join('/nlp/scr/rewang/public_language_modeling_via_stochastic_processes/encoder/code', 'encoder_models', args.exp_name, 'checkpoints')
    print("CKPT AT {}".format(os.path.join(args.exp_name, 'checkpoints')))

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        # os.path.join(wandb.run.dir, 'checkpoints'),
        # os.path.join(args.exp_name, 'checkpoints'),
        ckpt_dir,
        save_top_k=-1,
        period=config.checkpoint_steps,
    )

    SystemClass = SYSTEM[config.system]
    system = SystemClass(config)

    # save_directory = os.path.join(JUICE_DIR, config.exp_name)
    # if not os.path.exists(save_directory):
    #     os.makedirs(save_directory)
    # print("Save directory is {}".format(save_directory))
    print("Save directory is {}".format(wandb.run.dir))

    # bar = RoseProgressBar()
    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=1,
        checkpoint_callback=ckpt_callback,
        max_epochs=int(config.num_epochs),
        min_epochs=int(config.num_epochs),
        # callbacks=[bar]
    )

    trainer.fit(system)

    ## Save the model
    system.save(directory=wandb.run.dir)
    utils.send_message(f'Done with run {config.exp_name}')

    ## Evaluation:
    trainer.test(system)

def seed_everything(seed, use_cuda=True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Seeding CUDA convolution operations
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Seeding algorithms
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.set_deterministic(True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/ou.json",
                        help='path to config file')
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--k', type=int, required=True) # num sentences off when sampling
    parser.add_argument('--contrast', type=str, required=True) # num sentences off when sampling
    parser.add_argument('--dryrun', default=False, action='store_true')
    parser.add_argument('--use-section-ids', default=False, action='store_true')
    parser.add_argument('--dt', default=0.01, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--data-dim', default=0, type=int)
    parser.add_argument('--latent-dim', default=0, type=int)
    parser.add_argument('--num-epochs', default=0, type=int)
    parser.add_argument('--checkpoint-steps', default=0, type=int)
    parser.add_argument('--batch-size', default=0, type=int)
    parser.add_argument('--interval', default=0, type=int)
    parser.add_argument('--p', default=None)
    parser.add_argument('--tm-type', default="movies", type=str)
    parser.add_argument('--project', default="", type=str)
    parser.add_argument('--loss', default="", type=str)
    parser.add_argument('--dataset', default="", type=str)
    parser.add_argument('--tag', default="", type=str)
    args = parser.parse_args()

    run(args)



