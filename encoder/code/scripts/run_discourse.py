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
from src.systems import discourse_system
from src.utils import load_json, RoseProgressBar
from src.setup import process_config
# from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from src.evaluation import recovery

torch.backends.cudnn.benchmark = True

SYSTEM = {
    discourse_system.DiscourseSystem
}

JUICE_DIR = "/juice/u/rewang/experiments"

stepDict = {1: [int(5e3), int(5e3)],
            2: [int(1e4), int(1e4)],
            3: [int(1e4), int(1e4)],
            4: [int(1e4), int(1e4)],
            5: [int(1e4), int(1e4)]}

def get_local_dir():
    machine_name = platform.node().split(".")[0]
    scratch_dirs = os.listdir(f"/{machine_name}")
    return os.path.join("/{}".format(machine_name),
                        scratch_dirs[-1],
                        "rewang")

def run(args):
    if args.dryrun:
        print("Running in dryrun mode")
        os.environ['WANDB_MODE'] = 'dryrun'
    os.environ['WANDB_CONSOLE']='wrap'

    config_path = args.config
    config = process_config(config_path, args=args)

    if args.seed:
        config.seed = args.seed
        config.data_params.data_seed = args.seed
    seed_everything(config.seed, use_cuda=config.cuda)
    print("seed", config.seed)

    ### Modifying config with flags ###
    if args.encoder:
        config.model = args.encoder
    config.fpath = args.fpath
    config.dataset = args.dataset
    config.overfit = args.overfit
    config.use_gpt2 = args.use_gpt2
    config.use_section_ids = args.use_section_ids
    if args.project:
        config.project=args.project

    if args.batch_size:
        config.optim_params.batch_size = args.batch_size

    if args.k:
        config.k = args.k

    if args.lr:
        config.optim_params.learning_rate = args.lr

    config.data_params.tm_type = args.tm_type
    # local_jag_dir = get_local_dir()
    # config.exp_dir = local_jag_dir
    if args.latent_dim:
        config.data_params.latent_dim = args.latent_dim

    if args.exp_name:
        config.exp_name = args.exp_name

    if args.num_epochs:
        config.num_epochs = args.num_epochs

    if args.interval:
        config.data_params.sampling_interval = args.interval

    if args.project:
        config.project= args.project

    ####################################
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

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(wandb.run.dir, 'checkpoints'),
        save_top_k=-1,
        period=config.checkpoint_steps,
    )

    # SystemClass = SYSTEM[config.system]
    SystemClass = discourse_system.DiscourseSystem
    system = SystemClass(config)

    # save_directory = os.path.join(JUICE_DIR, config.exp_name)
    # if not os.path.exists(save_directory):
    #     os.makedirs(save_directory)
    # print("Save directory is {}".format(save_directory))
    # print("Save directory is {}".format(wandb.run.dir))

    bar = RoseProgressBar()
    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=1,
        # checkpoint_callback=ckpt_callback,
        max_epochs=int(config.num_epochs),
        min_epochs=int(config.num_epochs),
        # callbacks=[bar],
        num_sanity_val_steps=2
    )

    trainer.fit(system)

    ## Save the model
    # system.save(directory=wandb.run.dir)

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
    parser.add_argument('--dryrun', default=False, action='store_true')
    parser.add_argument('--overfit', default=False, action='store_true')
    parser.add_argument('--use-gpt2', default=False, action='store_true')
    parser.add_argument('--use-section-ids', default=False, action='store_true')
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', default=0, type=float)
    parser.add_argument('--latent-dim', default=0, type=int)
    parser.add_argument('--num-epochs', default=0, type=int)
    parser.add_argument('--batch-size', default=0, type=int)
    parser.add_argument('--interval', default=0, type=int)
    parser.add_argument('--fpath', default=None, type=str)
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--tm-type', default="movie", type=str)
    parser.add_argument('--tag', default="", type=str)
    parser.add_argument('--encoder', type=str, default="", help='either bert or cl')
    parser.add_argument('--dataset', type=str, required=True, help='recipe, wikisection, wikihow')
    parser.add_argument('--project', default="ou_final", type=str)
    args = parser.parse_args()

    run(args)



