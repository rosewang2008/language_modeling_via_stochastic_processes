import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2TimeLMHeadModel
import wandb

from language_modeling_via_stochastic_processes.src.objectives import vae
from language_modeling_via_stochastic_processes.src.systems import brownian_bridge_system

torch.autograd.set_detect_anomaly(True)


class VAESystem(brownian_bridge_system.BrownianBridgeSystem):

    def __init__(self, config):
        super().__init__(config=config)
        model_type = GPT2TimeLMHeadModel
        gpt2_config = GPT2Config()
        gpt2_config.use_contrastive_embeddings = True
        gpt2_config.debug_ids = False
        gpt2_config.embedding_type = "entireSection"
        gpt2_config.use_section_ids = False # TODO I think this needs to be True
        gpt2_config.use_section_null = False
        gpt2_config.use_noisy_embeddings = False
        gpt2_config.max_num_sections = len(self.train_dataset.section_names)
        gpt2_config.dataset_name = self.config.data_params.name.lower()
        gpt2_config.cl_latent_dim = self.config.model_params.latent_dim
        self.time_model = model_type.from_pretrained('gpt2', config=gpt2_config)
        self.time_model.resize_token_embeddings(len(self.train_dataset.tokenizer))

        self.model.fc_mu = nn.Linear(self.config.model_params.latent_dim, self.config.model_params.latent_dim)
        self.model.fc_var = nn.Linear(self.config.model_params.latent_dim, self.config.model_params.latent_dim)

    def get_losses_for_batch(self, batch, batch_idx):
        torch.cuda.empty_cache()
        # each batch has x_{t}, x_{t-1}, x_{t-2},
        # \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p(x_0|z_0)p(x_t|z_t)p(x_T|z_T)]
        # D_{\text{KL}}(q(z_t|z_0, z_T, x_t) \| p(z_t|z_0, z_T))
        # D_{\text{KL}}(q(z_0|x_0) \| p(z_0))
        # D_{\text{KL}}(q(z_T|x_T) \| p(z_T))

        obs_0 = batch['y_0']
        obs_t = batch['y_t']
        obs_T = batch['y_T']
        total_T = batch['total_t']
        t_s = batch['t_'].float()
        ts = batch['t'].float()
        Ts = batch['T'].float()
        z_0 = self.get_feats(obs_0)
        z_t = self.get_feats(obs_t)
        z_T = self.get_feats(obs_T)

        r_0 = vae.Reconstruction(
            obs=obs_0,
            z=z_0,
            decoder=self.time_model,
            config=self.config,
            tokenizer=self.train_dataset.tokenizer
        ).get_loss()

        r_t = vae.Reconstruction(
            obs=obs_t,
            z=z_t,
            decoder=self.time_model,
            config=self.config,
            tokenizer=self.train_dataset.tokenizer
        ).get_loss()

        r_T = vae.Reconstruction(
            obs=obs_T,
            z=z_T,
            decoder=self.time_model,
            config=self.config,
            tokenizer=self.train_dataset.tokenizer
        ).get_loss()

        loss_fn = vae.KL(
            z_0=z_0,
            z_t=z_t,
            z_T=z_T,
            t_=t_s,
            t=ts,
            T=Ts,
            total_t=total_T,
            fc_mu=self.model.fc_mu,
            fc_var=self.model.fc_var,
            config=self.config
        )
        kl_loss = loss_fn.get_loss()
        loss = r_0 + r_t + r_T + kl_loss

        wandb.log({'kl': kl_loss, 'r0': r_0, 'rt': r_t, 'rT': r_T})

        return loss
