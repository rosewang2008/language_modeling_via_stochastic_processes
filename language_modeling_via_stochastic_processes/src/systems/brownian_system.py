import torch

from language_modeling_via_stochastic_processes.src.systems import brownian_bridge_system
from language_modeling_via_stochastic_processes.src.objectives import brownian

class BrownianSystem(brownian_bridge_system.BrownianBridgeSystem):

    def __init__(self, config):
        super().__init__(config=config)

    def get_losses_for_batch(self, batch, batch_idx):
        del batch_idx
        torch.cuda.empty_cache()
        obs_0 = batch['y_0']
        obs_t = batch['y_t']
        obs_T = batch['y_T']
        t_s = batch['t_'].float()
        ts = batch['t'].float()
        Ts = batch['T'].float()
        feats_0 = self.get_feats(obs_0)
        feats_t = self.get_feats(obs_t)
        feats_T = self.get_feats(obs_T)
        log_q_y_tp1 = self.model.get_log_q(feats_t)
        loss_fn = brownian.BrownianLoss(
            z_0=feats_0,
            z_t=feats_t,
            z_T=feats_T,
            t_=t_s,
            t=ts,
            T=Ts,
            alpha=0,
            var=0,
            log_q_y_T=log_q_y_tp1,
            loss_type=self.config.loss_params.name,
            eps=self.config.model_params.eps,
            max_seq_len=batch['total_t'].float(),
            config=self.config
        )
        loss = loss_fn.get_loss()
        return loss