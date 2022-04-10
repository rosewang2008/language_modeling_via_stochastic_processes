from language_modeling_via_stochastic_processes.src.objectives import brownian_bridge

class BrownianLoss(brownian_bridge.BrownianBridgeLoss):
    def __init__(self,
                 z_0, z_t, z_T, t_, t, T,
                 alpha, var,
                log_q_y_T,
                 loss_type,
                 eps,
                 max_seq_len,
                 config,
                 C_eta=None,
                 label=None):
        super().__init__(
            z_0=z_0, z_t=z_t, z_T=z_T,
            t_=t_, t=t, T=T, alpha=alpha, var=var,
            log_q_y_T=log_q_y_T, loss_type=loss_type,
            eps=eps, max_seq_len=max_seq_len,
            C_eta=C_eta,
            label=label,
        )

    def reg_loss(self):
        return 0.0

    def _log_p(self, z_0, z_t, z_T, t_0, t_1, t_2):
        delta = z_T - z_t
        var = t_2 - t_1
        log_p =  -1/(2*var + self.eps) * (delta*delta).sum(-1) + self.C_eta # (512,)
        if len(log_p.shape) > 1: # (1, bsz)
            log_p = log_p.squeeze(0)
        return log_p
