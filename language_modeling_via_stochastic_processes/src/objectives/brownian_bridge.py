import torch
from torch import nn


class BrownianBridgeLoss(object):
    """Everything is a brownian bridge...

    p(z_t | mu_0, mu_T) = \mathcal{N}(mu_0 * t/T + mu_T * (1-t/T), I t*(T-t)/T)

    normalization constant: -1/(2 * t*(T-t)/T)
    """

    def __init__(self,
                 z_0, z_t, z_T,
                 t_, t, T,
                 alpha, var,
                log_q_y_T,
                 loss_type,
                 eps,
                 max_seq_len,
                 C_eta=None,
                 label=None):
        super().__init__()
        self.log_q_y_T = log_q_y_T
        self.z_0 = z_0
        self.z_t = z_t
        self.z_T = z_T
        self.t_ = t_
        self.t = t
        self.T = T
        self.alpha = alpha
        self.var = var
        NAME2LOSS = {
            'simclr': self.simclr_loss,
        }
        self.loss_f = NAME2LOSS[loss_type]
        self.eps= eps
        self.max_seq_len = max_seq_len
        self.sigmoid = nn.Sigmoid()
        self.label = label

        if C_eta is None:
            C_eta = 0.0
        self.C_eta = C_eta
        self.end_pin_val = 1.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _log_p(self, z_0, z_t, z_T, t_0, t_1, t_2):
        T = t_2-t_0
        t = t_1-t_0

        alpha = (t/(T+self.eps)).view(-1, 1)
        delta = z_0 * (1-alpha) + z_T * (alpha) - z_t
        var = (t * (T - t)/ (T + self.eps))
        log_p =  -1/(2*var + self.eps) * (delta*delta).sum(-1) + self.C_eta # (512,)
        if len(log_p.shape) > 1: # (1, bsz)
            log_p = log_p.squeeze(0)
        return log_p

    def _logit(self, z_0, z_T, z_t, t_, t, T):
        """
        Calculating log p(z_tp1, z_t) = -|| h(z_{t+dt}) - h(z_t)(1-dt)||^2_2
        """
        log_p = self._log_p(z_0=z_0, z_t=z_t, z_T=z_T,
                            t_0=t_, t_1=t, t_2=T)
        log_p = log_p.unsqueeze(-1)
        log_q = self.log_q_y_T
        logit = log_p # - log_q
        return logit # should be (bsz, 1)

    def reg_loss(self):
        loss = 0.0
        mse_loss_f = nn.MSELoss()
        # start reg
        start_idxs = torch.where((self.t_) == 0)[0]
        if start_idxs.nelement():
            vals = self.z_0[start_idxs, :]
            start_reg = mse_loss_f(vals, torch.zeros(vals.shape, device=self.device))
            loss += start_reg
        # end reg
        end_idxs = torch.where((self.T) == self.max_seq_len - 1)[0]
        if end_idxs.nelement():
            vals = torch.abs(self.z_T[end_idxs, :])
            end_reg = mse_loss_f(vals, torch.ones(vals.shape, device=self.device)*self.end_pin_val)
            loss += end_reg
        return loss


    def simclr_loss(self):
        """
        log p = -1/(2*eta) \| x' - x - \mu(x) \|^2_2 + C_{\eta}

        logit = log p - log q
        """
        loss = 0.0
        # Positive pair
        pos_logit = self._logit(z_0=self.z_0, z_T=self.z_T, z_t=self.z_t,
                                t_=self.t_, t=self.t, T=self.T)
        pos_probs = torch.exp(pos_logit) # (bsz,1)
        for idx in range(self.z_T.shape[0]):
            # Negative pair: logits over all possible contrasts
            # Nominal contrast for random triplet - contrast from in between
            neg_i_logit = self._logit(
                z_0=self.z_0, z_T=self.z_T, z_t=self.z_t[idx],
                t_=self.t_, t=self.t[idx], T=self.T)
            neg_i_probs = torch.exp(neg_i_logit) # (bsz,1)
            loss_i = -(pos_logit[idx] - torch.log(neg_i_probs.sum() + self.eps))
            loss += loss_i

        loss = loss / self.z_T.shape[0]
        # Regularization for pinning start and end of bridge
        reg_loss = self.reg_loss()
        loss += reg_loss
        return loss

    def get_loss(self):
        return self.loss_f()
