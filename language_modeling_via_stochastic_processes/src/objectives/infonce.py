import torch

class InfoNCE(object):

    def __init__(self, c_t, z_tpk, W_k, config):
        super().__init__()
        self.c_t = c_t
        self.z_tpk = z_tpk
        self.W_k = W_k
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = self.config.model_params.eps

    def _logit(self, z_tpk, c_t):
        """
        z_{t+k}^T W_k c_t
        """
        c_t_ = self.W_k(c_t)
        if len(c_t_.shape) > 1:
            bsz = z_tpk.shape[0]
            feat_dim = z_tpk.shape[-1]
            logit = torch.bmm(z_tpk.view(bsz, 1, feat_dim), c_t_.view(bsz, feat_dim, 1))
            logit = logit.squeeze(1)
        else:
            logit = z_tpk @ c_t_
            logit = logit.unsqueeze(1)

        return logit # should be (bsz, 1)

    def loss_f(self):
        """
        log p = -1/(2*eta) \| x' - x - \mu(x) \|^2_2 + C_{\eta}

        logit = log p - log q
        """
        loss = 0.0
        pos_logit = self._logit(z_tpk=self.z_tpk, c_t=self.c_t)
        pos_probs = torch.exp(pos_logit) # (bsz,1)
        for idx in range(self.z_tpk.shape[0]):
            # contast c_t with all other z's
            neg_i_logit = self._logit(z_tpk=self.z_tpk, c_t=self.c_t[idx])
            neg_i_probs = torch.exp(neg_i_logit) # (bsz,1)
            loss_i = -(pos_logit[idx] - torch.log(neg_i_probs.sum() + self.eps))
            loss += loss_i
        loss = loss / self.z_tpk.shape[0]
        return loss

    def get_loss(self):
        return self.loss_f()

