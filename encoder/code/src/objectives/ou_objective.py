import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class OULoss(object):
    """Playing around with different losses

    min loss = min -log p(x_{t+1}|x_{t})


    """

    def __init__(self, preds_t, preds_tp1,
                log_q_y_tp1,
                 x_t, x_tp1,
                 dt, sigma, eps,
                 loss_type,
                 C_eta=None,
                 label=None):
        super().__init__()
        NAME2LOSS = {
            'simclr': self.simclr_loss,
            'classification': self.classification_loss
        }

        self.preds_t = preds_t
        self.preds_tp1 = preds_tp1
        self.log_q_y_tp1 = log_q_y_tp1
        self.x_t = x_t
        self.x_tp1 = x_tp1
        self.dt = dt
        self.sigma = sigma
        self.eps = eps
        self.loss_f = NAME2LOSS[loss_type]
        self.normalization = 2*self.dt * self.sigma**2 # 2 \eta
        self.sigmoid = nn.Sigmoid()
        self.label = label

        if C_eta is None:
            C_eta = 0.0
        self.C_eta = C_eta

    def _log_p(self, x_tp1, x_t):
        # time = (1 - self.dt/(1-self.t)).view(-1, 1)
        delta = x_tp1 - x_t * (1-self.dt).view(-1, 1)
        log_p = -1./self.normalization * (delta*delta).sum(-1) + self.C_eta
        if len(log_p.shape) > 1: # (1, bsz)
            log_p = log_p.squeeze(0)
        return log_p

    def _logit(self, x_tp1, x_t):
        """
        Calculating log p(x_tp1, x_t) = -|| h(x_{t+dt}) - h(x_t)(1-dt)||^2_2
        """
        log_p = self._log_p(x_tp1=x_tp1, x_t=x_t)
        log_p = log_p.unsqueeze(-1)
        log_q = self.log_q_y_tp1
        logit = log_p - log_q
        return logit # should be (bsz, 1)

    def simclr_loss(self):
        """
        log p = -1/(2*eta) \| x' - x - \mu(x) \|^2_2 + C_{\eta}

        logit = log p - log q
        """
        loss = 0.0
        # Positive pair
        pos_logit = self._logit(x_tp1=self.preds_tp1, x_t=self.preds_t)
        pos_probs = torch.exp(pos_logit) # (bsz,1)
        for idx in range(self.preds_tp1.shape[0]):
            # Negative pair: logits over all possible contrasts
            neg_i_logit = self._logit(x_tp1=self.preds_tp1, x_t=self.preds_t[idx])
            neg_i_probs = torch.exp(neg_i_logit) # (bsz,1)
            loss_i = -(pos_logit[idx] - torch.log(neg_i_probs.sum() + self.eps))
            loss += loss_i

        loss = loss / self.preds_tp1.shape[0]
        return loss

    def classification_loss(self):
        loss_f = nn.BCELoss()
        logit = self._logit(x_tp1=self.preds_tp1, x_t=self.preds_t)
        logit = logit.squeeze(1) # (bsz,)
        assert self.label is not None
        probs = self.sigmoid(logit)
        loss = loss_f(probs, self.label)
        self.acc = torch.sum(((probs>0.5) == self.label).int())/logit.shape[0]
        return loss

    def get_loss(self):
        return self.loss_f()

class BrownianBridgeEverythingLoss(object):
    """Everything is a brownian bridge...

    p(z_t | mu_0, mu_T) = \mathcal{N}(mu_0 * t/T + mu_T * (1-t/T), I t*(T-t)/T)

    normalization constant: -1/(2 * t*(T-t)/T)

    T and t varies.

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
        # self.normalization = 2*(self.var + self.eps).view(-1, 1)
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
        # if t is not None:
        #     T = self.T -self.t_ # contrasts only vary small t, and not big T
        #     t = self.t[t] - self.t_[t]
        # else:
        #     T = self.T - self.t_
        #     t = self.t - self.t_

        ## TRIAL 1
        # alpha = alpha.view(-1, 1)
        # delta = z_0 * (1-alpha) + z_T * (alpha) - z_t
        # log_p =  -1/norm * (delta*delta).sum(-1) + self.C_eta # (512,)

        ## TRIAL 2
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
        mse_loss_f = nn.MSELoss()
        vals = self.z_0
        start_reg = mse_loss_f(vals, torch.zeros(vals.shape, device=self.device))
        # end
        vals = torch.abs(self.z_T)
        # print('end vals: {}'.format(vals[:5]))
        end_reg = mse_loss_f(vals, torch.ones(vals.shape, device=self.device)*self.end_pin_val)
        return start_reg + end_reg

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
            # TRIAL 1: nominal contrast for random triplet - contrast from in between
            if self.config.contrast == "t":
                neg_i_logit = self._logit(z_0=self.z_0, z_T=self.z_T, z_t=self.z_t[idx],
                                          t_=self.t_, t=self.t[idx], T=self.T)
            # TRIAL 2: contrast from the end point - I just change the tail value, but not the tail indices
            elif self.config.contrast == "T":
                neg_i_logit = self._logit(z_0=self.z_0, z_T=self.z_T[idx], z_t=self.z_t,
                                          t_=self.t_, t=self.t, T=self.T)
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

class BrownianBridgeRandomTLoss(BrownianBridgeEverythingLoss):
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
            label=label
        )
        self.config=config

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
            # vals = self.z_T[end_idxs, :]
            # print('end vals: {}'.format(vals[:5]))
            end_reg = mse_loss_f(vals, torch.ones(vals.shape, device=self.device)*self.end_pin_val)
            loss += end_reg
        return loss


class BrownianRandomTLoss(BrownianBridgeRandomTLoss):
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
            config=config
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


class BrownianBridgeLoss(OULoss):

    def __init__(self, preds_t, preds_tp1,
                log_q_y_tp1,
                 x_t, x_tp1, t,
                 dt, sigma, eps,
                 loss_type,
                 C_eta=None,
                 label=None):
        super().__init__(
            preds_t=preds_t, preds_tp1=preds_tp1,
            log_q_y_tp1=log_q_y_tp1, x_t=x_t, x_tp1=x_tp1,
            dt=dt, sigma=sigma, eps=eps, loss_type=loss_type,
            C_eta=C_eta, label=label
        )
        self.t = t
        self.end_pin_val = 1.0

    def _log_p(self, x_tp1, x_t):
        # x_tp1 = x_t * (1- self.dt/(1. - t)) + (self.dt/(1.-t))*self.B_T + noise
        time = (1 - self.dt/(1-self.t)).view(-1, 1)
        offset = ((self.dt/(1.-self.t))* self.end_pin_val).view(-1, 1)
        # BUG???????????? Aug 19 :'( tragic
        # delta = x_tp1 - time * x_t + offset
        delta = x_tp1 - time * x_t - offset
        log_p = -1./self.normalization * (delta*delta).sum(-1) + self.C_eta # (512,)
        # check scale of normalization and delta
        if len(log_p.shape) > 1: # (1, bsz)
            log_p = log_p.squeeze(0)
        return log_p

class BrownianBridgeRegLoss(BrownianBridgeLoss):
    """
    Dynamics regularization - forward and bakward
    """

    def __init__(self, preds_t, preds_tp1,
                log_q_y_tp1,
                 x_t, x_tp1, t,
                 dt, sigma, eps,
                 loss_type,
                 C_eta=None,
                 label=None):
        super().__init__(
            preds_t=preds_t, preds_tp1=preds_tp1,
            log_q_y_tp1=log_q_y_tp1, x_t=x_t, x_tp1=x_tp1, t=t,
            dt=dt, sigma=sigma, eps=eps, loss_type=loss_type,
            C_eta=C_eta, label=label
        )

    def _log_p_forward(self, x_tp1, x_t):
        # x_tp1 = x_t * (1- self.dt/(1. - t)) + (self.dt/(1.-t))*self.B_T + noise
        time = (1 - self.dt/(1-self.t)).view(-1, 1)
        offset = ((self.dt/(1.-self.t))* self.end_pin_val).view(-1, 1)
        delta = x_tp1 - time * x_t - offset
        log_p = (-1./self.normalization * (delta*delta).sum(-1) + self.C_eta) # (512,)
        if len(log_p.shape) > 1: # (1, bsz)
            log_p = log_p.squeeze(0)
        return log_p

    def _log_p_backward(self, x_tp1, x_t):
        # x_tp1 = x_t * (1- self.dt/(1. - t)) + (self.dt/(1.-t))*self.B_T + noise
        time = (1 - self.dt/(1-self.t)).view(-1, 1)
        offset = ((self.dt/(1.-self.t))* self.end_pin_val).view(-1, 1)
        delta = time * x_t + offset - x_tp1
        log_p = (-1./self.normalization * (delta*delta).sum(-1) + self.C_eta) # (512,)
        if len(log_p.shape) > 1: # (1, bsz)
            log_p = log_p.squeeze(0)
        return log_p

    def _logit(self, x_tp1, x_t, mode):
        """
        Calculating log p(x_tp1, x_t) = -|| h(x_{t+dt}) - h(x_t)(1-dt)||^2_2
        """
        if mode == 'forward':
            log_p = self._log_p_forward(x_tp1=x_tp1, x_t=x_t)
        else:
            log_p = self._log_p_backward(x_tp1=x_tp1, x_t=x_t)
        log_p = log_p.unsqueeze(-1)
        log_q = self.log_q_y_tp1
        logit = log_p - log_q
        return logit # should be (bsz, 1)

    def simclr_(self, mode):
        loss = 0.0
        # Positive pair
        pos_logit = self._logit(x_tp1=self.preds_tp1, x_t=self.preds_t, mode=mode)
        pos_probs = torch.exp(pos_logit) # (bsz,1)
        for idx in range(self.preds_tp1.shape[0]):
            # Negative pair: logits over all possible contrasts
            neg_i_logit = self._logit(x_tp1=self.preds_tp1, x_t=self.preds_t[idx], mode=mode)
            neg_i_probs = torch.exp(neg_i_logit) # (bsz,1)
            loss_i = -(pos_logit[idx] - torch.log(neg_i_probs.sum() + self.eps))
            loss += loss_i
        loss = loss / self.preds_tp1.shape[0]
        return loss


    def simclr_loss(self):
        """
        log p = -1/(2*eta) \| x' - x - \mu(x) \|^2_2 + C_{\eta}

        logit = log p - log q
        """
        forward = self.simclr_(mode='forward')
        backward = self.simclr_(mode='backward')
        loss = forward + backward
        return loss


class BrownianBridgeRegPinLoss(BrownianBridgeLoss):
    """
    Single dyanmics + reg from pinning start and end
    """

    def __init__(self, preds_t, preds_tp1,
                log_q_y_tp1,
                 x_t, x_tp1, t,
                 dt, sigma, eps,
                 loss_type,
                 C_eta=None,
                 label=None):
        super().__init__(
            preds_t=preds_t, preds_tp1=preds_tp1,
            log_q_y_tp1=log_q_y_tp1, x_t=x_t, x_tp1=x_tp1, t=t,
            dt=dt, sigma=sigma, eps=eps, loss_type=loss_type,
            C_eta=C_eta, label=label
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def simclr_loss(self):
        """
        log p = -1/(2*eta) \| x' - x - \mu(x) \|^2_2 + C_{\eta}

        logit = log p - log q
        """
        loss = 0.0
        # Positive pair
        pos_logit = self._logit(x_tp1=self.preds_tp1, x_t=self.preds_t)
        pos_probs = torch.exp(pos_logit) # (bsz,1)
        for idx in range(self.preds_tp1.shape[0]):
            # Negative pair: logits over all possible contrasts
            neg_i_logit = self._logit(x_tp1=self.preds_tp1, x_t=self.preds_t[idx])
            neg_i_probs = torch.exp(neg_i_logit) # (bsz,1)
            loss_i = -(pos_logit[idx] - torch.log(neg_i_probs.sum() + self.eps))
            loss += loss_i

        loss = loss / self.preds_tp1.shape[0]

        # Regularization for pinning start and end of bridge
        mse_loss_f = nn.MSELoss()
        start_idxs = torch.where((self.dt + self.t) == self.dt)[0]
        if start_idxs.nelement():
            vals = self.preds_tp1[start_idxs, :]
            start_reg = mse_loss_f(vals, torch.zeros(vals.shape, device=self.device))
            loss += start_reg
        end_idxs = torch.where((self.dt + self.t) == self.end_pin_val)[0]
        if end_idxs.nelement():
            vals = self.preds_tp1[end_idxs, :]
            end_reg = mse_loss_f(vals, torch.ones(vals.shape, device=self.device)*self.end_pin_val)
            loss += end_reg
        return loss

class BrownianBridgeRegBothLoss(BrownianBridgeRegLoss):

    def __init__(self, preds_t, preds_tp1,
                log_q_y_tp1,
                 x_t, x_tp1, t,
                 dt, sigma, eps,
                 loss_type,
                 C_eta=None,
                 label=None):
        super().__init__(
            preds_t=preds_t, preds_tp1=preds_tp1,
            log_q_y_tp1=log_q_y_tp1, x_t=x_t, x_tp1=x_tp1, t=t,
            dt=dt, sigma=sigma, eps=eps, loss_type=loss_type,
            C_eta=C_eta, label=label
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def simclr_loss(self):
        """
        log p = -1/(2*eta) \| x' - x - \mu(x) \|^2_2 + C_{\eta}

        logit = log p - log q
        """
        forward = self.simclr_(mode='forward')
        backward = self.simclr_(mode='backward')
        loss = forward + backward

        # Regularization for pinning start and end of bridge
        mse_loss_f = nn.MSELoss()
        start_idxs = torch.where((self.dt + self.t) == self.dt)[0]
        if start_idxs.nelement():
            vals = self.preds_tp1[start_idxs, :]
            start_reg = mse_loss_f(vals, torch.zeros(vals.shape, device=self.device))
            loss += start_reg
        end_idxs = torch.where((self.dt + self.t) == self.end_pin_val)[0]
        if end_idxs.nelement():
            vals = self.preds_tp1[end_idxs, :]
            end_reg = mse_loss_f(vals, torch.ones(vals.shape, device=self.device)*self.end_pin_val)
            loss += end_reg
        return loss



class NoisyOULoss(OULoss):

    def __init__(self, preds_t, preds_tp1,
                log_q_y_tp1,
                 x_t, x_tp1,
                z_t, z_mu_t, z_logvar_t,
                z_tp1, z_mu_tp1, z_logvar_tp1,
                 dt, sigma, eps,
                 loss_type,
                 C_eta=None,
                 label=None):
        super().__init__(
            preds_t=preds_t, preds_tp1=preds_tp1,
            log_q_y_tp1=log_q_y_tp1, x_t=x_t, x_tp1=x_tp1,
            dt=dt, sigma=sigma, eps=eps, loss_type=loss_type,
            C_eta=C_eta, label=label
        )
        self.z_t = z_t
        self.z_tp1 = z_tp1
        self.z_mu_t = z_mu_t
        self.z_logvar_t = z_logvar_t
        self.z_mu_tp1 = z_mu_tp1
        self.z_logvar_tp1 = z_logvar_tp1
        self.log_2pi = torch.log(torch.tensor([2*np.pi], device=self.z_t.device))

        if len(self.log_q_y_tp1.shape) > 2:
            self.log_q_y_tp1 = self.log_q_y_tp1[:, -1]

        self.f = nn.MSELoss(reduction='none')


    def _log_p(self, x_tp1, x_t, x_tilde_tp1, x_tilde_t, z_tp1, z_t):
        delta = z_tp1 - z_t * (1-self.dt)
        # p(z', z)
        log_p = -1./self.normalization * (delta*delta).sum(-1) + self.C_eta

        if len(log_p.shape) > 1: # (1, bsz)
            log_p = log_p.squeeze(0)
        if len(x_tp1.shape) > 2: # bsz, seq_len, data_dim
            x_tp1 = x_tp1[:, -1, :]
        if len(x_t.shape) > 2: # bsz, seq_len, data_dim
            x_t = x_t[:, -1, :]

        # p(x|z), p(x'|z')
        reconstruct_tp1 = self.f(x_tilde_tp1, x_tp1).sum(-1)
        reconstruct_t = self.f(x_tilde_t, x_t).sum(-1)
        return log_p + reconstruct_t + reconstruct_tp1

    def _logit(self, x_tp1, x_t, x_tilde_tp1, x_tilde_t,
               z_tp1, z_t, z_mu_tp1, z_mu_t, z_logvar_tp1, z_logvar_t):
        """
        log p \geq \mathbb{E}_{q}[\ln p(x'| z')p(x| z)p(z', z) ] - \mathbb{E}_{q}[\ln q(z)]

        log q = over contrasts
        """
        log_p = self._log_p(
            x_tp1=x_tp1, x_t=x_t,
            x_tilde_tp1=x_tilde_tp1, x_tilde_t=x_tilde_t,
            z_tp1=z_tp1, z_t=z_t)
        # \ln p(z'|x') p(z|x)
        log_q_z = self._log_normal(x=z_tp1, mu=z_mu_tp1, logvar=z_logvar_tp1) + self._log_normal(
                    x=z_t, mu=z_mu_t, logvar=z_logvar_t)

        # contrast distribution
        log_q = self.log_q_y_tp1
        if len(log_p.shape) == 1:
            log_p = log_p.unsqueeze(-1)
        if len(log_q.shape) == 1:
            log_q = log_q.unsqueeze(-1)
        logit = log_p - log_q
        return logit # should be (bsz, 1)

    def _log_normal(self, x, mu, logvar):
        var = torch.exp(logvar)
        return -0.5 * torch.sum(
            logvar + self.log_2pi + torch.square(x-mu)/var, dim=-1) # (bsz,)

    def simclr_loss(self):
        """
        log p = -1/(2*eta) \| x' - x - \mu(x) \|^2_2 + C_{\eta}

        logit = log p - log q
        """
        loss = 0.0
        pos_logit = self._logit(
            x_tp1=self.x_tp1, x_t=self.x_t,
            x_tilde_tp1=self.preds_tp1, x_tilde_t=self.preds_t,
            z_tp1=self.z_tp1, z_t=self.z_t,
            z_mu_tp1=self.z_mu_tp1, z_logvar_tp1=self.z_logvar_tp1,
            z_mu_t=self.z_mu_t, z_logvar_t=self.z_logvar_t
        )
        pos_probs = torch.exp(pos_logit) # (bsz,1)

        for idx in range(self.preds_tp1.shape[0]):
            # Negative pair: logits over all possible contrasts
            neg_i_logit = self._logit(
                x_tp1=self.x_tp1, x_t=self.x_t[idx, -1],
                x_tilde_tp1=self.preds_tp1, x_tilde_t=self.preds_t[idx, ],
                z_tp1=self.z_tp1, z_t=self.z_t[idx, -1],
                z_mu_tp1=self.z_mu_tp1, z_logvar_tp1=self.z_logvar_tp1,
                z_mu_t=self.z_mu_t, z_logvar_t=self.z_logvar_t
            )

            neg_i_probs = torch.exp(neg_i_logit) # (bsz,1)
            loss_i = -(pos_logit[idx] - torch.log(neg_i_probs.sum() + self.eps))
            loss += loss_i
        loss = loss / self.preds_tp1.shape[0]
        return loss

    def classification_loss(self):
        # TODO edit x's into z's for logit
        loss_f = nn.BCELoss()
        logit = self._logit(x_tp1=self.preds_tp1, x_t=self.preds_t)
        logit = logit.squeeze(1) # (bsz,)
        assert self.label is not None
        probs = self.sigmoid(logit)
        loss = loss_f(probs, self.label)
        self.acc = torch.sum(((probs>0.5) == self.label).int())/logit.shape[0]
        return loss

