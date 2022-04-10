import numpy as np
import torch

class Reconstruction(object):
    """
        r_T = VAE.Reconstruction(
            obs=obs_T,
            z=z_T,
            decoder=time_model,
            tokenizer=self.train_dataset.tokenizer
        ).get_loss()
    """

    def __init__(self, obs, z, decoder, tokenizer, config):
        """
        """
        super().__init__()
        self.obs = obs
        self.z = z
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.config = config
        self.eps = self.config.model_params.eps

    def get_loss(self):
        input_ids = self.tokenizer(
            self.obs,
            padding=True,
            return_tensors="pt").input_ids
        # Ensure input ids doesn't exceed max context length
        input_ids = input_ids[:, :1024].to(self.decoder.device)
        # From https://discuss.huggingface.co/t/generation-probabilities-how-to-compute-probabilities-of-output-scores-for-gpt2/3175/6 gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:]
        generated_outputs = self.decoder.generate(
            input_ids,
            cl_feats=self.z,
            do_sample=True,
            max_length=input_ids.shape[-1]+1,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True)
        gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:]
        # let's stack the logits generated at each step to a tensor and transform
        # logits to probs
        probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)  # -> shape [3, 15, vocab_size]
        # now we need to collect the probability of the generated token
        # we need to add a dummy dim in the end to make gather work
        gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)

        # Avoid 0
        gen_probs += self.eps

        return torch.log(gen_probs).mean()


class KL(object):
    """
        loss_fn = vae.KL(
            z_0=z_0,
            z_t=z_t,
            z_T=z_T,
            t_=t_s,
            t=ts,
            T=Ts,
            config=self.config
        )
        kl_loss = loss_fn.get_loss()

    """

    def __init__(self, z_0, z_t, z_T, t_, t, T, total_t, fc_mu, fc_var, config):
        """
        """
        super().__init__()
        self.z_0 = z_0
        self.z_t = z_t
        self.z_T = z_T
        self.t0 = t_
        self.t1 = t
        self.t2 = T
        self.total_t = total_t
        self.fc_mu = fc_mu
        self.fc_var = fc_var
        self.config = config
        self.eps = self.config.model_params.eps

        self.z_prior_mu = torch.tensor([0.0], device=z_0.device)
        self.z_prior_logvar = torch.tensor([np.log(1.0)], device=z_0.device)
        self.log_2pi = torch.log(torch.tensor([2*np.pi], device=z_0.device))

    def _log_normal(self, x, mu, logvar):
        var = torch.exp(logvar)
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        return -0.5 * torch.sum(
            logvar + self.log_2pi + torch.square(mu)/var, dim=-1) # (bsz,)

        return -0.5 * torch.sum(
            logvar + self.log_2pi + torch.square(x-mu)/var, dim=-1) # (bsz,)

    def get_loss(self):
        ones = torch.ones(self.z_0.shape, device=self.z_0.device)
        # kl for t = 0
        mu = (self.t0/self.total_t)[:, None] * ones # using density estimate of brownian
        var = (self.t0*(self.total_t -self.t0)/(self.total_t) + self.eps)[:, None]
        kl_0 = (
            self._log_normal(x=self.z_0, mu=self.fc_mu(self.z_0), logvar=self.fc_var(self.z_0))
            - self._log_normal(x=self.z_0, mu=torch.tensor(mu, device=self.z_0.device), logvar=torch.tensor(torch.log(var), device=self.z_0.device))).mean()

        # kl for T
        mu = (self.t2/self.total_t)[:, None] * ones # using density estimate of brownian
        var = (self.t2*(self.total_t -self.t2)/(self.total_t) + self.eps)[:, None]
        kl_T = (
            self._log_normal(x=self.z_T, mu=self.fc_mu(self.z_T), logvar=self.fc_var(self.z_T))
            - self._log_normal(x=self.z_T, mu=torch.tensor(mu, device=self.z_0.device), logvar=torch.tensor(torch.log(var), device=self.z_0.device))).mean()

        # kl for t - define a new bridge
        t = (self.t1-self.t0)
        T = (self.t2-self.t0)
        alpha = t/T
        mu = (1-alpha)[:, None]*self.z_0 + alpha[:, None]*self.z_T # using density estimate of brownian
        var = (t*(T-t)/(t+self.eps))[:, None]
        kl_t = (
            self._log_normal(x=self.z_t, mu=self.fc_mu(self.z_t), logvar=self.fc_var(self.z_t))
            - self._log_normal(x=self.z_t, mu=torch.tensor(mu, device=self.z_0.device), logvar=torch.tensor(torch.log(var), device=self.z_0.device))).mean()

        return kl_0 + kl_T + kl_t
