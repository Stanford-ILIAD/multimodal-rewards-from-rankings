import torch
import torch.nn.functional as F
import random
import numpy as np
import tqdm as tqdm
import torch.distributions as td
import scipy.optimize
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools
from abc import ABC, abstractmethod

plt.style.use('seaborn-whitegrid')


class Agent(ABC):
    @abstractmethod
    def act(self, D):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    def __init__(self, items, scale, slate, num_mix, mh_samples, mh_iterations, mc_iterations, mc_batch, device):
        self.device = device
        items = items - items.mean(dim=0, keepdim=True)
        items = items / items.std(dim=0, keepdim=True)
        items[torch.isnan(items)] = 0.
        self.items = items.to(device)
        self.scale = scale
        self.dim = items.size(1)
        self.slate = slate
        self.p_v = td.Normal(torch.zeros(num_mix, self.dim, device=self.device),
                             scale * torch.ones(num_mix, self.dim, device=self.device))
        self.p_mix = td.Exponential(torch.tensor(num_mix * [1.], device=self.device))
        self.mh_samples = mh_samples
        self.mh_iterations = mh_iterations
        self.mc_batch = mc_batch
        self.mc_iterations = mc_iterations
        self.num_mix = num_mix

    # sample observations D from mixtures batch v for the provided slates
    def sample_D(self, v, mixture, slates):
        assert v.size(0) == slates.size(0)
        mix_idx = td.Categorical(mixture).sample()
        v_sel = v.gather(
            1, mix_idx[..., None, None].expand(-1, 1, self.dim)
        ).squeeze(1)
        v_ = torch.einsum('brk,bk->br', slates, v_sel)
        D_idx = torch.empty(v.size(0), 0, dtype=torch.long, device=self.device)
        for _ in range(slates.size(1)):
            sel = td.Categorical(logits=v_).sample()[:, None]
            D_idx = torch.cat([D_idx] + [sel], dim=1)
            v_.scatter_(-1, D_idx, -np.inf)
        return slates.gather(
            1, D_idx.unsqueeze(-1).expand_as(slates)
        )

    # Get MAP v given D
    def pred_v(self, D, M=50, itr=20):
        v_ = self.sample_prior(M).requires_grad_()
        m_ = self.sample_mixture(M).requires_grad_()
        opt = torch.optim.LBFGS([v_, m_])

        def closure():
            m_exp = (m_ - m_.logsumexp(dim=-1, keepdim=True)).exp()
            loss = -self.log_p_v(v_, m_exp, D).sum()
            opt.zero_grad()
            loss.backward()
            return loss

        for _ in range(itr):
            opt.step(closure)

        m_ = (m_ - m_.logsumexp(dim=-1, keepdim=True)).exp()
        idx = self.log_p_v(v_, m_, D).argmax()
        return v_[idx].detach(), m_[idx].detach()

    # we assume a prior of the unit n-Gaussian for v
    def sample_prior(self, *batch_shape):
        return self.p_v.sample(sample_shape=batch_shape)

    # we assume a uniform prior over n-simplex for m
    def sample_mixture(self, *batch_shape):
        v_ = self.p_mix.sample(sample_shape=batch_shape)
        return v_ / v_.sum(dim=-1, keepdim=True)

    # sample `batch` elements from p(v|D) using Metropolis-Hastings
    def sample_v(self, D, batch=1, temp=0.15, horizon=150):
        v_curr = self.sample_prior(batch)
        m_curr = self.sample_mixture(batch)
        log_p_curr = self.log_p_v(v_curr, m_curr, D)
        for _ in range(horizon):
            noise = temp * self.sample_prior(batch)
            v_prop = v_curr + noise
            m_prop = self.sample_mixture(batch)
            log_p_prop = self.log_p_v(v_prop, m_prop, D)
            log_alpha = log_p_prop - log_p_curr
            accepted = torch.rand([batch], device=self.device) <= log_alpha.exp()
            v_curr[accepted, :] = v_prop[accepted, :]
            m_curr[accepted, :] = m_prop[accepted, :]
            log_p_curr[accepted] = log_p_prop[accepted]
        return v_curr, m_curr

    # return log probability of batch of rankings D under mixture batch v
    def log_p_rank(self, D, v, mixture):
        assert D.size(0) == v.size(0)
        if D.size(0) == 0: return torch.zeros([0], device=self.device)
        v_ = torch.einsum('brk,bmk->bmr', D, v)
        denom = v_.flip(-1).logcumsumexp(dim=-1).flip(-1)
        probs = (v_ - denom).sum(dim=-1)
        assert mixture.shape == probs.shape
        return mixture.add(1e-4).log().add(probs).logsumexp(dim=-1)

    # return log probability of rankings D under mixture batch v
    def log_p_D(self, D, v, mixture):
        return self.log_p_rank(
            D[None, :, ...].expand(v.size(0), *D.shape)
                .flatten(end_dim=1),
            v[:, None, ...].expand(v.size(0), D.size(0), *v.shape[1:])
                .flatten(end_dim=1),
            mixture[:, None, ...].expand(v.size(0), D.size(0), *mixture.shape[1:])
                .flatten(end_dim=1),
        ).view(v.size(0), D.size(0)).sum(dim=1)

    # return p(D|v)p(v) for v, D in batch
    def log_p_v(self, v, mixture, D):
        return self.log_p_D(D, v, mixture) + self.p_v.log_prob(v).sum(dim=-1).sum(dim=-1)

    # sample from posterior using standard hyperparameters
    def post_sample(self, D):
        return self.sample_v(D, batch=self.mh_samples, horizon=self.mh_iterations)

    # return p(D_|D) where D is the set of past observations and D_ is a single observation
    def log_p_obs(self, D_, D, post_sample=None):
        v_, m_ = (
            self.sample_v(D, batch=self.mh_samples, horizon=self.mh_iterations)
            if post_sample is None
            else post_sample
        )
        return self.log_p_D(D_.unsqueeze(0), v_, m_).mean()

    # compute entropy (up to a constant) over v given batch of
    # actions, observations D, and samples v_~p(v|D)
    def H_v(self, actions, v_, m_):
        assert list(v_.shape)[1:] == [self.num_mix, self.dim]
        assert list(m_.shape)[1:] == [self.num_mix]
        assert v_.size(0) == m_.size(0)
        A = actions.size(0)
        M = v_.size(0)
        v_expand = v_[None, ...].expand(A, M, -1, -1).flatten(end_dim=1)
        m_expand = m_[None, ...].expand(A, M, -1).flatten(end_dim=1)
        actions_expand = actions[:, None, ...].expand(A, M, -1, -1).flatten(end_dim=1)
        D_ = self.sample_D(v_expand, m_expand, actions_expand)
        D_v = D_.reshape(A, M, 1, self.slate, self.dim).expand(A, M, M, -1, -1).flatten(end_dim=2)
        v_D = v_expand.reshape(A, 1, M, self.num_mix, -1).expand(A, M, M, -1, -1).flatten(end_dim=2)
        m_D = m_expand.reshape(A, 1, M, self.num_mix).expand(A, M, M, -1).flatten(end_dim=2)
        log_like = self.log_p_rank(D_v, v_D, m_D).reshape(A, M, M)
        h_samp = log_like.exp().mean(dim=-1).log() - torch.diagonal(log_like, dim1=1, dim2=2)
        return h_samp.mean(dim=-1)

    def random_act(self, D):
        return torch.stack([
            *random.sample(list(self.items), self.slate)
        ])


class InfoAgent(Agent):

    # select batch of indices to compare by maximizing information
    def act(self, D):
        v_, m_ = self.sample_v(D, batch=self.mh_samples, horizon=self.mh_iterations)
        action = self.mcmc_sa(self.items, v_, m_, batch=self.mc_batch, horizon=self.mc_iterations)
        return action

    # perturb each element of a batch of actions each represented as the first
    # 'slate' elements of a permutation of [0..n-1] by swapping out one element
    # of each action for a random other element
    def mcmc_transition(self, actions_ext):
        i = torch.randint(0, self.slate, [actions_ext.size(0), 1], device=self.device)
        j = torch.randint(self.slate, actions_ext.size(1), [actions_ext.size(0), 1], device=self.device)
        return (
            actions_ext
                .scatter(1, i, actions_ext.gather(1, j))
                .scatter(1, j, actions_ext.gather(1, i))
        )

    # Lookup actions in embedding keyed by index
    def lookup_actions(self, items, action_idx):
        items_ = items[None, None, :, :].expand(action_idx.size(0), self.slate, -1, -1)
        action_idx_ = action_idx[:, :self.slate, None, None].expand(-1, -1, 1, items.size(-1))
        selected = items_.gather(2, action_idx_).squeeze(2)
        return selected

    # run simulated annealing to get a minimal-entropy action using
    # the provided parameters given samples v_~p(v|D); 'batch'
    # simulations are run in parallel and the lowest entropy action
    # found across all simulations is returned
    def mcmc_sa(self, items, v_, m_, batch=1, horizon=10, T=10., cooling=0.9):
        n = items.size(0)
        actions_cur = torch.rand([batch, n], device=self.device).sort(dim=1).indices
        H_cur = self.H_v(self.lookup_actions(items, actions_cur), v_, m_)
        for i in tqdm.trange(horizon):
            best_cur, best_idx = torch.min(H_cur, dim=0)
            if not i or best_cur < best_H:
                best_H = best_cur
                best_action = self.lookup_actions(items, actions_cur)[best_idx]
            actions_prop = self.mcmc_transition(actions_cur)
            H_prop = self.H_v(self.lookup_actions(items, actions_cur), v_, m_)
            accept_prob = torch.exp((H_cur - H_prop) / T)
            accepted = torch.rand([batch], device=self.device) < accept_prob
            actions_cur[accepted] = actions_prop[accepted]
            H_cur[accepted] = H_prop[accepted]
            T *= cooling
        return best_action

    @property
    def name(self):
        return f"information-{self.num_mix}"


class VolumeAgent(InfoAgent):

    # compute entropy (up to a constant) over v given batch of
    # actions, observations D, and samples v_~p(v|D)
    def H_v(self, actions, v_, m_):
        assert list(v_.shape)[1:] == [self.num_mix, self.dim]
        assert list(m_.shape)[1:] == [self.num_mix]
        assert v_.size(0) == m_.size(0)
        A = actions.size(0)
        M = v_.size(0)
        v_expand = v_[None, ...].expand(A, M, -1, -1).flatten(end_dim=1)
        m_expand = m_[None, ...].expand(A, M, -1).flatten(end_dim=1)
        actions_expand = actions[:, None, ...].expand(A, M, -1, -1).flatten(end_dim=1)
        D_ = self.sample_D(v_expand, m_expand, actions_expand)
        D_v = D_.reshape(A, M, 1, self.slate, self.dim).expand(A, M, M, -1, -1).flatten(end_dim=2)
        v_D = v_expand.reshape(A, 1, M, self.num_mix, -1).expand(A, M, M, -1, -1).flatten(end_dim=2)
        m_D = m_expand.reshape(A, 1, M, self.num_mix).expand(A, M, M, -1).flatten(end_dim=2)
        log_like = self.log_p_rank(D_v, v_D, m_D).reshape(A, M, M)
        h_samp = log_like.exp().mean(dim=-1)
        return h_samp.mean(dim=-1)


    @property
    def name(self):
        return f"volume-{self.num_mix}"


class RandomAgent(Agent):

    # select random slate to compare
    def act(self, D):
        return self.random_act(D)

    @property
    def name(self):
        return f"random-{self.num_mix}"
