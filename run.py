import torch
import random
import numpy as np
import tqdm as tqdm
import torch.distributions as td
import scipy.optimize
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools
from abc import ABC, abstractmethod
import argparse
from environments import *
from agents import *


# we assume a prior of the unit n-Gaussian for v
def sample_prior(*batch_shape):
    return p_v.sample(sample_shape=batch_shape)

# we assume a uniform prior over n-simplex for m
def sample_mixture(*batch_shape):
    v_ = p_mix.sample(sample_shape=batch_shape)
    return v_ / v_.sum(dim=-1, keepdim=True)

# return a trajectory of observations from running the
# provided agent against the Plackett-Luce model v_true
def run_agent(agent, v_true, m_true, horizon=15):
    D = torch.empty(0, slate, dim, device=device)
    mses = []
    logps = []
    for _ in tqdm.trange(horizon):
        a = agent.act(D)
        assert a.shape == (slate, dim)
        D_ = agent.sample_D(v_true[None, ...], m_true[None, ...], a[None, ...])
        D = torch.cat([D] + [D_], dim=0)
        v_pred, m_pred = agent.pred_v(D)
        mses.append(mse_metric(v_pred, v_true))
        logps.append(logp_metric(D, v_true, m_true))
    return torch.stack(mses), torch.stack(logps), (v_pred, m_pred), (v_true, m_true)

def mse_metric(v_pred, v_true):
    pdist_mat = torch.norm(
        v_pred.unsqueeze(0) - v_true.unsqueeze(1),
        p=2,
        dim=-1
    )
    cost_matrix = (pdist_mat ** 2).cpu()
    rows, cols = scipy.optimize.linear_sum_assignment(cost_matrix)
    return cost_matrix[rows, cols].sum()

def logp_metric(D, v_true, m_true, M=20):
    logps = []
    agent = random_agent
    post_sample = agent.post_sample(D)
    for _ in range(M):
        slate = agent.random_act(D)
        D_ = agent.sample_D(v_true[None, ...], m_true[None, ...], slate[None, ...])[0]
        logps.append(agent.log_p_obs(D_, D, post_sample=post_sample))
    return torch.stack(logps).mean()

# compare the performance of agents
def compare(*agents):
    v_true = sample_prior()
    m_true = sample_mixture()
    result = {}
    for agent in agents:
        run_result = run_agent(agent, v_true, m_true)
        result[agent.name + '-mse'] = run_result[0]
        result[agent.name + '-logp'] = run_result[1]
        result[agent.name + '-pred'] = run_result[2]
        result[agent.name + '-true'] = run_result[3]
    return result

class SyntheticEnvironment:
    features = torch.cat(
        [torch.normal(mean=0., std=1., size=[10, 3])]
        + [torch.normal(mean=0., std=1e-1, size=[100, 3])]
        + [torch.normal(mean=0., std=1e-2, size=[1000, 3])],
        dim=0,
    )


environments = dict(lunar=Lunar, fetch=Fetch, synthetic=SyntheticEnvironment)
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='environment for runs')
    args = parser.parse_args()

    plt.style.use('seaborn-whitegrid')

    device = torch.device('cpu')
    items = environments[args.data.lower()].features
    dim = items.size(1)
    scale = 1.
    slate = 6
    num_mix = 3
    hparams = dict(
            scale=scale, 
            slate=slate,
            num_mix=num_mix, 
            mh_samples=100, 
            mh_iterations=200, 
            mc_iterations=30, 
            mc_batch=10, 
            device=device,
    )

    info_agent = InfoAgent(items, **hparams)
    volume_agent = VolumeAgent(items, **hparams)
    random_agent = RandomAgent(items, **hparams)

    p_v = td.Normal(torch.zeros(num_mix, dim, device=device), scale * torch.ones(num_mix, dim, device=device))
    p_mix = td.Exponential(torch.tensor(num_mix * [1.], device=device))


    results = defaultdict(list)
    for _ in tqdm.trange(500):
        result = compare(info_agent, volume_agent, random_agent)
        for x in result:
            results[x].append(result[x])
        for label in ['logp', 'mse']:
            plt.figure()
            for k, tau in results.items():
                if not k.endswith(label):
                    continue
                tau = torch.stack(tau)
                torch.save(tau, f'{args.data.lower()}-sim/{k}.pt')
                mu, sigma = tau.mean(dim=0), tau.std(dim=0)
                se = sigma / np.sqrt(tau.size(0))
                p = plt.plot(mu, label=k)
                plt.fill_between(
                    torch.arange(mu.size(0)), mu - se, mu + se, color=p[0].get_color(), alpha=0.2
                )
            plt.xlabel('Observations')
            plt.ylabel('Metric')
            plt.legend()
            plt.savefig(f'{args.data.lower()}-sim/{label}')
        for label in ['pred', 'true']:
            for k, params in results.items():
                if not k.endswith(label):
                    continue
                torch.save(params, f'{args.data.lower()}-sim/{k}.pt')

