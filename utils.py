import torch
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

def select_actions(pi, deterministic=False):
    """
    Select - actions
    """
    cate_dist = Categorical(pi)
    if deterministic:
        return torch.argmax(pi, dim=1).item()
    else:
        return cate_dist.sample().unsqueeze(-1)

def select_state(mu, deterministic=False):
    """
    Select Δs_t from Multivariate normal with mean mu
    """
    if deterministic:
        return mu
    else:
        shape = mu.shape
        mu = mu.view(-1)
        gauss = MultivariateNormal(mu.view(-1), torch.eye(mu.shape[0]))
        return gauss.sample().view(shape)

def evaluate_actions(pi, actions):
    """
    Get the action log prob and entropy... Will be used for entropy regularization

    """
    cate_dist = Categorical(pi)
    return cate_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1), cate_dist.entropy().mean()

def evaluate_actions_sil(pi, actions):
    """
    Get the action log prob and entropy...
    """
    cate_dist = Categorical(pi)
    return cate_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1), cate_dist.entropy().unsqueeze(-1)

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1.-done)
        discounted.append(r)
    return discounted[::-1]

def select_mj(mu, sigma, deterministic=False):
    """
    Select Δs_t from Multivariate normal with mean mu and cov_matrix
    """
    if deterministic:
        return mu
    else:
        shape = mu.shape
        mu = mu.view(-1)
        sigma = sigma.view(-1)
        gauss = MultivariateNormal(mu, torch.diag(sigma))
        return gauss.sample().view(shape)

def evaluate_actions_mj(mu, sigma, actions):
    shape = mu.shape
    mu = mu.view(-1)
    sigma = sigma.view(-1)
    actions = actions.view(-1)
    gauss = MultivariateNormal(mu, torch.diag(sigma))
    return gauss.log_prob(actions)