import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import ipdb

class NetMJ(nn.Module):
    """
    The Policy Network.
    """
    def __init__(self, obs_shape, action_shape):
        super(NetMJ, self).__init__()
        num_outputs = action_shape.shape[0]
        self.layer1 = nn.Linear(obs_shape.shape[0],128)
        self.layer2 = nn.Linear(128,128)
        self.critic = nn.Linear(128, 1)
        self.actor = nn.Linear(128, num_outputs)
        #Initialize hidden layers of mlp
        nn.init.orthogonal_(self.layer1.weight.data)
        nn.init.constant_(self.layer1.bias.data, 0)
        nn.init.orthogonal_(self.layer2.weight.data)
        nn.init.constant_(self.layer2.bias.data, 0)
        # init the linear layer..
        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.constant_(self.critic.bias.data, 0)
        # init the policy layer...
        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        nn.init.constant_(self.actor.bias.data, 0)

    def forward(self, inputs):
        x = F.tanh(self.layer1(inputs))
        x = F.tanh(self.layer2(x))
        value = self.critic(x)
        a_mu = self.actor(x)
        return value, a_mu

class ActGenMJ(nn.Module):
    """
    The action generator. P(a_t | s_t+1)
    """
    def __init__(self, obs_shape, action_shape):
        super(ActGenMJ, self).__init__()
        num_outputs = action_shape.shape[0]
        self.layer1 = nn.Linear(obs_shape.shape[0],128)
        self.layer2 = nn.Linear(128,128)
        self.actor = nn.Linear(128, num_outputs)
        #Initialize hidden layers of mlp
        nn.init.orthogonal_(self.layer1.weight.data)
        nn.init.constant_(self.layer1.bias.data, 0)
        nn.init.orthogonal_(self.layer2.weight.data)
        nn.init.constant_(self.layer2.bias.data, 0)
        # init the policy layer...
        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        nn.init.constant_(self.actor.bias.data, 0)

    def forward(self, inputs):
        x = F.tanh(self.layer1(inputs))
        x = F.tanh(self.layer2(x))
        return self.actor(x)

class StateGenMJ(nn.Module):
    """
    The State Generator. P(Δs_t | a_t, s_t+1). For forward and backward Model
    """
    def __init__(self, obs_shape, action_shape):
        super(StateGenMJ, self).__init__()
        self.layer1 = nn.Linear(obs_shape.shape[0] + action_shape.shape[0],128)
        self.layer2 = nn.Linear(128,128)
        self.sigma = nn.Linear(128, num_outputs)
        self.mu = nn.Linear(128, num_outputs)
        #Initialize hidden layers of mlp
        nn.init.orthogonal_(self.layer1.weight.data)
        nn.init.constant_(self.layer1.bias.data, 0)
        nn.init.orthogonal_(self.layer2.weight.data)
        nn.init.constant_(self.layer2.bias.data, 0)
        # init the linear layer..
        nn.init.orthogonal_(self.sigma.weight.data)
        nn.init.constant_(self.sigma.bias.data, 0)
        # init the policy layer...
        nn.init.orthogonal_(self.mu.weight.data, gain=0.01)
        nn.init.constant_(self.mu.bias.data, 0)

    def forward(self, obs, actions):
        x = torch.cat((obs, actions),1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.mu(x), self.sigma(x)

