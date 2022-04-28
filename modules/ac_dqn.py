import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.dqn import DQNNetwork

CriticDQNNetwork = DQNNetwork


class ActorDQNNetwork(DQNNetwork):
    def __init__(self, in_dim, hidden_dim, action_dim):
        super(ActorDQNNetwork, self).__init__(in_dim, hidden_dim, action_dim)

    def forward(self, x, return_log_pi=False):
        logit = super(ActorDQNNetwork, self).forward(x)
        if return_log_pi:
            return F.softmax(logit, dim=-1), F.log_softmax(logit, dim=-1)
        else:
            return F.softmax(logit, dim=-1)
