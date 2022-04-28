import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.dgn import DGNNetwork

CriticDGNNetwork = DGNNetwork


class ActorDGNNetwork(DGNNetwork):
    def __init__(self, in_dim, hidden_dim, action_dim, num_head=4, skip_connect=False):
        super(ActorDGNNetwork, self).__init__(in_dim, hidden_dim, action_dim, num_head, skip_connect)

    def forward(self, x, mask, return_log_pi=False):
        logit = super(ActorDGNNetwork, self).forward(x, mask)
        if return_log_pi:
            return F.softmax(logit, dim=-1), F.log_softmax(logit, dim=-1)
        else:
            return F.softmax(logit, dim=-1)
