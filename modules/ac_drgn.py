import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.drgn import DRGNNetwork

CriticDRGNNetwork = DRGNNetwork


class ActorDRGNNetwork(DRGNNetwork):
    def __init__(self, in_dim, hidden_dim, action_dim, num_head=4, skip_connect=False):
        super(ActorDRGNNetwork, self).__init__(in_dim, hidden_dim, action_dim, num_head, skip_connect)

    def forward(self, x, mask, hidden_state, return_log_pi=False):
        logit, actor_hidden_state = super(ActorDRGNNetwork, self).forward(x, mask, hidden_state)
        if return_log_pi:
            return F.softmax(logit, dim=-1), actor_hidden_state, F.log_softmax(logit, dim=-1)
        else:
            return F.softmax(logit, dim=-1), actor_hidden_state
