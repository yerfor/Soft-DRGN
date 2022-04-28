import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.base import MultiHeadAttentionLayer


class DGNResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_head, with_non_linearity=False):
        super(DGNResidualBlock, self).__init__()
        self.att_1 = MultiHeadAttentionLayer(in_dim, hidden_dim, hidden_dim, num_head)
        self.att_2 = MultiHeadAttentionLayer(hidden_dim, hidden_dim, hidden_dim, num_head)
        self.linear = nn.Linear(3 * hidden_dim, out_dim)
        self.with_non_linearity = with_non_linearity

    def forward(self, x, mask, with_non_linearity=None):
        if with_non_linearity is None:
            with_non_linearity = self.with_non_linearity
        h1, att_weight_1 = self.att_1(x, mask)
        h2, att_weight_2 = self.att_2(h1, mask)
        h2 = torch.cat([x, h1, h2], dim=-1)
        if with_non_linearity:
            h3 = F.relu(self.linear(h2))
        else:
            h3 = self.linear(h2)
        return h3


class DGNStackBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_head, with_non_linearity=False):
        super(DGNStackBlock, self).__init__()
        self.att_1 = MultiHeadAttentionLayer(in_dim, hidden_dim, hidden_dim, num_head)
        self.att_2 = MultiHeadAttentionLayer(hidden_dim, hidden_dim, hidden_dim, num_head)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.with_non_linearity = with_non_linearity

    def forward(self, x, mask, with_non_linearity=None):
        if with_non_linearity is None:
            with_non_linearity = self.with_non_linearity
        h1, att_weight_1 = self.att_1(x, mask)
        h2, att_weight_2 = self.att_2(h1, mask)
        if with_non_linearity:
            h3 = F.relu(self.linear(h2))
        else:
            h3 = self.linear(h2)
        return h3


class DGNNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, action_dim, num_head=4, skip_connect=False):
        super(DGNNetwork, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_head = num_head
        self.skip_connect = skip_connect
        self.encoder = nn.Linear(in_dim, hidden_dim)
        if skip_connect:
            self.dgn_block = DGNResidualBlock(hidden_dim, hidden_dim, action_dim, num_head, with_non_linearity=False)
        else:
            self.dgn_block = DGNStackBlock(hidden_dim, hidden_dim, action_dim, num_head, with_non_linearity=False)

    def forward(self, x, mask):
        bs, n_agent, _ = x.shape
        x = F.relu(self.encoder(x))
        qs = self.dgn_block(x, mask)
        qs = qs.reshape([bs, n_agent, self.action_dim])
        return qs
