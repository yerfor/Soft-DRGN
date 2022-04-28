import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.base import MultiHeadAttentionLayer


class DRGNResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_head, with_non_linearity=False):
        super(DRGNResidualBlock, self).__init__()
        self.att_1 = MultiHeadAttentionLayer(in_dim, hidden_dim, hidden_dim, num_head)
        self.att_2 = MultiHeadAttentionLayer(hidden_dim, hidden_dim, hidden_dim, num_head)
        self.recurrent_layer = nn.GRUCell(3 * hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.with_non_linearity = with_non_linearity

    def forward(self, x, mask, hidden_state, with_non_linearity=None):
        assert hidden_state is not None
        bs, n_ant, _ = x.shape
        if with_non_linearity is None:
            with_non_linearity = self.with_non_linearity
        h1, att_weight_1 = self.att_1(x, mask)
        h2, att_weight_2 = self.att_2(h1, mask)
        h2 = torch.cat([x, h1, h2], dim=-1)
        # Since GRU only support the [batch_size, hidden] as input, we need to reshape
        h2 = h2.reshape([bs * n_ant, -1])
        hidden_state = hidden_state.reshape([bs * n_ant, -1])
        h3 = self.recurrent_layer(h2, hidden_state).reshape([bs, n_ant, -1])
        next_hidden_state = h3
        if with_non_linearity:
            h4 = F.relu(self.linear(h3))
        else:
            h4 = self.linear(h3)
        return h4, next_hidden_state


class DRGNStackBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_head, with_non_linearity=False):
        super(DRGNStackBlock, self).__init__()
        self.att_1 = MultiHeadAttentionLayer(in_dim, hidden_dim, hidden_dim, num_head)
        self.att_2 = MultiHeadAttentionLayer(hidden_dim, hidden_dim, hidden_dim, num_head)
        self.recurrent_layer = nn.GRUCell(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.with_non_linearity = with_non_linearity

    def forward(self, x, mask, hidden_state, with_non_linearity=None):
        assert hidden_state is not None
        bs, n_ant, _ = x.shape
        if with_non_linearity is None:
            with_non_linearity = self.with_non_linearity
        h1, att_weight_1 = self.att_1(x, mask)
        h2, att_weight_2 = self.att_2(h1, mask)
        # Since GRU only support the [batch_size, hidden] as input, we need to reshape
        h2 = h2.reshape([bs * n_ant, -1])
        hidden_state = hidden_state.reshape([bs * n_ant, -1])
        h3 = self.recurrent_layer(h2, hidden_state).reshape([bs, n_ant, -1])
        next_hidden_state = h3
        if with_non_linearity:
            h4 = F.relu(self.linear(h3))
        else:
            h4 = self.linear(h3)
        return h4, next_hidden_state


class DRGNNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, action_dim, num_head=4, skip_connect=False):
        super(DRGNNetwork, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_head = num_head
        self.skip_connect = skip_connect

        self.encoder = nn.Linear(in_dim, hidden_dim)
        if skip_connect:
            self.drgn_block = DRGNResidualBlock(hidden_dim, hidden_dim, action_dim, num_head, with_non_linearity=False)
        else:
            self.drgn_block = DRGNStackBlock(hidden_dim, hidden_dim, action_dim, num_head, with_non_linearity=False)

    def forward(self, x, mask, hidden_state):
        bs, n_agent, _ = x.shape
        x = F.relu(self.encoder(x))
        qs, next_hidden_state = self.drgn_block(x, mask, hidden_state)
        qs = qs.reshape([bs, n_agent, self.action_dim])
        next_hidden_state = next_hidden_state.reshape([bs, n_agent, -1])
        return qs, next_hidden_state
