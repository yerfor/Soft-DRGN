import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.base import MultiHeadAttentionLayer
from utils.hparams import hparams


class HGATLayer(nn.Module):
    def __init__(self, in_dims, hidden_dim, out_dims, num_head):
        super(HGATLayer, self).__init__()
        self.in_dims = in_dims
        self.hidden_dim = hidden_dim
        self.out_dims = out_dims
        self.num_head = num_head
        assert type(hidden_dim) is int
        self.sqrt_dk = self.hidden_dim ** 0.5

        self.num_groups = len(in_dims)
        self.fc_v = nn.ModuleList()
        self.fc_k = nn.ModuleList()
        self.fc_q = nn.ModuleList()
        self.fc_out_final = nn.ModuleList()
        self.fc_outs = []

        self.tmp_q_vectors = [None] * self.num_groups
        self.tmp_k_vectors = [None] * self.num_groups
        self.tmp_v_vectors = [None] * self.num_groups

    def init_from_groupwise_connection(self, groupwise_connection):
        in_dims, hidden_dim, out_dims = self.in_dims, self.hidden_dim, self.out_dims
        self.fc_outs = nn.ModuleList([None] * self.num_groups)  # 2D-list to store the fc_layers for each group to all groups
        for i in range(self.num_groups):
            self.fc_v.append(nn.Linear(in_dims[i], self.hidden_dim * self.num_head))
            self.fc_k.append(nn.Linear(in_dims[i], self.hidden_dim * self.num_head))
            self.fc_q.append(nn.Linear(in_dims[i], self.hidden_dim * self.num_head))
            self.fc_out_final.append(nn.Linear(hidden_dim * len(groupwise_connection[i]), out_dims[i]))

            fc_out_i = self.fc_outs[i] = nn.ModuleList([None] * self.num_groups)
            for j in groupwise_connection[i]:
                fc_out_i[j] = nn.Linear(hidden_dim * self.num_head, hidden_dim)
        self.cuda()

    def inner_group_forward(self, x, mask, idx):
        """
        mask: Tensor, Adjacency Matrix of agents in same groups
        """
        self.tmp_q_vectors[idx] = q_idx = F.relu(self.fc_q[idx](x))
        self.tmp_k_vectors[idx] = k_idx = F.relu(self.fc_k[idx](x))
        self.tmp_v_vectors[idx] = v_idx = F.relu(self.fc_v[idx](x))

        out = []
        for i_head in range(self.num_head):
            q_i = q_idx[:, :, i_head * self.hidden_dim: (i_head + 1) * self.hidden_dim]
            k_i = k_idx[:, :, i_head * self.hidden_dim: (i_head + 1) * self.hidden_dim]
            v_i = v_idx[:, :, i_head * self.hidden_dim: (i_head + 1) * self.hidden_dim]
            attention_logit_i = torch.bmm(q_i, k_i.permute(0, 2, 1)) / self.sqrt_dk  # [b,na,na]
            masked_logit_i = torch.mul(attention_logit_i, mask) - 9e15 * (1 - mask)
            attention_weight_i = F.softmax(masked_logit_i, dim=2)
            out_i = torch.bmm(attention_weight_i, v_i)  # [b,na,na] @ [b,na,hid] ==> [b,na,hid]
            out.append(out_i)
        out = torch.cat(out, dim=-1)
        fc_out = F.relu(self.fc_outs[idx][idx](out))
        self.inner_group_idx_out = fc_out
        return fc_out

    def inter_group_forward(self, mask_lst_row, idx, groupwise_connection):
        comm_indices = groupwise_connection[idx]
        q_vector = self.tmp_q_vectors[idx]
        fc_outs = []

        for i in comm_indices:
            if i == idx:
                continue
            out_vectors_from_i = []
            mask_i = mask_lst_row[i]
            k_vector = self.tmp_k_vectors[i]
            v_vector = self.tmp_v_vectors[i]
            has_neighbors_in_mask = (mask_i.sum(dim=-1, keepdim=True) > 0).float()
            for j_head in range(self.num_head):
                q_j = q_vector[:, :, j_head * self.hidden_dim: (j_head + 1) * self.hidden_dim]
                k_j = k_vector[:, :, j_head * self.hidden_dim: (j_head + 1) * self.hidden_dim]
                v_j = v_vector[:, :, j_head * self.hidden_dim: (j_head + 1) * self.hidden_dim]
                attention_logit_j = torch.bmm(q_j, k_j.permute(0, 2, 1)) / self.sqrt_dk  # [b,na,na]
                masked_logit_j = torch.mul(attention_logit_j, mask_i) - 9e15 * (1 - mask_i)
                attention_weight_j = F.softmax(masked_logit_j, dim=2)
                attention_weight_j = attention_weight_j * has_neighbors_in_mask
                out_j = torch.bmm(attention_weight_j, v_j)  # [b,na,na] @ [b,na,hid] ==> [b,na,hid]
                out_vectors_from_i.append(out_j)
            out = torch.cat(out_vectors_from_i, dim=-1)
            fc_out_idx_from_i = F.relu(self.fc_outs[idx][i](out))
            fc_outs.append(fc_out_idx_from_i)
        if len(fc_outs) == 0:
            return None
        fc_out = torch.cat(fc_outs, dim=-1)  # [batch, n_agent_in_group_idx, hidden*(num_comm_idx-1)]
        return fc_out

    def forward(self, x, mask_lst, groupwise_connection):
        """
        mask_lst : 2D-lst that stores adjacency matrix in the corresponding index
        """
        inner_forward_outs = []  # list of [batch, n_agent_in_group_idx, hidden]
        for idx in range(self.num_groups):
            inner_for_out_idx = self.inner_group_forward(x[idx], mask_lst[idx][idx], idx)
            inner_forward_outs.append(inner_for_out_idx)

        inter_forward_outs = []  # list of [batch, n_agent_in_group_idx, hidden*(num_comm_idx-1)]
        for idx in range(self.num_groups):
            inter_for_out_idx = self.inter_group_forward(mask_lst[idx], idx, groupwise_connection)
            inter_forward_outs.append(inter_for_out_idx)

        forward_outs = []
        for idx in range(self.num_groups):
            if inter_forward_outs[idx] is None:
                forward_outs.append(inner_forward_outs[idx])
            else:
                forward_outs.append(torch.cat([inner_forward_outs[idx], inter_forward_outs[idx]], dim=-1))
                # list of [batch, n_agent_in_group_idx, hidden*num_comm_idx]

        final_forward_outs = []
        for idx in range(self.num_groups):
            final_forward_outs.append(self.fc_out_final[idx](forward_outs[idx]))
        return final_forward_outs  # list of [batch, n_agent_in_group_idx, hidden]


class HGNNetwork(nn.Module):
    def __init__(self, in_dims, hidden_dim, act_dims, num_head=4, skip_connect=False):
        super(HGNNetwork, self).__init__()
        self.in_dim = in_dims
        self.hidden_dim = hidden_dim
        self.action_dim = act_dims
        self.num_head = num_head
        self.skip_connect = skip_connect
        assert type(hidden_dim) is int

        self.num_groups = len(in_dims)
        self.encoders = nn.ModuleList()
        self.linears = nn.ModuleList()
        for idx in range(self.num_groups):
            self.encoders.append(nn.Linear(self.in_dim[idx], self.hidden_dim))
            if self.skip_connect:
                self.linears.append(nn.Linear(3 * self.hidden_dim, self.action_dim[idx]))
            else:
                self.linears.append(nn.Linear(self.hidden_dim, self.action_dim[idx]))
        hgat_in_dims = [self.hidden_dim] * self.num_groups
        hgat_out_dims = [self.hidden_dim] * self.num_groups
        self.att_1 = HGATLayer(hgat_in_dims, self.hidden_dim, hgat_out_dims, self.num_head)
        self.att_2 = HGATLayer(hgat_in_dims, self.hidden_dim, hgat_out_dims, self.num_head)

    def forward(self, obs, adj):
        h1, h4, qs = [], [], []
        for idx in range(self.num_groups):
            h1_i = F.relu(self.encoders[idx](obs[idx]))
            h1.append(h1_i)

        h2 = self.att_1(h1, adj, self.groupwise_connection)
        h3 = self.att_2(h2, adj, self.groupwise_connection)
        if self.skip_connect:
            for idx in range(self.num_groups):
                h4_i = torch.cat([h1[idx], h2[idx], h3[idx]], dim=-1)
                h4.append(h4_i)
        else:
            h4 = h3

        for idx in range(self.num_groups):
            qs_i = self.linears[idx](h4[idx])
            qs.append(qs_i)
        return qs

    def init_from_groupwise_connection(self, groupwise_connection):
        """
        adjs: dict
        """
        self.groupwise_connection = groupwise_connection
        self.att_1.init_from_groupwise_connection(self.groupwise_connection)
        self.att_2.init_from_groupwise_connection(self.groupwise_connection)
