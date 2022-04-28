import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_head):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_head = num_head
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.sqrt_dk = self.hidden_dim**0.5
        self.out_dim = out_dim
        self.fc_v = nn.Linear(in_dim, hidden_dim * num_head)
        self.fc_k = nn.Linear(in_dim, hidden_dim * num_head)
        self.fc_q = nn.Linear(in_dim, hidden_dim * num_head)
        self.fc_out = nn.Linear(num_head * hidden_dim, out_dim)

    def forward(self, x, mask):
        # x:[batch,n_agent,in_dim] ==>q k v [batch, n_agent, hidden_dim*head]
        q = F.relu(self.fc_q(x))
        k = F.relu(self.fc_k(x))
        v = F.relu(self.fc_v(x))
        out, attention_weights = [], []
        for i_head in range(self.num_head):
            q_i = q[:, :, i_head * self.hidden_dim: (i_head + 1) * self.hidden_dim]  # [b,na,hid]
            k_i = k[:, :, i_head * self.hidden_dim: (i_head + 1) * self.hidden_dim]
            v_i = v[:, :, i_head * self.hidden_dim: (i_head + 1) * self.hidden_dim]
            attention_logit_i = torch.bmm(q_i, k_i.permute(0, 2, 1))/self.sqrt_dk  # [b,na,na]
            masked_logit_i = torch.mul(attention_logit_i, mask) - 9e15 * (1 - mask)
            attention_weight_i = F.softmax(masked_logit_i, dim=2)
            out_i = torch.bmm(attention_weight_i, v_i)  # [b,na,na] @ [b,na,hid] ==> [b,na,hid]
            out.append(out_i)
            attention_weights.append(attention_weight_i)
        out = torch.cat(out, dim=-1)  # [b,na,hidden*num_head]
        out = F.relu(self.fc_out(out))
        return out, torch.cat(attention_weights, dim=0)

