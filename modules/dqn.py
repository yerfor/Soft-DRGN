import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        h1 = F.relu(self.encoder(x))
        h2 = F.relu(self.linear1(h1))
        h3 = F.relu(self.linear2(h2))
        qs = self.linear3(h3)
        return qs
