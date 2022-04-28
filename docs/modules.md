# Modules

The modules module is used to define the neural network structure, and the specific network structure of any algorithm model is defined in it. Any operations outside the network structure, such as action sampling and loss calculation, should not be implemented in this module.

The Network class defined in this module is used in the Agent class of the agents module, and the Agent class implements the above-mentioned high-level functions such as action sampling and loss calculation.

Specifically, a qualified Network class only needs to provide the following interfaces:

1. init method: define the network parameters required by the network structure, such as Linear, MultiAttentionLayer, etc.
2. forward method: define the forward propagation of the networ

```python
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
```
