# Modules模块

modules模块用于定义神经网络结构，任何算法模型的具体网络结构都在其中定义。任何在网络结构之外的操作，比如做动作采样、计算loss等功能，都不应该在本模块中实现。

本模块定义的Network类，被使用于agents模块的Agent类中，由Agent类来实现上面提到的动作采样、计算loss等高阶功能。

具体来说，一个合格的Network类只需要提供如下接口：

1. init方法：定义网络结构需要的网络参数，如Linear、MultiAttentionLayer等
2. forward方法：定义网络的前向传播

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

