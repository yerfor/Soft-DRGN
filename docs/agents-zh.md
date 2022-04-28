# Agents模块

Agents模块实现了Agent类，它在Modules模块提供的Network类的基础上，提供了强化学习算法中需要的动作采样、计算loss等功能。根据是否是Actor-Critic的算法，我们设计了两类Agent：ValueBasedAgent和ActorCriticAgent，它们分别需要满足以下接口：

```python
class BaseAgent:
    def __init__(self):
        self.learned_model = None # 是Network类的实例
        self.target_model = None
        
    def action(self, sample, epsilon, action_mode):
        raise NotImplementedError

    def cal_q_loss(self, sample, losses, log_vars):
        raise NotImplementedError

    def update_target(self):
        raise NotImplementedError


class BaseActorCriticAgent:
    def __init__(self):
        self.learned_actor_model = None
        self.target_actor_model = None
        self.learned_critic_model = None
        self.target_critic_model = None

    def action(self, sample, epsilon, action_mode):
        raise NotImplementedError

    def cal_p_loss(self, sample, losses, log_vars):
        raise NotImplementedError

    def cal_q_loss(self, sample, losses, log_vars):
        raise NotImplementedError

    def update_target(self):
        raise NotImplementedError
```