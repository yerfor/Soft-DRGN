# Agents

The Agents module implements the Agent class. Based on the Network class provided by the Modules module, it provides functions such as action sampling and loss calculation required in the reinforcement learning algorithm. According to whether it is an Actor-Critic algorithm, we have designed two types of Agents: ValueBasedAgent and ActorCriticAgent, which respectively need to meet the following interfaces:

```python
class BaseAgent:
    def __init__(self):
        self.learned_model = None # is an object of Network class
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
