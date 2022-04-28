import torch.nn as nn


class BaseAgent(nn.Module):
    def __init__(self):
        self.learned_model = None
        self.target_model = None

    def action(self, sample, epsilon, action_mode):
        raise NotImplementedError

    def cal_q_loss(self, sample, losses, log_vars):
        raise NotImplementedError

    def update_target(self):
        raise NotImplementedError


class BaseActorCriticAgent(nn.Module):
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
