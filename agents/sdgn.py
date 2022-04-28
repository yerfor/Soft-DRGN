import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agents.dgn import DGNAgent
from modules.dgn import DGNNetwork

from utils.numba_utils import *
from utils.torch_utils import *
from utils.hparams import hparams


def soft_value_based_model_entropy_activation_function(alpha):
    a = alpha.item()
    if a >= 1:
        return 1.7182818284 + alpha
    # elif a > 5E-2:
    elif a > 0.1:
        return alpha.exp()
    elif a > 0.01:
        return alpha.exp() * 0.1
    elif a > 0.001:
        return alpha.exp() * 0.01
    elif a > 0:
        return alpha.exp() * 0.001
    else:
        # assert alpha > 0!!!
        raise ValueError("The temperature should > 0")


class SoftDGNAgent(DGNAgent):
    def __init__(self, in_dim, act_dim):
        super(SoftDGNAgent, self).__init__(in_dim, act_dim)
        self.alpha = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.target_entropy = -np.log(1.0 / self.act_dim) * hparams['entropy_target_factor']

    def action(self, obs, adj, epsilon=0.3, action_mode='epsilon-categorical'):
        """
        :param obs: ndarray with [n_agent, hidden], or Tensor with [batch, n_agent, hidden]
        :param adj: ndarray with [n_agent, n_agent], or Tensor with [batch, n_agent, hidden]
        :param epsilon: float
        :param action_mode: str
        :return:
        """
        is_batched_input = obs.ndim == 3
        if isinstance(obs, np.ndarray):
            # from environment
            assert obs.ndim == 2
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).cuda()
            adj = torch.tensor(adj, dtype=torch.float32).unsqueeze(0).cuda()
        elif isinstance(obs, torch.Tensor):
            # from replay buffer
            assert obs.ndim == 3
            obs, adj = to_cuda(obs), to_cuda(adj)
        else:
            raise TypeError

        with torch.no_grad():
            v_values = self.learned_model(obs, adj)
            advantage = v_values - self.alpha * torch.logsumexp(v_values / self.alpha, dim=-1).unsqueeze(dim=-1)
            q = torch.softmax(advantage / self.alpha, dim=-1).squeeze().cpu().numpy()
            if is_batched_input:
                batch_size = obs.shape[0]
                action = []
                for b_i in range(batch_size):
                    q_i = q[b_i]
                    action_i = self._sample_action_from_q_values(q_i, epsilon, action_mode)  # [n_agent, ]
                    action.append(action_i)
                action = np.stack(action, axis=0)
            else:
                action = self._sample_action_from_q_values(q, epsilon, action_mode)  # [n_agent, ]
        return action

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        """
        sample: dict of cuda_Tensor.
        losses: dict to store loss_Tensor
        log_vars: dict to store scalars, formatted as (global_steps, vars)
        global_steps: int
        """
        obs = sample['obs']
        adj = sample['adj']
        action = sample['action']
        reward = sample['reward']
        next_obs = sample['next_obs']
        next_adj = sample['next_adj']
        done = sample['done']

        batch_size, n_ant, _ = obs.shape

        # q_values: Q(s,a), [b, n_agent, n_action]
        q_values = self.learned_model(obs, adj)
        # target_q_values: max Q'(s',a'), [b, n_agent,]
        with torch.no_grad():
            v_values = self.target_model(next_obs, next_adj)
            v_values = self.alpha * torch.logsumexp(v_values / self.alpha, dim=-1)
            v_values = v_values.cpu().numpy()  # [batch, n_agent]
            target_q_values = v_values

        numpy_q_values = q_values.detach().cpu().numpy()
        expected_q = numba_get_expected_q(numpy_q_values, action.cpu().numpy(), reward.cpu().numpy(),
                                          done.cpu().numpy(), hparams['gamma'], target_q_values,
                                          batch_size, n_ant)
        expected_q = torch.tensor(expected_q).cuda()

        # q_loss: MSE calculated on the sampled action index!
        q_loss = (q_values - expected_q).pow(2).mean()
        losses['q_loss'] = q_loss

    def cal_alpha_loss(self, sample, losses, log_vars=None, global_steps=None):
        obs = sample['obs']
        adj = sample['adj']

        with torch.no_grad():
            v_values = self.learned_model(obs, adj)
            probs = torch.softmax(v_values/self.alpha, dim=-1)
            entropies = (-probs*torch.log(probs+1e-10)).sum(-1)
            if log_vars is not None:
                entropy = entropies.mean().item()
                log_vars['action_entropy'] = (global_steps, entropy)
        entropy_loss = - soft_value_based_model_entropy_activation_function(self.alpha) * (self.target_entropy - entropies).mean()
        losses['entropy_loss'] = entropy_loss

    def clip_alpha_grad(self, log_vars=None, global_steps=None):
        torch.nn.utils.clip_grad_norm_(self.alpha, max_norm=self.alpha.item() * 0.01, norm_type=1)
        self.alpha.data = torch.max(self.alpha.data, torch.ones_like(self.alpha.data) * 1e-5)
        if log_vars is not None:
            log_vars['alpha'] = (global_steps, self.alpha.item())
