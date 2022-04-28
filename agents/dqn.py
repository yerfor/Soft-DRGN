import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agents.base_agent import BaseAgent
from modules.dqn import DQNNetwork

from utils.numba_utils import *
from utils.torch_utils import *
from utils.hparams import hparams


class DQNAgent(BaseAgent):
    def __init__(self, in_dim, act_dim):
        nn.Module.__init__(self)
        self.in_dim = in_dim
        self.act_dim = act_dim
        self.hidden_dim = hparams['hidden_dim']

        self.learned_model = DQNNetwork(in_dim, hparams['hidden_dim'], act_dim)
        self.target_model = DQNNetwork(in_dim, hparams['hidden_dim'], act_dim)
        self.target_model.load_state_dict(self.learned_model.state_dict())

    def action(self, obs, adj, epsilon=0.3, action_mode='epsilon-greedy'):
        """
        :param obs: ndarray with [n_agent, hidden], or Tensor with [batch, n_agent, hidden]
        :param adj: unused for DQN
        :param epsilon: float
        :param action_mode: str
        :return:
        """
        is_batched_input = obs.ndim == 3
        if isinstance(obs, np.ndarray):
            # from environment
            assert obs.ndim == 2
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).cuda()
        elif isinstance(obs, torch.Tensor):
            # from replay buffer
            assert obs.ndim == 3
            obs = to_cuda(obs)
        else:
            raise TypeError

        with torch.no_grad():
            if is_batched_input:
                batch_size = obs.shape[0]
                q = self.learned_model(obs).squeeze().cpu().numpy()  # [batch,n_agent]
                action = []
                for b_i in range(batch_size):
                    q_i = q[b_i]
                    action_i = self._sample_action_from_q_values(q_i, epsilon, action_mode)  # [n_agent, ]
                    action.append(action_i)
                action = np.stack(action, axis=0)
            else:
                q = self.learned_model(obs).squeeze().cpu().numpy()  # [n_agent,]
                action = self._sample_action_from_q_values(q, epsilon, action_mode)  # [n_agent, ]
        return action

    def _sample_action_from_q_values(self, q_values, epsilon, action_mode):
        """
        :param q_values: np.ndarray [n_agent, n_action]
        :param epsilon: float
        :param action_mode: str
        :return: action, np.ndarray [n_agent, ]
        """
        action = []
        assert q_values.ndim == 2
        n_agent = q_values.shape[0]
        if action_mode == 'epsilon-greedy':
            for i in range(n_agent):  # agent-wise epsilon-greedy
                if np.random.rand() < epsilon:
                    a = np.random.randint(self.act_dim)
                else:
                    a = q_values[i].argmax().item()
                action.append(a)
        elif action_mode == 'categorical':
            action = numba_categorical_sample(q_values)
        elif action_mode == 'epsilon-categorical':
            action = numba_epsilon_categorical_sample(q_values, epsilon)
        elif action_mode == 'greedy':
            for i in range(n_agent):
                a = q_values[i].argmax().item()
                action.append(a)
        else:
            raise ValueError
        action = np.array(action, dtype=np.float32).reshape([n_agent, ])
        return action

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        """
        sample: dict of cuda_Tensor.
        losses: dict to store loss_Tensor
        log_vars: dict to store scalars, formatted as (global_steps, vars)
        global_steps: int
        """
        obs = sample['obs']
        action = sample['action']
        reward = sample['reward']
        next_obs = sample['next_obs']
        done = sample['done']

        batch_size, n_ant, _ = obs.shape

        # q_values: Q(s,a), [b, n_agent, n_action]
        q_values = self.learned_model(obs)
        # target_q_values: max Q'(s',a'), [b, n_agent,]
        with torch.no_grad():
            # Calculate target Q with DQN: Q'(s',a'), where a' = argmax Q'(s',a')
            target_q_values = self.target_model(next_obs).max(dim=2)[0]
            target_q_values = target_q_values.cpu().numpy()

        # expected_q: r+maxQ'(s',a'), only the sampled action index are different with Q_values,[b, n_agent, n_action].
        numpy_q_values = q_values.detach().cpu().numpy()
        expected_q = numba_get_expected_q(numpy_q_values, action.cpu().numpy(), reward.cpu().numpy(),
                                          done.cpu().numpy(), hparams['gamma'], target_q_values,
                                          batch_size, n_ant)
        expected_q = torch.tensor(expected_q).cuda()

        # q_loss: MSE calculated on the sampled action index!
        q_loss = (q_values - expected_q).pow(2).mean()
        losses['q_loss'] = q_loss

    def update_target(self):
        if hparams['soft_update_target_network']:
            soft_update(self.learned_model, self.target_model, hparams['tau'])
        else:
            hard_update(self.learned_model, self.target_model)
