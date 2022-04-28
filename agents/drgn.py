import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agents.dgn import DGNAgent
from modules.drgn import DRGNNetwork

from utils.numba_utils import *
from utils.torch_utils import *
from utils.hparams import hparams


class DRGNAgent(DGNAgent):
    def __init__(self, in_dim, act_dim):
        nn.Module.__init__(self)
        self.in_dim = in_dim
        self.act_dim = act_dim
        self.hidden_dim = hparams['hidden_dim']
        self.num_head = hparams['num_head']

        self.learned_model = DRGNNetwork(in_dim, hparams['hidden_dim'], act_dim, hparams['num_head'],
                                         hparams['skip_connect'])
        self.target_model = DRGNNetwork(in_dim, hparams['hidden_dim'], act_dim, hparams['num_head'],
                                        hparams['skip_connect'])
        self.target_model.load_state_dict(self.learned_model.state_dict())

        # We need to maintain current_critic_hidden_state because it is necessary in action()
        # previous_critic_hidden_state is necessary for training the model.
        self.previous_critic_hidden_states = None
        self.current_critic_hidden_states = None

    def reset_hidden_states(self, n_ant):
        """
        We need to call this function at the start of each episode
        """
        self.previous_critic_hidden_states = torch.zeros([n_ant, self.hidden_dim]).cuda()
        self.current_critic_hidden_states = torch.zeros([n_ant, self.hidden_dim]).cuda()

    def get_hidden_states(self):
        previous_critic_hidden = to_cpu(self.previous_critic_hidden_states)
        current_critic_hidden = to_cpu(self.current_critic_hidden_states)
        return {'cri_hid': previous_critic_hidden, 'next_cri_hid': current_critic_hidden}

    def action(self, obs, adj, epsilon=0.3, action_mode='epsilon-categorical'):
        """
        :param obs: ndarray with [n_agent, hidden], or Tensor with [batch, n_agent, hidden]
        :param adj: ndarray with [n_agent, n_agent], or Tensor with [batch, n_agent, hidden]
        :param epsilon: float
        :param action_mode: str
        :return:
        """
        assert self.previous_critic_hidden_states is not None and self.current_critic_hidden_states is not None
        critic_hidden_states = self.current_critic_hidden_states
        self.previous_critic_hidden_states = self.current_critic_hidden_states

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
            if is_batched_input:
                batch_size = obs.shape[0]
                q, next_critic_hidden_state = self.learned_model(obs, adj, critic_hidden_states)
                q = q.squeeze().cpu().numpy()
                action = []
                for b_i in range(batch_size):
                    q_i = q[b_i]
                    action_i = self._sample_action_from_q_values(q_i, epsilon, action_mode)  # [n_agent, ]
                    action.append(action_i)
                action = np.stack(action, axis=0)
            else:
                q, next_critic_hidden_state = self.learned_model(obs, adj, critic_hidden_states)
                q = q.squeeze().cpu().numpy()
                action = self._sample_action_from_q_values(q, epsilon, action_mode)  # [n_agent, ]
        self.current_critic_hidden_states = next_critic_hidden_state
        return action

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        """
        sample: dict of cuda_Tensor.
        losses: dict to store loss_Tensor
        log_vars: dict to store scalars, formatted as (global_steps, vars)
        global_steps: int
        """
        obs = sample['obs']
        cri_hid = sample['cri_hid']
        adj = sample['adj']
        action = sample['action']
        reward = sample['reward']
        next_obs = sample['next_obs']
        next_cri_hid = sample['next_cri_hid']
        next_adj = sample['next_adj']
        done = sample['done']

        batch_size, n_ant, _ = obs.shape

        # q_values: Q(s,a), [b, n_agent, n_action]
        q_values, _ = self.learned_model(obs, adj, cri_hid)
        # target_q_values: max Q'(s',a'), [b, n_agent,]
        with torch.no_grad():
            # Calculate target Q with DQN: Q'(s',a'), where a' = argmax Q'(s',a')
            # We find that when calculating Q', it is better to use use true dynamic graph,
            # instead of the fixed graph (as suggested in DGN paper)
            target_q_values, _ = self.target_model(next_obs, next_adj, next_cri_hid)
            target_q_values = target_q_values.max(dim=2)[0]
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
