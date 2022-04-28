import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.ac_dgn import ActorDGNNetwork, CriticDGNNetwork
from agents.ac_dqn import ActorCriticDQNAgent

from utils.hparams import hparams
from utils.numba_utils import *
from utils.torch_utils import *


class ActorCriticDGNAgent(ActorCriticDQNAgent):
    def __init__(self, in_dim, act_dim):
        nn.Module.__init__(self)
        self.in_dim = in_dim
        self.hidden_dim = hparams['hidden_dim']
        self.act_dim = act_dim
        self.num_head = hparams['num_head']

        self.actor_learned_model = ActorDGNNetwork(in_dim, hparams['hidden_dim'], act_dim, hparams['num_head'],
                                                   hparams['skip_connect'])
        self.actor_target_model = ActorDGNNetwork(in_dim, hparams['hidden_dim'], act_dim, hparams['num_head'],
                                                  hparams['skip_connect'])
        self.actor_target_model.load_state_dict(self.actor_learned_model.state_dict())

        self.critic_learned_model = CriticDGNNetwork(in_dim, hparams['hidden_dim'], act_dim, hparams['num_head'],
                                                     hparams['skip_connect'])
        self.critic_target_model = CriticDGNNetwork(in_dim, hparams['hidden_dim'], act_dim, hparams['num_head'],
                                                    hparams['skip_connect'])
        self.critic_target_model.load_state_dict(self.critic_learned_model.state_dict())

    def action(self, obs, adj, epsilon=0.3, action_mode='epsilon-greedy'):
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
            if is_batched_input:
                batch_size = obs.shape[0]
                p = self.actor_learned_model(obs, adj).squeeze().cpu().numpy()  # [batch, n_agent]
                action = []
                for b_i in range(batch_size):
                    p_i = p[b_i]
                    action_i = self._sample_action_from_p(p_i, epsilon, action_mode)  # [n_agent, ]
                    action.append(action_i)
                action = np.stack(action, axis=0)
            else:
                p = self.actor_learned_model(obs, adj).squeeze().cpu().numpy()  # [batch, n_agent]
                action = self._sample_action_from_p(p, epsilon, action_mode)  # [n_agent, ]
        return action

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        obs = sample['obs']
        adj = sample['adj']
        action = sample['action']
        reward = sample['reward']
        next_obs = sample['next_obs']
        next_adj = sample['next_adj']
        done = sample['done']
        batch_size, n_ant, _ = obs.shape

        # q_values : [b,n_agent,n_action]
        q_values = self.critic_learned_model(obs, adj)
        # target_q_values: [b,n_agent,]
        with torch.no_grad():
            # when calculating, use the fixed graph (in paper) or true dynamic graph (experimentally better)
            next_probs = self.actor_learned_model(next_obs, next_adj, return_log_pi=False)
            target_q_values = self.critic_target_model(next_obs, next_adj)
            target_q_values = (target_q_values * next_probs).sum(dim=-1)  # [batch, n_agent]
            target_q_values = target_q_values.cpu().numpy()
        numpy_q_values = q_values.detach().cpu().numpy()
        expected_q = numba_get_expected_q(numpy_q_values, action.cpu().numpy(), reward.cpu().numpy(),
                                          done.cpu().numpy(), hparams['gamma'], target_q_values,
                                          batch_size, n_ant)

        # q_loss: MSE calculated on the sampled action index!
        q_loss = (q_values - torch.tensor(expected_q).cuda()).pow(2).mean()
        losses['q_loss'] = q_loss

    def cal_p_loss(self, sample, losses, log_vars=None, global_steps=None):
        obs = sample['obs']
        adj = sample['adj']
        batch_size, n_ant, _ = obs.shape

        probs, log_probs = self.actor_learned_model(obs, adj, return_log_pi=True)
        log_probs = torch.log(probs + 1e-15)  # [batch, agent, action]

        with torch.no_grad():
            # q_values: Q(s,a), [b, n_agent, n_action]
            q_values = self.critic_learned_model(obs, adj)
            # baseline, V(s)=E_a[Q(s,a)]=\Sigma \pi(a|s)*Q(s,a)
            v_values = (probs * q_values).sum(dim=-1, keepdim=True)
            # advantage, A(s,a)=Q(s,a)-V(s)
            advantages = q_values - v_values
            # loss_p: \Sigma log\pi(a|s)*A(s,a)
        # p_loss = (-masked_log_probs * advantages).mean()
        p_loss = (-log_probs * advantages).mean()
        losses['p_loss'] = p_loss
