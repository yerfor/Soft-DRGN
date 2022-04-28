import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.ac_drgn import ActorDRGNNetwork, CriticDRGNNetwork
from agents.ac_dgn import ActorCriticDGNAgent

from utils.hparams import hparams
from utils.numba_utils import *
from utils.torch_utils import *


class ActorCriticDRGNAgent(ActorCriticDGNAgent):
    def __init__(self, in_dim, act_dim):
        nn.Module.__init__(self)
        self.in_dim = in_dim
        self.hidden_dim = hparams['hidden_dim']
        self.act_dim = act_dim
        self.num_head = hparams['num_head']

        self.actor_learned_model = ActorDRGNNetwork(in_dim, hparams['hidden_dim'], act_dim, hparams['num_head'],
                                                    hparams['skip_connect'])
        self.actor_target_model = ActorDRGNNetwork(in_dim, hparams['hidden_dim'], act_dim, hparams['num_head'],
                                                   hparams['skip_connect'])
        self.actor_target_model.load_state_dict(self.actor_learned_model.state_dict())

        self.critic_learned_model = CriticDRGNNetwork(in_dim, hparams['hidden_dim'], act_dim, hparams['num_head'],
                                                      hparams['skip_connect'])
        self.critic_target_model = CriticDRGNNetwork(in_dim, hparams['hidden_dim'], act_dim, hparams['num_head'],
                                                     hparams['skip_connect'])
        self.critic_target_model.load_state_dict(self.critic_learned_model.state_dict())

        self.previous_actor_hidden_states = None
        self.current_actor_hidden_states = None
        self.previous_critic_hidden_states = None
        self.current_critic_hidden_states = None

    def reset_hidden_states(self, n_ant):
        """
        We need to call this function at the start of each episode
        """
        self.previous_actor_hidden_states = torch.zeros([n_ant, self.hidden_dim]).cuda()
        self.previous_critic_hidden_states = torch.zeros([n_ant, self.hidden_dim]).cuda()
        self.current_actor_hidden_states = torch.zeros([n_ant, self.hidden_dim]).cuda()
        self.current_critic_hidden_states = torch.zeros([n_ant, self.hidden_dim]).cuda()

    def get_hidden_states(self):
        previous_actor_hidden = to_cpu(self.previous_critic_hidden_states)
        previous_critic_hidden = to_cpu(self.previous_critic_hidden_states)
        current_actor_hidden = to_cpu(self.current_critic_hidden_states)
        current_critic_hidden = to_cpu(self.current_critic_hidden_states)
        return {'act_hid':previous_actor_hidden,'cri_hid': previous_critic_hidden,
                'next_act_hid':current_actor_hidden,'next_cri_hid': current_critic_hidden}

    def action(self, obs, adj, epsilon=0.3, action_mode='epsilon-greedy', batched_act_hid_state=None):
        """
        :param obs: ndarray with [n_agent, hidden], or Tensor with [batch, n_agent, hidden]
        :param adj: ndarray with [n_agent, n_agent], or Tensor with [batch, n_agent, hidden]
        :param epsilon: float
        :param action_mode: str
        :param batched_act_hid_state: used for cal_target_q_value during training. CUDATensor [batch, agent, hidden]
        :return:
        """
        if batched_act_hid_state is not None:
            assert batched_act_hid_state.ndim == 3  # must be [batch, n_agent, hidden]
            actor_hidden_states = batched_act_hid_state
        else:
            assert self.previous_critic_hidden_states is not None and self.current_critic_hidden_states is not None
            actor_hidden_states = self.current_actor_hidden_states
            critic_hidden_states = self.current_critic_hidden_states
            self.previous_actor_hidden_states = self.current_actor_hidden_states
            self.previous_critic_hidden_states = self.current_critic_hidden_states

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
            if batched_act_hid_state is not None:
                batch_size = obs.shape[0]
                p = self.actor_learned_model(obs, adj, actor_hidden_states).squeeze().cpu().numpy()  # [batch, n_agent]
                action = []
                for b_i in range(batch_size):
                    p_i = p[b_i]
                    action_i = self._sample_action_from_p(p_i, epsilon, action_mode)  # [n_agent, ]
                    action.append(action_i)
                action = np.stack(action, axis=0)  # [batch, n_agent]
            else:
                p, next_actor_hidden_states = self.actor_learned_model(obs, adj, actor_hidden_states)
                p = p.squeeze().cpu().numpy()
                # [batch, n_agent]
                action = self._sample_action_from_p(p, epsilon, action_mode)  # [n_agent, ]

            if batched_act_hid_state is None:
                q_values, next_critic_hidden_states = self.critic_learned_model(obs, adj, critic_hidden_states)
                self.current_actor_hidden_states = next_actor_hidden_states
                self.current_critic_hidden_states = next_critic_hidden_states
        return action

    def _sample_action_from_p(self, p, epsilon, action_mode):
        """
        :param p: np.ndarray [n_agent, n_action]
        :param epsilon: float
        :param action_mode: str
        :return: action, np.ndarray [n_agent, ]
        """
        action = []
        assert p.ndim == 2
        n_agent = p.shape[0]
        if action_mode == 'epsilon-greedy':
            for i in range(n_agent):  # agent-wise epsilon-greedy
                if np.random.rand() < epsilon:
                    a = np.random.randint(self.act_dim)
                else:
                    a = p[i].argmax().item()
                action.append(a)
        elif action_mode == 'categorical':
            action = numba_categorical_sample(p)
        elif action_mode == 'epsilon-categorical':
            action = numba_epsilon_categorical_sample(p, epsilon)
        elif action_mode == 'greedy':
            for i in range(n_agent):
                a = p[i].argmax().item()
                action.append(a)
        else:
            raise ValueError
        action = np.array(action, dtype=np.float32).reshape([n_agent, ])
        return action

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        obs = sample['obs']
        adj = sample['adj']
        cri_hid = sample['cri_hid']
        action = sample['action']
        reward = sample['reward']
        next_obs = sample['next_obs']
        next_act_hid = sample['next_act_hid']
        next_cri_hid = sample['next_cri_hid']
        next_adj = sample['next_adj']
        done = sample['done']
        batch_size, n_ant, _ = obs.shape

        # q_values : [b,n_agent,n_action]
        q_values, _ = self.critic_learned_model(obs, adj, cri_hid)
        # target_q_values: [b,n_agent,]
        with torch.no_grad():
            # when calculating, use the fixed graph (in paper) or true dynamic graph (experimentally better)
            # next_probs, _ = self.actor_target_model(next_obs, next_adj, next_act_hid, return_log_pi=False)
            next_probs, _ = self.actor_learned_model(next_obs, next_adj, next_act_hid, return_log_pi=False)
            target_q_values, _ = self.critic_target_model(next_obs, next_adj, next_cri_hid)
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
        act_hid = sample['act_hid']
        cri_hid = sample['cri_hid']
        batch_size, n_ant, _ = obs.shape

        probs, _, _ = self.actor_learned_model(obs, adj, act_hid, return_log_pi=True)
        log_probs = torch.log(probs + 1e-15)  # [batch, agent, action]

        with torch.no_grad():
            # q_values: Q(s,a), [b, n_agent, n_action]
            q_values, _ = self.critic_learned_model(obs, adj, cri_hid)
            # baseline, V(s)=E_a[Q(s,a)]=\Sigma \pi(a|s)*Q(s,a)
            v_values = (probs * q_values).sum(dim=-1, keepdim=True)
            # advantage, A(s,a)=Q(s,a)-V(s)
            advantages = q_values - v_values
            # loss_p: \Sigma log\pi(a|s)*A(s,a)
        # p_loss = (-masked_log_probs * advantages).mean()
        p_loss = (-log_probs * advantages).mean()
        losses['p_loss'] = p_loss

    def update_target(self):
        if hparams['soft_update_target_network']:
            soft_update(self.actor_learned_model, self.actor_target_model, hparams['tau'])
            soft_update(self.critic_learned_model, self.critic_target_model, hparams['tau'])
        else:
            hard_update(self.actor_learned_model, self.actor_target_model)
            hard_update(self.critic_learned_model, self.critic_target_model)
