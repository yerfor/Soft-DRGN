import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.ac_drgn import ActorCriticDRGNAgent
from agents.sac_dgn import soft_actor_critic_model_entropy_activation_function

from utils.hparams import hparams
from utils.numba_utils import *
from utils.torch_utils import *


class SoftActorCriticDRGNAgent(ActorCriticDRGNAgent):
    def __init__(self, in_dim, act_dim):
        super(SoftActorCriticDRGNAgent, self).__init__(in_dim, act_dim)
        self.alpha = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.target_entropy = -np.log(1.0 / self.act_dim) * hparams['entropy_target_factor']

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        obs = sample['obs']
        act_hid = sample['act_hid']
        cri_hid = sample['cri_hid']
        adj = sample['adj']
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
            # Soft Value Function: V(s) = E_a[Q'(s',a')-log\pi(a'|s')] = \Sigma (\pi(a'|s') * Q'(s',a') - log\pi(a'|s'))
            next_probs, _, next_log_probs = self.actor_learned_model(next_obs, next_adj, next_act_hid, return_log_pi=True)
            v_values, _ = self.critic_target_model(next_obs, next_adj, next_cri_hid)
            v_values = (next_probs * (v_values - self.alpha * next_log_probs)).sum(dim=-1)
            v_values = v_values.cpu().numpy()  # [batch, n_agent]
            target_q_values = v_values
        numpy_q_values = q_values.detach().cpu().numpy()
        expected_q = numba_get_expected_q(numpy_q_values, action.cpu().numpy(), reward.cpu().numpy(),
                                          done.cpu().numpy(), hparams['gamma'], target_q_values,
                                          batch_size, n_ant)

        # q_loss: MSE calculated on the sampled action index!
        q_loss = (q_values - torch.tensor(expected_q).cuda()).pow(2).mean()
        losses['q_loss'] = q_loss

    def cal_p_loss(self, sample, losses, log_vars=None, global_steps=None):
        obs = sample['obs']
        act_hid = sample['act_hid']
        cri_hid = sample['cri_hid']
        adj = sample['adj']
        batch_size, n_ant, _ = obs.shape

        probs,_, log_probs = self.actor_learned_model(obs, adj, act_hid, return_log_pi=True)
        log_probs = torch.log(probs + 1e-15)  # [batch, agent, action]

        with torch.no_grad():
            # q_values: Q(s,a), [b, n_agent, n_action]
            q_values, _ = self.critic_learned_model(obs, adj, cri_hid)
            # baseline, V(s)=E_a[Q(s,a)]=\Sigma \pi(a|s)*Q(s,a)
            v_values = (probs * q_values).sum(dim=-1, keepdim=True)
            # advantage, A(s,a)=Q(s,a)-V(s)
            advantages = q_values - v_values
        # p_loss: \Sigma log\pi(a|s) * (A(s,a) - log\pi(a|s))
        p_loss = -(log_probs * (advantages - self.alpha * log_probs)).mean()
        losses['p_loss'] = p_loss

    def cal_alpha_loss(self, sample, losses, log_vars=None, global_steps=None):
        obs = sample['obs']
        act_hid = sample['act_hid']
        adj = sample['adj']
        with torch.no_grad():
            probs, _, log_probs = self.actor_learned_model(obs, adj, act_hid, return_log_pi=True)
            entropies = (-probs * log_probs).sum(dim=-1, keepdim=True)  # [b,agent]
            if log_vars is not None:
                entropy = entropies.mean().item()
                log_vars['action_entropy'] = (global_steps, entropy)
        entropy_loss = (- soft_actor_critic_model_entropy_activation_function(self.alpha) * (self.target_entropy - entropies)).mean()
        losses['entropy_loss'] = entropy_loss

    def clip_alpha_grad(self, log_vars=None, global_steps=None):
        torch.nn.utils.clip_grad_norm_(self.alpha, max_norm=self.alpha.item()*0.01, norm_type=1)
        self.alpha.data = torch.max(self.alpha.data, torch.ones_like(self.alpha.data)*1e-5)
        if log_vars is not None:
            log_vars['alpha'] = (global_steps, self.alpha.item())

