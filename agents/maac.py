import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.sac_dgn import SoftActorCriticDGNAgent
from modules.maac import ActorMAACNetwork, CriticMAACNetwork
from agents.sac_dgn import soft_actor_critic_model_entropy_activation_function

from utils.hparams import hparams
from utils.numba_utils import *
from utils.torch_utils import *


class MAACAgent(SoftActorCriticDGNAgent):
    def __init__(self, in_dim, act_dim):
        nn.Module.__init__(self)
        self.in_dim = in_dim
        self.hidden_dim = hparams['hidden_dim']
        self.act_dim = act_dim
        self.num_head = hparams['num_head']

        self.actor_visibility = hparams['actor_visibility']
        self.critic_visibility = hparams['critic_visibility']
        self.n_agent = 0
        self.batch_size = 0
        self.actor_adj = None
        self.next_actor_adj = None
        self.critic_adj = None
        self.next_critic_adj = None

        self.actor_learned_model = ActorMAACNetwork(in_dim, hparams['hidden_dim'], act_dim, hparams['num_head'],
                                                   hparams['skip_connect'])
        self.actor_target_model = ActorMAACNetwork(in_dim, hparams['hidden_dim'], act_dim, hparams['num_head'],
                                                  hparams['skip_connect'])
        self.actor_target_model.load_state_dict(self.actor_learned_model.state_dict())

        self.critic_learned_model = CriticMAACNetwork(in_dim, hparams['hidden_dim'], act_dim, hparams['num_head'],
                                                     hparams['skip_connect'])
        self.critic_target_model = CriticMAACNetwork(in_dim, hparams['hidden_dim'], act_dim, hparams['num_head'],
                                                    hparams['skip_connect'])
        self.critic_target_model.load_state_dict(self.critic_learned_model.state_dict())

        self.alpha = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.target_entropy = -np.log(1.0 / self.act_dim) * hparams['entropy_target_factor']

    def action(self, obs, adj, epsilon=0.3, action_mode='epsilon-greedy'):
        actor_adj = self.get_actor_critic_adj(adj, adj)
        action = super(SoftActorCriticDGNAgent, self).action(obs, actor_adj, epsilon, action_mode)
        return action

    @staticmethod
    def _get_adj_by_visibility(adj, visibility):
        bs, n_agent, _ = adj.shape
        if visibility == 'no_graph':
            return torch.eye(n_agent, dtype=torch.float32).cuda().unsqueeze(0).repeat([bs, 1, 1])
        elif visibility == 'adj_graph':
            return adj
        elif visibility == 'full_graph':
            return torch.ones([bs, n_agent, n_agent], dtype=torch.float32).cuda()

    def get_actor_critic_adj(self, adj, next_adj):
        if adj.ndim == 2:
            # for interaction
            n_agent = adj.shape[-1]
            return np.eye(n_agent, dtype=np.float32)

        bs, n_agent, _ = adj.shape
        if n_agent != self.n_agent or bs != self.batch_size:
            self.n_agent = n_agent
            self.batch_size = bs
            self.actor_adj = self._get_adj_by_visibility(adj, self.actor_visibility)
            self.next_actor_adj = self._get_adj_by_visibility(next_adj, self.actor_visibility)
            self.critic_adj = self._get_adj_by_visibility(adj, self.critic_visibility)
            self.next_critic_adj = self._get_adj_by_visibility(next_adj, self.critic_visibility)
        return self.actor_adj, self.critic_adj, self.next_actor_adj, self.next_critic_adj

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        obs = sample['obs']
        adj = sample['adj']
        action = sample['action']
        reward = sample['reward']
        next_obs = sample['next_obs']
        next_adj = sample['next_adj']
        actor_adj, critic_adj, next_actor_adj, next_critic_adj, = self.get_actor_critic_adj(adj, next_adj)
        done = sample['done']
        batch_size, n_ant, _ = obs.shape

        # q_values : [b,n_agent,n_action]
        q_values = self.critic_learned_model(obs, critic_adj)
        # target_q_values: [b,n_agent,]
        with torch.no_grad():
            # Soft Value Function: V(s) = E_a[Q'(s',a')-log\pi(a'|s')] = \Sigma (\pi(a'|s') * Q'(s',a') - log\pi(a'|s'))
            next_probs, next_log_probs = self.actor_learned_model(next_obs, next_actor_adj, return_log_pi=True)
            v_values = self.critic_target_model(next_obs, next_critic_adj)
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
        adj = sample['adj']
        actor_adj, critic_adj, _, _, = self.get_actor_critic_adj(adj, adj)

        batch_size, n_ant, _ = obs.shape

        probs, log_probs = self.actor_learned_model(obs, actor_adj, return_log_pi=True)
        log_probs = torch.log(probs + 1e-15)  # [batch, agent, action]

        with torch.no_grad():
            # q_values: Q(s,a), [b, n_agent, n_action]
            q_values = self.critic_learned_model(obs, critic_adj)
            # baseline, V(s)=E_a[Q(s,a)]=\Sigma \pi(a|s)*Q(s,a)
            v_values = (probs * q_values).sum(dim=-1, keepdim=True)
            # advantage, A(s,a)=Q(s,a)-V(s)
            advantages = q_values - v_values
        # p_loss: \Sigma log\pi(a|s) * (A(s,a) - log\pi(a|s))
        p_loss = -(log_probs * (advantages - self.alpha * log_probs)).mean()
        losses['p_loss'] = p_loss

    def cal_alpha_loss(self, sample, losses, log_vars=None, global_steps=None):
        obs = sample['obs']
        adj = sample['adj']
        actor_adj, critic_adj, _, _, = self.get_actor_critic_adj(adj, adj)

        with torch.no_grad():
            probs, log_probs = self.actor_learned_model(obs, actor_adj, return_log_pi=True)
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

