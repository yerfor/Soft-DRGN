import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.hgn import HGNNetwork

from utils.numba_utils import *
from utils.torch_utils import *
from utils.hparams import hparams
from copy import deepcopy


class HGNAgent(nn.Module):
    def __init__(self, in_dims, act_dims):
        super(HGNAgent, self).__init__()
        self.in_dims = in_dims
        self.act_dims = act_dims
        self.hidden_dim = hparams['hidden_dim']
        self.num_head = hparams['num_head']
        self.skip_connect = hparams['skip_connect']
        self.num_group = len(in_dims)
        self.learned_model = HGNNetwork(in_dims, self.hidden_dim, act_dims, self.num_head, self.skip_connect).cuda()
        self.target_model = HGNNetwork(in_dims, self.hidden_dim, act_dims, self.num_head, self.skip_connect).cuda()
        self.inter_group_comm_network_enabled = False

    def _init_inter_group_comm_network_from_an_example_adjs(self, adjs):
        """
        adjs: dict of array, we use its keys to calculated the connection between all groups
        """
        self.groupwise_connection = [[] for _ in range(self.num_group)]
        for k, v in adjs.items():
            begin_node = int(k.split('_')[1])
            end_node = int(k.split('_')[2])
            self.groupwise_connection[begin_node].append(end_node)
        self.learned_model.init_from_groupwise_connection(self.groupwise_connection)
        self.target_model.init_from_groupwise_connection(self.groupwise_connection)
        self.target_model.load_state_dict(self.learned_model.state_dict())
        self.inter_group_comm_network_enabled = True

    def action(self, obs_dict, adj_dict, epsilon=0.3, action_mode='epsilon-greedy'):
        if not self.inter_group_comm_network_enabled:
            self._init_inter_group_comm_network_from_an_example_adjs(adj_dict)
        obs_dict_ = {}  # we build a copy, to prevent edit the array type in original obs_dict
        adj_dict = deepcopy(adj_dict)
        for k, v in obs_dict.items():
            if isinstance(v, np.ndarray):
                # from environment
                assert v.ndim == 2
                obs_dict_[k] = torch.tensor(v, dtype=torch.float32).unsqueeze(0).cuda()
            elif isinstance(v, torch.Tensor):
                # from replay buffer
                assert v.ndim == 3
                obs_dict_[k] = to_cuda(v)
            else:
                raise TypeError
        obs_lst = list(obs_dict_.values())

        for k, v in adj_dict.items():
            if isinstance(v, np.ndarray):
                # from environment
                assert v.ndim == 2
                adj_dict[k] = torch.tensor(v, dtype=torch.float32).unsqueeze(0).cuda()
            elif isinstance(v, torch.Tensor):
                # from replay buffer
                assert v.ndim == 3
                adj_dict[k] = to_cuda(v)
            else:
                raise TypeError

        adj_lst = self._parse_adj_dict_to_lst(adj_dict, prefix='adj')
        with torch.no_grad():
            action = {}
            qs = self.learned_model(obs_lst, adj_lst)
            for group_i in range(self.num_group):
                q_i = qs[group_i].squeeze().cpu().numpy()
                act_i = self._sample_action_from_q_values(q_i, epsilon, action_mode)
                action['act_' + str(group_i)] = np.array(act_i)
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
        n_agent, act_dim = q_values.shape
        if action_mode == 'epsilon-greedy':
            for i in range(n_agent):  # agent-wise epsilon-greedy
                if np.random.rand() < epsilon:
                    a = np.random.randint(act_dim)
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

    def _parse_adj_dict_to_lst(self, sample, prefix):
        adj_lst = [[None for _ in range(self.num_group)] for _ in range(self.num_group)]
        for i in range(self.num_group):
            for j in self.groupwise_connection[i]:
                adj_name = prefix + '_' + str(i) + '_' + str(j)
                adj_lst[i][j] = sample[adj_name]
        return adj_lst

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        obs = [sample['obs_' + str(i)] for i in range(self.num_group)]
        adj = self._parse_adj_dict_to_lst(sample, prefix='adj')
        act = [sample['act_' + str(i)] for i in range(self.num_group)]
        rew = [sample['rew_' + str(i)] for i in range(self.num_group)]
        done = [sample['done_' + str(i)] for i in range(self.num_group)]
        next_obs = [sample['next_obs_' + str(i)] for i in range(self.num_group)]
        next_adj = self._parse_adj_dict_to_lst(sample, prefix='next_adj')

        # q_values: Q(s,a), [b, n_agent, n_action]
        q_values = self.learned_model(obs, adj)
        with torch.no_grad():
            target_q_values = self.target_model(next_obs, next_adj)
            target_q_values = [t_g_v.max(dim=2)[0].cpu().numpy() for t_g_v in target_q_values]

        numpy_q_values = [q.detach().cpu().numpy() for q in q_values]

        for i in range(self.num_group):
            batch_size, n_agent_of_group_i, _ = q_values[i].shape
            expected_q_i = numba_get_expected_q(numpy_q_values[i], act[i].cpu().numpy(), rew[i].cpu().numpy(),
                                                done[i].cpu().numpy(), hparams['gamma'], target_q_values[i], batch_size,
                                                n_agent_of_group_i)
            expected_q_i = torch.tensor(expected_q_i).cuda()
            loss_i = (q_values[i] - expected_q_i).pow(2).mean()
            losses[f"loss_group_{i}"] = loss_i

    def update_target(self, soft=False):
        if soft:
            soft_update(self.learned_model, self.target_model, self.config.tau)
        else:
            hard_update(self.learned_model, self.target_model)
