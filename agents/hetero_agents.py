import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agents.dqn import DQNAgent
from modules.hgn import HGNNetwork

from utils.numba_utils import *
from utils.torch_utils import *
from utils.class_utils import get_cls_from_path
from utils.hparams import hparams
from copy import deepcopy


class HeteroValueBasedAgents(DQNAgent):
    """
    This class is used to transplant Homogeneous ValueBasedAgents (like DGN/Soft-DGN) into Heterogeneous Env.
    Specifically, we train a independent homogeneous agent for each group of agent.
    There is only intra-group communication. For inter-group comm, consider use HGATLayer.
    """
    def __init__(self, in_dims, act_dims):
        nn.Module.__init__(self)
        self.agent_class = get_cls_from_path(hparams['sub_algorithm_path'])
        self.in_dims = in_dims
        self.act_dims = act_dims
        self.hidden_dim = hparams['hidden_dim']
        self.num_head = hparams['num_head']
        self.skip_connect = hparams['skip_connect']
        self.num_group = len(in_dims)
        self.sub_agents = nn.ModuleList([
            self.agent_class(in_dims[i], act_dims[i]) for i in range(self.num_group)])
        self.learned_model = nn.ModuleList([agent.learned_model for agent in self.sub_agents])
        self.target_model = nn.ModuleList([agent.target_model for agent in self.sub_agents])

    def action(self, obs_dict, adj_dict, epsilon=0.3, action_mode='epsilon-greedy'):
        obs_lst = list(obs_dict.values())
        adj_lst = [adj_dict[f'adj_{i}_{i}'] for i in range(self.num_group)]

        with torch.no_grad():
            action = {}
            for group_i in range(self.num_group):
                act_i = self.sub_agents[group_i].action(obs_lst[group_i], adj_lst[group_i], epsilon, action_mode)
                action['act_' + str(group_i)] = np.array(act_i)
        return action

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        losses_list = [{} for _ in range(self.num_group)]
        total_loss = 0
        for group_i in range(self.num_group):
            sample_i = {}
            sample_i['obs'] = sample[f'obs_{group_i}']
            sample_i['adj'] = sample[f'adj_{group_i}_{group_i}']
            sample_i['action'] = sample[f'act_{group_i}']
            sample_i['reward'] = sample[f'rew_{group_i}']
            sample_i['done'] = sample[f'done_{group_i}']
            sample_i['next_obs'] = sample[f'next_obs_{group_i}']
            sample_i['next_adj'] = sample[f'next_adj_{group_i}_{group_i}']
            if 'cri_hid_0' in sample.keys():
                sample_i['cri_hid'] = sample[f'cri_hid_{group_i}']
                sample_i['next_cri_hid'] = sample[f'next_cri_hid_{group_i}']
            if 'act_hid_0' in sample.keys():
                sample_i['act_hid'] = sample[f'act_hid_{group_i}']
                sample_i['next_act_hid'] = sample[f'next_act_hid_{group_i}']

            self.sub_agents[group_i].cal_q_loss(sample_i, losses_list[group_i], None, None)
            losses_list[group_i] = {f"group{group_i}_" + k: v for k, v in losses_list[group_i].items()}
            total_loss += sum((v.item() for v in losses_list[group_i].values()))
        log_vars['Training/total_q_loss'] = (global_steps, total_loss)
        for loss_dict in losses_list:
            losses.update(loss_dict)

    def update_target(self, soft=False):
        if soft:
            soft_update(self.learned_model, self.target_model, self.config.tau)
        else:
            hard_update(self.learned_model, self.target_model)
