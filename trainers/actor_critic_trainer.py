import os
import torch
import numpy as np
from trainers.value_based_trainer import ValueBasedTrainer

from utils.scheduler import epsilon_scheduler
from utils.hparams import hparams
from utils.replay_buffer import ReplayBuffer
from utils.tb_logger import TensorBoardLogger
from utils.torch_utils import *
from utils.class_utils import *
from utils.checkpoint_utils import *
from utils.os_utils import *

import logging
import tqdm


class ActorCriticTrainer(ValueBasedTrainer):
    """
    This is the Main Controller for training an *Actor-Critic* DRL algorithm.
    """

    def __init__(self):
        self.work_dir = os.path.join("checkpoints", hparams['exp_name'])
        os.makedirs(self.work_dir, exist_ok=True)
        self.log_dir = self.work_dir if 'log_dir' not in hparams else os.path.join(self.work_dir, hparams['log_dir'])
        os.makedirs(self.log_dir, exist_ok=True)

        self.env = get_cls_from_path(hparams['scenario_path'])()
        self.agent = get_cls_from_path(hparams['algorithm_path'])(self.env.obs_dim, self.env.act_dim).cuda()
        self.replay_buffer = ReplayBuffer()
        self.actor_optimizer = torch.optim.Adam(self.agent.actor_learned_model.parameters(),
                                                lr=hparams['learning_rate'])
        self.critic_optimizer = torch.optim.Adam(self.agent.critic_learned_model.parameters(),
                                                 lr=hparams['learning_rate'])
        self.tb_logger = TensorBoardLogger(self.log_dir)

        self.i_iter_actor = 0
        self.i_iter_critic = 0
        self.i_episode = 1
        # Note that if word_dir already has config.yaml, it will override your manual setting!
        # So delete the old config.yaml when you want to do some modifications.
        self.load_from_checkpoint_if_possible()
        self.best_eval_reward = -1e15
        self.save_best_ckpt = False

    @property
    def i_iter_dict(self):
        return {'i_actor': self.i_iter_actor, 'i_critic': self.i_iter_critic, 'i_episode': self.i_episode}

    def _load_i_iter_dict(self, i_iter_dict):
        self.i_iter_actor = i_iter_dict['i_actor']
        self.i_iter_critic = i_iter_dict['i_critic']
        self.i_episode = i_iter_dict['i_episode']

    def _load_checkpoint(self, checkpoint):
        self.agent.load_state_dict(checkpoint['agent'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self._load_i_iter_dict(checkpoint['i_iter_dict'])
        logging.info("Checkpoint loaded successfully!")

    def save_checkpoint(self):
        # before save checkpoint, first delete redundant old checkpoints
        all_ckpt_path = get_all_ckpts(self.work_dir)
        if len(all_ckpt_path) >= hparams['num_max_keep_ckpt'] - 1:
            ckpt_to_delete = all_ckpt_path[hparams['num_max_keep_ckpt'] - 1:]
            remove_files(ckpt_to_delete)
        ckpt_path = os.path.join(self.work_dir, f"model_ckpt_episodes_{self.i_episode}.ckpt")
        checkpoint = {}
        checkpoint['agent'] = self.agent.state_dict()
        checkpoint['actor_optimizer'] = self.actor_optimizer.state_dict()
        checkpoint['critic_optimizer'] = self.critic_optimizer.state_dict()
        checkpoint['i_iter_dict'] = self.i_iter_dict
        torch.save(checkpoint, ckpt_path)

    def _training_step(self, log_vars):
        if not self.i_episode % hparams['training_interval'] == 0:
            return
        for _ in range(hparams['training_times']):
            batched_sample = self.replay_buffer.sample(hparams['batch_size'])
            if batched_sample is None:
                # The replay buffer has not store enough sample.
                break
            q_losses = {}
            self.agent.cal_q_loss(batched_sample, q_losses, log_vars=log_vars, global_steps=self.i_iter_critic)
            total_q_loss = sum(q_losses.values())
            self.critic_optimizer.zero_grad()
            total_q_loss.backward()
            for loss_name, loss in q_losses.items():
                log_vars[f'Training/{loss_name}'] = (self.i_iter_critic, loss.item())
            log_vars['Training/q_grad'] = (self.i_iter_critic, get_grad_norm(self.agent.critic_learned_model, l=2))
            self.critic_optimizer.step()
            self.i_iter_critic += 1

            p_losses = {}
            self.agent.cal_p_loss(batched_sample, p_losses, log_vars=log_vars, global_steps=self.i_iter_actor)
            total_p_loss = sum(p_losses.values())
            self.actor_optimizer.zero_grad()
            total_p_loss.backward()
            for loss_name, loss in p_losses.items():
                log_vars[f'Training/{loss_name}'] = (self.i_iter_actor, loss.item())
            log_vars['Training/p_grad'] = (self.i_iter_actor, get_grad_norm(self.agent.actor_learned_model, l=2))
            self.actor_optimizer.step()
            self.i_iter_actor += 1

            if self.i_iter_critic % 5 == 0:
                self.agent.update_target()
