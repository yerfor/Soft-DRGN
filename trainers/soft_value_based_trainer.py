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


class SoftValueBasedTrainer(ValueBasedTrainer):
    """
    This is the Main Controller for training a *Soft-Value-based* DRL algorithm.
    """

    def __init__(self):
        self.work_dir = os.path.join("checkpoints", hparams['exp_name'])
        os.makedirs(self.work_dir, exist_ok=True)
        self.log_dir = self.work_dir if 'log_dir' not in hparams else os.path.join(self.work_dir, hparams['log_dir'])
        os.makedirs(self.log_dir, exist_ok=True)

        self.env = get_cls_from_path(hparams['scenario_path'])()
        self.agent = get_cls_from_path(hparams['algorithm_path'])(self.env.obs_dim, self.env.act_dim).cuda()
        self.replay_buffer = ReplayBuffer()
        self.optimizer = torch.optim.Adam(self.agent.learned_model.parameters(), lr=hparams['learning_rate'])
        self.optimizer_alpha = torch.optim.Adam([self.agent.alpha], lr=hparams['learning_rate'])
        self.tb_logger = TensorBoardLogger(self.log_dir)

        self.i_iter_critic = 0
        self.i_iter_alpha = 0
        self.i_episode = 0
        # Note that if work_dir already has config.yaml, it might override your manual setting!
        # So delete the old config.yaml when you want to do some modifications.
        self.load_from_checkpoint_if_possible()
        self.best_eval_reward = -1e15
        self.save_best_ckpt = False

    @property
    def i_iter_dict(self):
        return {'i_critic': self.i_iter_critic, 'i_alpha': self.i_iter_alpha, 'i_episode': self.i_episode}

    def _load_i_iter_dict(self, i_iter_dict):
        self.i_iter_critic = i_iter_dict['i_critic']
        self.i_iter_alpha = i_iter_dict['i_alpha']
        self.i_episode = i_iter_dict['i_episode']

    def _load_checkpoint(self, checkpoint):
        self.agent.load_state_dict(checkpoint['agent'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.optimizer_alpha.load_state_dict(checkpoint['optimizer_alpha'])
        self._load_i_iter_dict(checkpoint['i_iter_dict'])
        logging.info("Checkpoint loaded successfully!")

    def load_from_checkpoint_if_possible(self):
        ckpt, ckpt_path = get_last_checkpoint(self.work_dir)
        if ckpt is None:
            logging.info("No checkpoint found, learn the agent from scratch!")
        else:
            logging.info(f"Latest checkpoint found at f{ckpt_path}, try loading...")
            try:
                self._load_checkpoint(checkpoint=ckpt)
            except:
                logging.warning("Checkpoint loading failed, now learn from scratch!")

    def save_checkpoint(self):
        # before save checkpoint, first delete redundant old checkpoints
        all_ckpt_path = get_all_ckpts(self.work_dir)
        if len(all_ckpt_path) >= hparams['num_max_keep_ckpt'] - 1:
            ckpt_to_delete = all_ckpt_path[hparams['num_max_keep_ckpt'] - 1:]
            remove_files(ckpt_to_delete)
        ckpt_path = os.path.join(self.work_dir, f"model_ckpt_episodes_{self.i_episode}.ckpt")
        checkpoint = {}
        checkpoint['agent'] = self.agent.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['optimizer_alpha'] = self.optimizer_alpha.state_dict()
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

            losses = {}
            self.agent.cal_q_loss(batched_sample, losses, log_vars=log_vars, global_steps=self.i_iter_critic)
            total_loss = sum(losses.values())
            self.optimizer.zero_grad()
            total_loss.backward()
            for loss_name, loss in losses.items():
                log_vars[f'Training/{loss_name}'] = (self.i_iter_critic, loss.item())
            log_vars['Training/q_grad'] = (self.i_iter_critic, get_grad_norm(self.agent.learned_model, l=2))
            self.optimizer.step()
            self.i_iter_critic += 1

            entropy_losses = {}
            self.agent.cal_alpha_loss(batched_sample, entropy_losses, log_vars=log_vars, global_steps=self.i_iter_critic)
            total_entropy_loss = sum(entropy_losses.values())
            self.optimizer_alpha.zero_grad()
            total_entropy_loss.backward()
            for loss_name, loss in entropy_losses.items():
                log_vars[f'Training/{loss_name}'] = (self.i_iter_alpha, loss.item())
            self.agent.clip_alpha_grad(log_vars=log_vars, global_steps=self.i_iter_alpha)
            self.optimizer_alpha.step()
            self.i_iter_alpha += 1

            if self.i_iter_critic % 5 == 0:
                self.agent.update_target()

