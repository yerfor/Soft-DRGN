import os
import torch
import numpy as np
from trainers.base_trainer import BaseTrainer

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


class ValueBasedTrainer(BaseTrainer):
    """
    This is the Main Controller for training a *Value-based* DRL algorithm.
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
        self.tb_logger = TensorBoardLogger(self.log_dir)

        self.i_iter_critic = 0
        self.i_episode = 0
        # Note that if word_dir already has config.yaml, it might override your manual setting!
        # So delete the old config.yaml when you want to do some modifications.
        self.load_from_checkpoint_if_possible()
        self.best_eval_reward = -1e15
        self.save_best_ckpt = False

    @property
    def i_iter_dict(self):
        return {'i_critic': self.i_iter_critic, 'i_episode': self.i_episode}

    def _load_i_iter_dict(self, i_iter_dict):
        self.i_iter_critic = i_iter_dict['i_critic']
        self.i_episode = i_iter_dict['i_episode']

    def _load_checkpoint(self, checkpoint):
        self.agent.load_state_dict(checkpoint['agent'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
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
        checkpoint['i_iter_dict'] = self.i_iter_dict
        torch.save(checkpoint, ckpt_path)
        if self.save_best_ckpt:
            ckpt_path = os.path.join(self.work_dir, f"model_ckpt_best.ckpt")
            torch.save(checkpoint, ckpt_path)

    def _interaction_step(self, log_vars):
        obs, adj = self.env.reset()
        self.i_episode += 1
        epsilon = epsilon_scheduler(self.i_episode)
        self.tb_logger.add_scalars({'Epsilon': (self.i_episode, epsilon)})
        if hasattr(self.agent, 'reset_hidden_states'):
            self.agent.reset_hidden_states(obs.shape[0])
        tmp_reward_lst = []
        for t in range(hparams['episode_length']):
            action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['training_action_mode'])
            reward, next_obs, next_adj, done = self.env.step(action)
            sample = {'obs': obs, 'adj': adj, 'action': action, 'reward': reward, 'next_obs': next_obs,
                      'next_adj': next_adj, 'done': done}
            if hasattr(self.agent, 'get_hidden_states'):
                sample.update(self.agent.get_hidden_states())
            self.replay_buffer.push(sample)
            obs, adj = next_obs, next_adj
            tmp_reward_lst.append(sum(reward))
        log_vars['Interaction/episodic_reward'] = (self.i_episode, sum(tmp_reward_lst))
        if hasattr(self.env, "get_log_vars"):
            tmp_env_log_vars = {f"Interaction/{k}": (self.i_episode, v) for k, v in self.env.get_log_vars().items()}
            log_vars.update(tmp_env_log_vars)

    def _training_step(self, log_vars):
        if not self.i_episode % hparams['training_interval'] == 0:
            return
        for _ in range(hparams['training_times']):
            self.i_iter_critic += 1
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

            if self.i_iter_critic % 5 == 0:
                self.agent.update_target()

    def _testing_step(self, log_vars):
        if not self.i_episode % hparams['testing_interval'] == 0:
            return
        episodic_reward_lst = []
        if hasattr(self.env, "get_log_vars"):
            episodic_env_log_vars = {}
        for _ in tqdm.tqdm(range(1, hparams['testing_episodes'] + 1), desc='Testing Episodes: '):
            obs, adj = self.env.reset()
            if hasattr(self.agent, 'reset_hidden_states'):
                self.agent.reset_hidden_states(obs.shape[0])
            tmp_reward_lst = []
            for t in range(hparams['episode_length']):
                epsilon = hparams['min_epsilon']
                action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
                reward, next_obs, next_adj, done = self.env.step(action)
                obs, adj = next_obs, next_adj
                tmp_reward_lst.append(sum(reward))
            episodic_reward_lst.append(sum(tmp_reward_lst))
            if hasattr(self.env, "get_log_vars"):
                tmp_env_log_vars = self.env.get_log_vars()
                for k, v in tmp_env_log_vars.items():
                    if k not in episodic_env_log_vars.keys():
                        episodic_env_log_vars[k] = []
                    episodic_env_log_vars[k].append(v)
        if hasattr(self.env, "get_log_vars"):
            episodic_env_log_vars = {f"Testing/{k}": (self.i_episode, np.mean(v)) for k, v in
                                     episodic_env_log_vars.items()}
            log_vars.update(episodic_env_log_vars)
        # Record the total reward obtain by all agents at each time step
        episodic_reward_mean = np.mean(episodic_reward_lst)
        episodic_reward_std = np.std(episodic_reward_lst)
        log_vars['Testing/mean_episodic_reward'] = (self.i_episode, episodic_reward_mean)
        log_vars['Testing/std_episodic_reward'] = (self.i_episode, episodic_reward_std)

        logging.info(
            f"Episode {self.i_episode} evaluation reward: mean {episodic_reward_mean},"
            f" std {episodic_reward_std}")
        # Save checkpoint when each testing phase is end.
        if episodic_reward_mean > self.best_eval_reward:
            self.save_best_ckpt = True
            logging.info(
                f"Best evaluation reward update: {self.best_eval_reward} ==> {episodic_reward_mean}")
            self.best_eval_reward = episodic_reward_mean
        else:
            self.save_best_ckpt = False
        self.save_checkpoint()

    def run_training_loop(self):
        start_episode = self.i_episode
        for _ in tqdm.tqdm(range(start_episode, hparams['num_episodes'] + 1), desc='Training Episode: '):
            log_vars = {}  # e.g. {'Training/q_loss':(16000, 0.999)}
            # Interaction Phase
            self._interaction_step(log_vars=log_vars)
            # Training Phase
            self._training_step(log_vars=log_vars)
            # Testing Phase
            self._testing_step(log_vars=log_vars)
            self.tb_logger.add_scalars(log_vars)

    def run_display_loop(self):
        while True:
            obs, adj = self.env.reset()
            if hasattr(self.agent, 'reset_hidden_states'):
                self.agent.reset_hidden_states(obs.shape[0])
            for t in range(hparams['episode_length']):
                self.env.render()
                epsilon = hparams['min_epsilon']
                action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
                reward, next_obs, next_adj, done = self.env.step(action)
                obs, adj = next_obs, next_adj

    def run_eval_loop(self):
        rew_array = np.zeros(shape=[hparams['eval_episodes']])
        for i_episode in tqdm.tqdm(range(0, hparams['eval_episodes']), desc='Eval Episodes: '):
            tmp_reward_lst = []
            obs, adj = self.env.reset()
            if hasattr(self.agent, 'reset_hidden_states'):
                self.agent.reset_hidden_states(obs.shape[0])
            for t in range(hparams['episode_length']):
                epsilon = hparams['min_epsilon']
                action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
                reward, next_obs, next_adj, done = self.env.step(action)
                obs, adj = next_obs, next_adj
                tmp_reward_lst.append(sum(reward))
            rew_array[i_episode] = sum(tmp_reward_lst) / hparams['episode_length']
        np.savetxt(os.path.join(self.work_dir, hparams['eval_result_name']), rew_array, delimiter=',')
        mean, std = rew_array.mean(), rew_array.std()
        logging.info(f"Evaluation complete, reward mean {mean}, std {std} .")
        logging.info(f"Evaluation result is saved at {os.path.join(self.work_dir, hparams['eval_result_name'])}.")

    def run(self):
        if hparams['display']:
            self.run_display_loop()
        elif hparams['evaluate']:
            self.run_eval_loop()
        else:
            self.run_training_loop()
