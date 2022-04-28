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


class HeteroValueBasedTrainer(ValueBasedTrainer):
    """
    This is the Main Controller for training a Heterogeneous *Value-based* DRL algorithm.
        Supported Agents: DQN, CommNet, HGN, HGRN.
        Supported Scenarios: CTC.
        To support Actor-Critic agents or Soft Agents:
            you can implement a similar Trainer from ActorCriticTrainer or SoftValueBased Trainer!.
    """

    def __init__(self):
        super(HeteroValueBasedTrainer, self).__init__()

    def _interaction_step(self, log_vars):
        obs, adj = self.env.reset()
        self.num_group = len(obs)
        self.i_episode += 1
        epsilon = epsilon_scheduler(self.i_episode)
        self.tb_logger.add_scalars({'Epsilon': (self.i_episode, epsilon)})
        if hasattr(self.agent, 'reset_hidden_states'):
            self.agent.reset_hidden_states(obs.shape[0])
        episodic_reward_array = np.zeros(shape=[len(obs)])  # for each group
        for t in range(hparams['episode_length']):
            action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['training_action_mode'])
            reward, next_obs, next_adj, done = self.env.step(action)
            sample = {'obs': obs, 'adj': adj, 'action': action, 'reward': reward, 'next_obs': next_obs,
                      'next_adj': next_adj, 'done': done}
            if hasattr(self.agent, 'get_hidden_states'):
                sample.update(self.agent.get_hidden_states())
            self.replay_buffer.push(sample)
            obs, adj = next_obs, next_adj
            group_current_rewards = np.array([r.sum() for r in reward.values()])
            episodic_reward_array += group_current_rewards
        log_vars['Interaction/episodic_reward'] = (self.i_episode, episodic_reward_array.sum())
        for group_i in range(self.num_group):  # record each group's reward
            log_vars[f'Interaction/group{group_i}_episodic_reward'] = (self.i_episode, episodic_reward_array[group_i])
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
        episodic_reward_array = np.zeros(shape=[self.num_group])  # for each group
        if hasattr(self.env, "get_log_vars"):
            episodic_env_log_vars = {}
        for _ in tqdm.tqdm(range(1, hparams['testing_episodes'] + 1), desc='Testing Episodes: '):
            obs, adj = self.env.reset()
            if hasattr(self.agent, 'reset_hidden_states'):
                self.agent.reset_hidden_states(obs.shape[0])
            for t in range(hparams['episode_length']):
                epsilon = hparams['min_epsilon']
                action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
                reward, next_obs, next_adj, done = self.env.step(action)
                group_current_rewards = np.array([r.sum() for r in reward.values()])
                episodic_reward_array += group_current_rewards / hparams['testing_episodes']
                obs, adj = next_obs, next_adj
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
        log_vars['Testing/episodic_reward'] = (self.i_episode, episodic_reward_array.sum())
        for group_i in range(self.num_group):  # record each group's reward
            log_vars[f'Testing/group{group_i}_episodic_reward'] = (self.i_episode, episodic_reward_array[group_i])

        logging.info(
            f"Episode {self.i_episode} evaluation reward: mean {episodic_reward_array.sum()}")
        # Save checkpoint when each testing phase is end.
        if episodic_reward_array.sum() > self.best_eval_reward:
            self.save_best_ckpt = True
            logging.info(
                f"Best evaluation reward update: {self.best_eval_reward} ==> {episodic_reward_array.sum()}")
            self.best_eval_reward = episodic_reward_array.sum()
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
