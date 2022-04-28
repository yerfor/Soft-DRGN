import torch
import numpy as np
from utils.hparams import hparams
from utils.torch_utils import *
import logging


class ReplayBuffer:
    def __init__(self):
        self.capacity = int(hparams['buffer_capacity'])
        self.current_idx = 0  # should be within [0, capacity-1]
        self.num_stored_sample = 0  # how many samples is in the buffer, should be within [0, capacity]
        self.buffer = None  # we use a dict to store all features

    def _initialize_buffer_with_an_example_sample(self, sample):
        self.buffer = {}
        for k, v in sample.items():
            # initialize the buffer
            if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                v = np.squeeze(v) if isinstance(v, np.ndarray) else torch.squeeze(v)
                if v.ndim == 1:
                    # reward, done, discrete_action: [n_agent]
                    n_agent = v.shape[0]
                    self.buffer[k] = torch.zeros(size=[self.capacity, n_agent], dtype=torch.float32)
                elif v.ndim == 2:
                    # obs, next_obs: [n_agent, obs_dim]; adj, next_adj: [n_agent, n_agent]
                    # continuous_action: [n_agent, action_dim]
                    n_agent, hid_dim = v.shape
                    self.buffer[k] = torch.zeros(size=[self.capacity, n_agent, hid_dim], dtype=torch.float32)
                elif v.ndim == 4:
                    # img_obs: [n_agent, hid_dim, x_dim, y_dim]
                    n_agent, hid_dim, x_dim, y_dim = v.shape
                    self.buffer[k] = torch.zeros(size=[self.capacity, n_agent, hid_dim, x_dim, y_dim],
                                                 dtype=torch.float32)
            elif isinstance(v, dict):
                # we also support complicated feature, you can represented it as a dict of np.ndarray
                # e.g. sample['obs'] = {'conv_obs': [n_agent, hid_dim, x_dim, y_dim], 'linear_obs': [n_agent, hid_dim]}
                prefix = 'next_' if 'next' in k else ''
                for k_, v_ in v.items():
                    k_ = prefix + k_
                    if isinstance(v_, np.ndarray) or isinstance(v_, torch.Tensor):
                        v_ = np.squeeze(v_) if isinstance(v_, np.ndarray) else torch.squeeze(v_)
                    elif isinstance(v_, dict):
                        raise TypeError("Now we only support one-layer dict, data like {'xx':{} ,} is not supported!")
                    else:
                        raise TypeError(f"We found Unsupported Type {type(v_)} in dict feature {k_}!")
                    if v_.ndim == 1:
                        n_agent = v_.size
                        self.buffer[k_] = torch.zeros(size=[self.capacity, n_agent], dtype=torch.float32)
                    elif v_.ndim == 2:
                        n_agent, hid_dim = v_.shape
                        self.buffer[k_] = torch.zeros(size=[self.capacity, n_agent, hid_dim], dtype=torch.float32)
                    elif v_.ndim == 4:
                        n_agent, hid_dim, x_dim, y_dim = v_.shape
                        self.buffer[k_] = torch.zeros(size=[self.capacity, n_agent, hid_dim, x_dim, y_dim],
                                                      dtype=torch.float32)
            else:
                raise TypeError("Unsupported type!")
        # print the detailed structure of the replay buffer
        logging.info("\nThe details of the replay buffer are as follows...")
        for k, v in self.buffer.items():
            logging.info(f"{k}, shape={v.shape}.")
        logging.info("Successfully constructed the replay buffer.")

    def push(self, sample):
        """
        :param sample: dict. eg: {'obs':np.array, 'action': np.array}
        :return: None
        """
        if self.buffer is None:
            self._initialize_buffer_with_an_example_sample(sample)

        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                torch_v = torch.tensor(v, dtype=torch.float32)
                self.buffer[k][self.current_idx] = torch_v
            elif isinstance(v, torch.Tensor):
                self.buffer[k][self.current_idx] = v
            elif isinstance(v, dict):
                prefix = 'next_' if 'next' in k else ''
                for k_, v_ in v.items():
                    k_ = prefix + k_
                    assert isinstance(v_, np.ndarray)
                    torch_v_ = torch.tensor(v_, dtype=torch.float32)
                    self.buffer[k_][self.current_idx] = torch_v_
            else:
                raise TypeError("Unsupported v type!")
        self.current_idx = (self.current_idx + 1) % self.capacity
        if self.num_stored_sample < self.capacity:
            self.num_stored_sample += 1

    def sample(self, batch_size):
        """
        :param batch_size: int
        :return:
        """
        if self.num_stored_sample < batch_size:
            return None
        sample = {}
        sample_idx = torch.randint(low=0, high=self.num_stored_sample, size=[batch_size, ])
        for k, v in self.buffer.items():
            cuda_sampled_v = to_tensor(v[sample_idx]).cuda()
            sample[k] = cuda_sampled_v
        return sample

    def __len__(self):
        return self.num_stored_sample
