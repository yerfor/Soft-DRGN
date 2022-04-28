import random

import numba
import numpy as np


@numba.njit
def numba_manual_seed(seed):
    np.random.seed(seed)


@numba.njit
def numba_categorical_sample(probs):
    # probs: A matrix likes [n_agent, n_action], normalized in dim=1
    action = []
    for i in range(probs.shape[0]):
        logit = random.random()
        sum_val = 0
        for a, val in enumerate(probs[i]):
            sum_val += val
            if logit <= sum_val:
                action.append(a)
                break
        if logit > sum_val:
            action.append(probs.shape[1] - 1)
    return action


@numba.njit
def numba_epsilon_categorical_sample(probs, epsilon):
    # probs: A matrix likes [n_agent, n_action], normalized in dim=1
    action = []
    for i in range(probs.shape[0]):
        if np.random.rand() < epsilon:
            a = np.random.randint(probs.shape[1])
            action.append(a)
        else:
            logit = random.random()
            sum_val = 0
            for a, val in enumerate(probs[i]):
                sum_val += val
                if logit <= sum_val:
                    action.append(a)
                    break
            if logit > sum_val:
                action.append(probs.shape[1] - 1)
    return action


@numba.njit()
def numba_get_action_mask(action_mask, action_arr, batch_size, n_agent):
    # action_arr : array [batch,n_agent]
    for j in range(batch_size):
        for i in range(n_agent):
            # sample[1][i]: action of agent i; target_q_values[j][i]: target q value of agent i
            action_i = action_arr[j][i]
            action_mask[j, i, action_i] = 1.0
    return action_mask


@numba.jit()
def numba_get_expected_q(expected_q, actions, rewards, dones, gamma, v_values, batch_size, n_ant):
    for j in range(batch_size):
        for i in range(n_ant):
            action_i = int(actions[j][i])
            expected_q[j][i][action_i] = rewards[j][i] + (1 - dones[j][i]) * gamma * v_values[j][i]
    return expected_q
