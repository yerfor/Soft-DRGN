import numpy as np
from utils.hparams import hparams


def epsilon_scheduler(i_episode):
    epsilon = hparams['initial_epsilon']
    if i_episode < hparams['burnin_episode']:
        return epsilon

    if hparams['epsilon_linear_decay']:
        epsilon = hparams['initial_epsilon'] - (i_episode - hparams['burnin_episode']) * hparams[
                                                'epsilon_decay_percent_episode']
    elif hparams['epsilon_exponential_decay']:
        epsilon = hparams['initial_epsilon'] * np.exp(-(i_episode - hparams['burnin_episode']) * hparams[
                                                        'epsilon_decay_temperature'])
    else:
        raise ValueError

    if epsilon < hparams['min_epsilon']:
        epsilon = hparams['min_epsilon']
    return epsilon
