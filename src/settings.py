from src.enums.initparamtype import InitParamType
from src.enums.trainingtype import TrainingType

import argparse


class GeneratorSettings(object):
    def __init__(self, num_qubits, depth, n_shots, paramtype, trainingtype, max_iter=50, step_rate=0.1):
        self.num_qubits = num_qubits
        self.depth = depth
        self.n_shots = n_shots
        self.paramtype = paramtype
        self.trainingtype = trainingtype 
        self.max_iter = max_iter 
        self.step_rate = step_rate


class DiscriminatorSettings(object):
    def __init__(self, num_qubits, depth, n_shots, paramtype, max_iter=80):
        self.num_qubits = num_qubits
        self.depth = depth
        self.n_shots = n_shots
        self.paramtype = paramtype
        self.max_iter = max_iter 


class DataSettings(object):
    def __init__(self, mu, sigma, batch_size, items_synthetic_size, items_real_size):
        self.mu = mu
        self.sigma = sigma
        self.batch_size = batch_size
        self.items_synthetic_size = items_synthetic_size
        self.items_real_size = items_real_size

def _init(repeats, items_s, items_r, g_num_qubits):
    # Generator specific settings
    global g_settings
    global d_settings
    global data_settings
    if repeats:
        data_settings.repeats = repeats
    if items_s:
        data_settings.items_synthetic_size = items_s
    if items_r:
        data_settings.items_real_size = items_r
    if g_num_qubits:
        g_settings.num_qubits = g_num_qubits
        d_settings.num_qubits = 2**g_num_qubits


def settings_init():
    parser = argparse.ArgumentParser(description='Simulate quantum generative adversarial network')
    parser.add_argument('-q','--qubits', type=int, help=f'Amount of qubits to use for generator (2^this is used for discriminator) (Default {g_settings.num_qubits})')
    parser.add_argument('-r', '--repeats', type=int, help=f'Amount of repeats to use for training')
    parser.add_argument('-is', '--items_synthetic', type=int, help=f'Amount of synthetic samples to process per network update (Default {data_settings.items_synthetic_size})')
    parser.add_argument('-it', '--items_real', type=int, help=f'Amount of real samples to process per network update (Default {data_settings.items_real_size})')
    # parser.add_argument('-exc','--excited_state', action='store_true', help=f'If set, searches for excited instead of ground state (default {default_excited_state})')
    args = parser.parse_args()
    _init(args.repeats, args.items_synthetic, args.items_real, args.qubits)

g_num_qubits = 3
g_depth = 2
g_n_shots = 6000
g_paramtype = InitParamType.RANDOM
trainingtype = TrainingType.ADAM
g_max_iter = 50
step_rate = 0.1
global g_settings
g_settings = GeneratorSettings(g_num_qubits, g_depth, g_n_shots, g_paramtype, trainingtype, g_max_iter, step_rate)

# Discriminator specific settings
d_num_qubits = 2**g_num_qubits
d_depth = 5
d_n_shots = 0
d_paramtype = InitParamType.RANDOM
d_max_iter = 80
global d_settings
d_settings = DiscriminatorSettings(d_num_qubits, d_depth, d_n_shots, d_paramtype, d_max_iter)

# Data settings
repeats = 10
mu = 1.0
sigma = 1.0
batch_size = 2000
items_synthetic_size = 100
items_real_size = 100
global data_settings
data_settings = DataSettings(mu, sigma, batch_size, items_synthetic_size, items_real_size)
