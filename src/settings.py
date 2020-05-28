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
    def __init__(self, num_qubits, depth, n_shots, paramtype, max_iter, d_type):
        self.num_qubits = num_qubits
        self.depth = depth
        self.n_shots = n_shots
        self.paramtype = paramtype
        self.max_iter = max_iter 
        self.type = d_type


class DataSettings(object):
    def __init__(self, mu, sigma, batch_size, items_synthetic_size, items_real_size, train_synthetic_size):
        self.mu = mu
        self.sigma = sigma
        self.batch_size = batch_size
        self.items_synthetic_size = items_synthetic_size
        self.items_real_size = items_real_size
        self.train_synthetic_size = train_synthetic_size


def _init(repeats, items_s, items_r, g_num_qubits, d_type):
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
    if d_type:
         d_settings.num_qubits = (2**g_settings.num_qubits if d_type == 2 else g_settings.num_qubits)
         d_settings.type = d_type

def settings_init():
    parser = argparse.ArgumentParser(description='Simulate quantum generative adversarial network')
    parser.add_argument('-q','--qubits', type=int, help=f'Amount of qubits to use for generator (2^this is used for discriminator) (Default {g_settings.num_qubits})')
    parser.add_argument('-r', '--repeats', type=int, help=f'Amount of repeats to use for training')
    parser.add_argument('-is', '--items_synthetic', type=int, help=f'Amount of synthetic samples to process per network update (Default {data_settings.items_synthetic_size})')
    parser.add_argument('-it', '--items_real', type=int, help=f'Amount of real samples to process per network update (Default {data_settings.items_real_size})')
    parser.add_argument('-dt', '--discriminator_type', type=int, choices=[1,2], help=f'Type to use for discriminator, as explained by Casper. Note that type 1 uses 2^qubits qubits, while type 2 uses just qubits qubits (Default {d_settings.type})')
    args = parser.parse_args()
    _init(args.repeats, args.items_synthetic, args.items_real, args.qubits, args.discriminator_type)

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
d_max_iter = 10
d_type = 2
global d_settings
d_settings = DiscriminatorSettings(d_num_qubits, d_depth, d_n_shots, d_paramtype, d_max_iter, d_type)

# Data settings
repeats = 10
mu = 1.0
sigma = 1.0
batch_size = 2000
items_synthetic_size = 100
items_real_size = 100

train_synthetic_size = 100
global data_settings
data_settings = DataSettings(mu, sigma, batch_size, items_synthetic_size, items_real_size, train_synthetic_size)
