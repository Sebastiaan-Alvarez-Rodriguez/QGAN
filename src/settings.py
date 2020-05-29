from src.enums.initparamtype import InitParamType
from src.enums.trainingtype import TrainingType

import logging
import argparse

'''
The main purpose of this file is to provide global settings for many categories.
Also, commandline argument parsing occurs here, as well as logging initialization.

In other files you may find statements as 'from src.settings import g_settings as gs'
Those statements include settings of either (g)enerator, (d)iscriminator, (data), (t)raining
'''
class GeneratorSettings(object):
    def __init__(self, num_qubits, depth, n_shots, paramtype, trainingtype, max_iter, step_rate):
        self.num_qubits = num_qubits
        self.depth = depth
        self.n_shots = n_shots
        self.paramtype = paramtype
        self.trainingtype = trainingtype 
        self.max_iter = max_iter 
        self.step_rate = step_rate


class DiscriminatorSettings(object):
    def __init__(self, num_qubits, depth, n_shots, paramtype, trainingtype, max_iter, d_type, step_rate):
        self.num_qubits = num_qubits
        self.depth = depth
        self.n_shots = n_shots
        self.paramtype = paramtype
        self.trainingtype = trainingtype
        self.max_iter = max_iter 
        self.type = d_type
        self.step_rate = step_rate


class DataSettings(object):
    def __init__(self, mu, sigma, batch_size, synthetic_size, real_size, gen_size):
        self.mu = mu
        self.sigma = sigma
        self.batch_size = batch_size
        self.synthetic_size = synthetic_size
        self.real_size = real_size
        self.gen_size = gen_size


class TrainingSettings(object):
    def __init__(self, repeats, print_accuracy, show_figs):
        self.repeats = repeats
        self.print_accuracy = print_accuracy
        self.show_figs = show_figs


def _init(args):
    # Generator specific settings
    global g_settings
    global d_settings
    global data_settings
    global t_settings

    if args.qubits:
        g_settings.num_qubits = args.qubits
        d_settings.num_qubits = 2**args.qubits
    if args.repeats:
        t_settings.repeats = args.repeats
    if args.synthetic:
        data_settings.synthetic_size = args.synthetic
    if args.real:
        data_settings.real_size = args.real
    if args.generator:
        data_settings.gen_size = args.generator
    if args.iter_discriminator:
        d_settings.max_iter = args.iter_discriminator
    if args.iter_generator:
        g_settings.max_iter = args.iter_generator
    
    if args.discriminator_type:
         d_settings.num_qubits = (2**g_settings.num_qubits if args.discriminator_type == 2 else g_settings.num_qubits)
         d_settings.type = args.discriminator_type
    if args.log:
        logging.basicConfig(format='%(asctime)s - %(name)s(%(levelname)s): %(message)s', level=getattr(logging, args.log.upper()))
    else:
        logging.basicConfig(format='%(asctime)s - %(name)s(%(levelname)s): %(message)s', level=logging.INFO)
    if args.print_accuracy:
        t_settings.print_accuracy = args.print_accuracy
    if args.show_figs:
        t_settings.show_figs = args.show_figs

    if args.log and args.print_accuracy and getattr(logging, args.log.upper()) > logging.INFO:
        raise ValueError('Accuracies are printed on INFO logging level. Set logging to info level or lower')


def settings_init():
    parser = argparse.ArgumentParser(description='Simulate quantum generative adversarial network')
    parser.add_argument('-q','--qubits', type=int, help=f'Amount of qubits to use for generator (2^this is used for discriminator) (Default {g_settings.num_qubits})')
    parser.add_argument('-r', '--repeats', type=int, help=f'Amount of repeats to use for training')
    parser.add_argument('-ds', '--synthetic', type=int, help=f'Amount of synthetic samples to process per discriminator minimize update (Default {data_settings.synthetic_size})')
    parser.add_argument('-dr', '--real', type=int, help=f'Amount of real samples to process per discriminator minimize update (Default {data_settings.real_size})')
    parser.add_argument('-dg', '--generator', type=int, help=f'Amount of real samples to process per generator minimize update (Default {data_settings.gen_size})')
    parser.add_argument('-id', '--iter_discriminator', type=int, help=f'Iterations to train discriminator in one network update (Default {d_settings.max_iter})')
    parser.add_argument('-ig', '--iter_generator', type=int, help=f'Iterations to train generator in one network update (Default {g_settings.max_iter})')
    parser.add_argument('-dis', '--discriminator_type', type=int, choices=[1,2], help=f'Type to use for discriminator, as explained by Casper. Note that type 1 uses 2^qubits qubits, while type 2 uses just qubits qubits (Default {d_settings.type})')
    parser.add_argument('-l', '--log', type=str, choices=['debug', 'info', 'warning', 'error', 'critical'], help=f'Logging severity (Default "info")')
    parser.add_argument('-pa', '--print_accuracy', action='store_true', help=f'Print intermediate accuracies for generator and discriminator (slows down training a bit) (Default {print_accuracy})')
    parser.add_argument('-sf', '--show_figs', action='store_true', help=f'Show figures (should be off if you are running on a server environment) (Default {t_settings.show_figs})')
    _init(parser.parse_args())

g_num_qubits = 3
g_depth = 2
g_n_shots = 6000
g_paramtype = InitParamType.RANDOM
g_trainingtype = TrainingType.COBYLA
g_max_iter = 50
g_step_rate = 0.1
global g_settings
g_settings = GeneratorSettings(g_num_qubits, g_depth, g_n_shots, g_paramtype, g_trainingtype, g_max_iter, g_step_rate)

# Discriminator specific settings
d_num_qubits = 2**g_num_qubits
d_depth = 5
d_n_shots = 0
d_paramtype = InitParamType.RANDOM
d_trainingtype = TrainingType.COBYLA
d_max_iter = 50
d_type = 2
d_step_rate = 0.1
global d_settings
d_settings = DiscriminatorSettings(d_num_qubits, d_depth, d_n_shots, d_paramtype, d_trainingtype, d_max_iter, d_type, d_step_rate)

# Data settings
mu = 1.0
sigma = 1.0
batch_size = 2000
synthetic_size = 10
real_size = 10
gen_size = 20
global data_settings
data_settings = DataSettings(mu, sigma, batch_size, synthetic_size, real_size, gen_size)


repeats = 10
print_accuracy = False
show_figs = False
global t_settings
t_settings = TrainingSettings(repeats, print_accuracy, show_figs)