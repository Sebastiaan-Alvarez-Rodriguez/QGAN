from src.enums.initparamtype import InitParamType
from src.enums.trainingtype import TrainingType

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
    def __init__(self, num_qubits, num_layers, n_shots):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.n_shots = n_shots
        
class DataSettings(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma


# Generator specific settings
g_num_qubits = 3
depth = 2
g_n_shots = 6000
paramtype = InitParamType.RANDOM
trainingtype = TrainingType.ADAM
max_iter = 50
step_rate = 0.1

global g_settings
g_settings = GeneratorSettings(g_num_qubits, depth, g_n_shots, paramtype, trainingtype, max_iter, step_rate)

# Discriminator specific settings
d_num_qubits = 3
num_layers = 5
d_n_shots = 0

global d_settings
d_settings = DiscriminatorSettings(d_num_qubits, num_layers, d_n_shots)

# Data settings
mu = 2**(g_num_qubits-1)-0.5
sigma = 2**(g_num_qubits-2)

global data_settings
data_settings = DataSettings(mu, sigma)
