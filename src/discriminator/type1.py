import numpy as np 
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.optimize import minimize
from functools import partial
from time import time
from sympy import Symbol
import cirq

from src.settings import d_settings as s
from src.enums.initparamtype import get_init_params


####################################
# Simulator generation
####################################
# Constructs a base simulator and required amount of qubits
def construct_base_simulator():
    qubits = [cirq.LineQubit(x) for x in range(s.num_qubits)]
    simulator = cirq.Simulator()
    return simulator, qubits


####################################
# Ansatz
####################################
class Ansatz(object):
    '''Class containing functions to generate circuit and parameters for ansatz'''
    def __init__(self):
        self.params = list(self.gen_params())

    def gen_params(self):
        return (Symbol(f'var_{x}') for x in range(2*s.depth*s.num_qubits))

    # Return lazy layer of single qubit z rotations
    def rot_z_layer(self, parameters):
        return (cirq.rz(2*parameters[x])(cirq.LineQubit(x)) for x in range(s.num_qubits))

    # Return lazy layer of single qubit y rotations
    def rot_y_layer(self, parameters):
        return (cirq.ry(parameters[x])(cirq.LineQubit(x)) for x in range(s.num_qubits))

    # Return (1 item or) lazy Layer of entangling CZ(i,i+1 % num_qubits) gates
    def entangling_layer(self):
        if s.num_qubits == 1:
            raise ValueError('A controlled-Z rotation needs at least 2 qubits')
        return cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(1)) if s.num_qubits == 2 else (cirq.CZ(cirq.LineQubit(x), cirq.LineQubit((x+1)%s.num_qubits)) for x in range(s.num_qubits))

    
    def get_circuit(self, qubits):
        for d in range(s.depth):
            # Adding single qubit rotations
            yield self.rot_z_layer(self.params[d*2*s.num_qubits : (d+1)*2*s.num_qubits : 2])
            yield self.rot_y_layer(self.params[d*2*s.num_qubits+1 : (d+1)*2*s.num_qubits+1 : 2])
            # Adding entangling layer
            yield self.entangling_layer()


####################################
# Measuring
####################################
# Note, we measure chance of all 1's on all qubits below. Simulate if n_shots>0, compute final state otherwise
def run_circuit(simulator, circuit, qubits, params, datapoint):
    if s.n_shots > 0:
        raise NotImplementedError('I cannot just prepare a state like that on hardware. Gotta fix the quantum RAM problem for that')    
    else:
        # Added to fill in parameters
        param_mapping = [(f'var_{x}', param) for x, param in enumerate(params)]
        resolver = cirq.ParamResolver(dict(param_mapping))
        resolved_circuit = cirq.resolve_parameters(circuit, resolver)
        # End of added code

        # Added trick to convert datapoint to initial state
        '''
        a_x = sqrt(point(x)) (point -> 1 van entries )
        a0|0> + a1|1> + a2|2> + ... =
        [
            a0,
            a1,
            ...
            a7
        ]
        '''
        init_state = np.array([np.sqrt(x) for x in datapoint])
        # End of trick

        results = simulator.simulate(resolved_circuit, initial_state=init_state)
        return abs(results.final_state[-1])


####################################
# Optimize
####################################
def sweep_data(simulator, circuit, qubits, params, data):
    return [run_circuit(simulator, circuit, qubits, params, dist) for dist in data]


def cost_to_optimize(simulator, circuit, qubits, data, labels, params):
    return mean_squared_error(labels, sweep_data(simulator, circuit, qubits, params, data))


####################################
# Measurements
####################################
def predict(simulator, circuit, qubits, params, data):
    return np.array([1 if p > .5 else 0 for p in sweep_data(simulator, circuit, qubits, params, data)])


####################################
# Train
####################################
def train(simulator, circuit, qubits, paramlist, data, labels, printing):
    cto = partial(cost_to_optimize, simulator, circuit, qubits, data, labels)
    if printing:
        print(f'Discriminator accuracy pre:  {accuracy_score(labels, predict(simulator, circuit, qubits, paramlist, data))}')

    start_time = time()
    res = minimize(cto, paramlist, method='COBYLA', options={'maxiter':s.max_iter})
    end_time = time()
    if printing:
        print(end_time-start_time)
        print(f'Discriminator accuracy post: {accuracy_score(labels, predict(simulator, circuit, qubits, res.x, data))}')
    return res.fun, res.x


####################################
# Main Control
####################################
class Discriminator1(object):
    def __init__(self):
        # self.params = np.random.uniform(-2*np.pi, 2*np.pi, size=num_params) # Old circle stuff
        self.params = get_init_params(s.paramtype, 2*s.depth*s.num_qubits)
        self.simulator, self.qubits = construct_base_simulator()
        self.ansatz = Ansatz()
        self.circuit = cirq.Circuit(self.ansatz.get_circuit(self.qubits))


    # Train discriminator to better learn difference between real and fake data
    def train(self, data, labels, printing=False):
        # Potential problem: Maybe there is no res.fun as loss
        loss, params_final = train(self.simulator, self.circuit, self.qubits, self.params, data, labels, printing)
        self.params = params_final
        return loss, params_final


    def predict(self, data):
        return predict(self.simulator, self.circuit, self.qubits, self.params, data)

    # Run discriminator for val. Val can be 1 datapoint or a list of points (then a generator is returned lazily)
    def run(self, val):
        if isinstance(val, list):
            return (run_circuit(self.simulator, self.circuit, self.qubits, self.params, datapoint) for datapoint in val)
        else:
            return run_circuit(self.simulator, self.circuit, self.qubits, self.params, val)