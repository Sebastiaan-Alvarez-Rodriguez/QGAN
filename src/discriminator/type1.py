import numpy as np 
from sklearn.metrics import mean_squared_error, accuracy_score
from functools import partial
from sympy import Symbol
import cirq
import logging

from src.optimize import optimize

from src.settings import d_settings as s
from src.enums.initparamtype import get_init_params
import src.util as util

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
def run_circuit(simulator, circuit, qubits, datapoint, params):
    if s.n_shots > 0:
        raise NotImplementedError('I cannot just prepare a state like that on hardware. Have to fix the quantum RAM problem for that')    
    else:
        # Added to fill in parameters
        param_mapping = [(f'var_{x}', param) for x, param in enumerate(params)]
        resolver = cirq.ParamResolver(dict(param_mapping))
        resolved_circuit = cirq.resolve_parameters(circuit, resolver)

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

        results = simulator.simulate(resolved_circuit, initial_state=init_state)
        return abs(results.final_state[-1])


####################################
# Optimize
####################################
def sweep_data(simulator, circuit, qubits, data, params):
    return [run_circuit(simulator, circuit, qubits, dist, params) for dist in data]


def cost_to_optimize(simulator, circuit, qubits, data, labels, params):
    return mean_squared_error(labels, sweep_data(simulator, circuit, qubits, data, params))


####################################
# Main Control
####################################
class Discriminator(object):
    def __init__(self):
        # self.params = np.random.uniform(-2*np.pi, 2*np.pi, size=num_params) # Old circle stuff
        self.params = get_init_params(s.paramtype, 2*s.depth*s.num_qubits)
        self.simulator, self.qubits = construct_base_simulator()
        self.ansatz = Ansatz()
        self.circuit = cirq.Circuit(self.ansatz.get_circuit(self.qubits))


    # Train discriminator to better learn difference between real and fake data
    def train(self, data, labels):
        cto = partial(cost_to_optimize, self.simulator, self.circuit, self.qubits, data, labels)
        logging.debug(f'params start: {self.params}')
        loss, params_final = optimize(cto, self.params, s.trainingtype, s.max_iter, s.step_rate)
        logging.debug(f'params end: {params_final}')
        self.params = params_final
        return loss, params_final

    # Returns mean squared error on given dataset (lower=better)
    def test(self, data, labels, params=None):
        return mean_squared_error(labels, self.get_chances(data, params))


    # Returns mean squared error on generated dataset, and detailed statistics
    def test2(self, data, labels, params=None):
        observed = self.get_chances(data, params)
        return mean_squared_error(labels, observed), util.stat(labels, np.array([1 if p > .5 else 0 for p in observed]))


    # Given a dataset, returns the predicted labels
    def predict(self, data, params=None):
        return np.array([1 if p > .5 else 0 for p in self.get_chances(data, params)])


    # Given a dataset, returns probabilities of measuring all 1's
    def get_chances(self, data, params=None):
        if params is None:
            params = self.params
        return sweep_data(self.simulator, self.circuit, self.qubits, data, params)


    def __str__(self):
        return self.circuit.to_text_diagram(transpose=True)

    def __repr__(self):
        return str(self)