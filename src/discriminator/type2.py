import numpy as np 
from sklearn.metrics import mean_squared_error
from functools import partial
from time import time
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
def construct_base_qubits():
    qubits = [cirq.LineQubit(x) for x in range(s.num_qubits)]
    return qubits


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

    # Generate gates required to embed datapoint into circuit
    @staticmethod
    def data_preparation(qubits, datapoint):
        # Old way: 8 qubits, each of 8 points sets x,y,z rotation on 1 line
        yield (cirq.rx(val).on(qubits[idx]) for idx, val in enumerate(datapoint))
        yield (cirq.ry(val).on(qubits[idx]) for idx, val in enumerate(datapoint))
        yield (cirq.rz(val).on(qubits[idx]) for idx, val in enumerate(datapoint))

    def get_circuit(self, qubits, datapoint):
        for d in range(s.depth):
            # Add datapoint
            yield self.data_preparation(qubits, datapoint)
            # Adding single qubit rotations
            yield self.rot_z_layer(self.params[d*2*s.num_qubits : (d+1)*2*s.num_qubits : 2])
            yield self.rot_y_layer(self.params[d*2*s.num_qubits+1 : (d+1)*2*s.num_qubits+1 : 2])
            # Adding entangling layer
            yield self.entangling_layer()


####################################
# Measuring
####################################
# Note, we measure chance of all 1's on all qubits below. Simulate if n_shots>0, compute final state otherwise
def run_circuit(ansatz, qubits, datapoint, params):
    circuit = cirq.Circuit(ansatz.get_circuit(qubits, datapoint))
    # Added to fill in parameters
    param_mapping = [(f'var_{x}', param) for x, param in enumerate(params)]
    resolver = cirq.ParamResolver(dict(param_mapping))
    resolved_circuit = cirq.resolve_parameters(circuit, resolver)

    if s.n_shots == 0: # Use statevector simulator
        final_state = cirq.final_wavefunction(resolved_circuit)
        return abs(final_state[-1])
    else: # Run the circuit
        # Adding measurement at the end.
        resolved_circuit.append(cirq.measure(*qubits, key='x'))
        results = cirq.sample(resolved_circuit, repetitions=s.n_shots)
        frequencies = results.histogram(key='x')
        return float(frequencies[s.num_qubits-1]) / s.n_shots if s.num_qubits-1 in frequencies.keys() else 0


####################################
# Optimize
####################################
def sweep_data(ansatz, qubits, data, params):
    return [run_circuit(ansatz, qubits, dist, params) for dist in data]


def cost_to_optimize(ansatz, qubits, data, labels, params):
    return mean_squared_error(labels, sweep_data(ansatz, qubits, data, params))


####################################
# Main Control
####################################
class Discriminator(object):
    def __init__(self):
        # self.params = np.random.uniform(-2*np.pi, 2*np.pi, size=num_params) # Old circle stuff
        self.params = get_init_params(s.paramtype, 2*s.depth*s.num_qubits)
        self.qubits = construct_base_qubits()
        self.ansatz = Ansatz()


    # Train discriminator to better learn difference between real and fake data
    def train(self, data, labels):
        cto = partial(cost_to_optimize, self.ansatz, self.qubits, data, labels)
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
        return sweep_data(self.ansatz, self.qubits, data, params)
