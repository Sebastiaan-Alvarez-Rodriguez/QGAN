from time import time
from functools import partial
import numpy as np
import cirq
from sympy import Symbol, Matrix
from enum import Enum
from sklearn.metrics import mean_squared_error
import logging

from src.optimize import optimize
from src.settings import g_settings as s
from src.settings import data_settings as ds

from src.data import get_real_data
from src.enums.initparamtype import InitParamType, get_init_params


####################################
# Optimize
####################################

# Estimate probabilities of distribution
def estimate_probs(circuit, params):
    # Fill in vars in circuit with provided params
    if s.n_shots == 0: # Use statevector simulator
        final_state = cirq.final_wavefunction(circuit)
        return np.array([np.abs(final_state[x])**2 for x in range(len(final_state))])
    else: # Run the circuit
        results = cirq.sample(circuit, repetitions=s.n_shots)
        frequencies = results.histogram(key='m')
        probs = np.zeros(2**s.num_qubits)
        for key, value in frequencies.items():
            probs[key] = value / s.n_shots
    return probs


def cost_to_optimize(generator, discriminator, params):
    data = list(generator.gen_synthetics(ds.gen_size, params=params))
    probs = discriminator.get_chances(data)
    return mean_squared_error([1.0 for x in range(len(probs))], probs)


####################################
# Circuit generation
####################################

# Setups initial states
def get_initial_states(qubits):
    if s.paramtype == InitParamType.RANDOM:
        for qubit in qubits: # according to paper, need H on all qubits
            yield cirq.H(qubit)
        else:
            pass

# Constructs a base circuit, with inputs set ready
def construct_base_circuit():
    qubits = [cirq.LineQubit(x) for x in range(s.num_qubits)]
    circuit = cirq.Circuit()
    circuit.append(get_initial_states(qubits))
    return circuit, qubits


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
# Main Control
####################################

class Generator(object):
    def __init__(self):
        self.params = get_init_params(s.paramtype, 2*s.depth*s.num_qubits)
        self.circuit, self.qubits = construct_base_circuit()
        ansatz = Ansatz()
        self.circuit.append(ansatz.get_circuit(self.qubits))


    # Train this thing to become better at providing synthetic data
    def train(self, discriminator):
        logging.debug(f'params start: {self.params}')
        cto = partial(cost_to_optimize, self, discriminator)
        loss, params_final = optimize(cto, self.params, s.trainingtype, s.max_iter, s.step_rate)
        logging.debug(f'params end: {params_final}')

        self.params = params_final
        return loss, params_final


    # Returns mean squared error on locally generated dataset (lower=better)
    def test(self, discriminator, params=None):
        if params is None:
            params = self.params
        return cost_to_optimize(self, discriminator, params)


    # Generate synthetic data, used to train discriminator
    def gen_synthetics(self, amount, params=None):
        if params is None:
            params = self.params
        param_mapping = [(f'var_{x}', param) for x, param in enumerate(params)]
        resolved_circuit = cirq.resolve_parameters(self.circuit, cirq.ParamResolver(dict(param_mapping)))
        if s.n_shots > 0:
            resolved_circuit.append(cirq.measure(*resolved_circuit.all_qubits(), key='m'))
            return (estimate_probs(resolved_circuit, params) for x in range(amount))
        else:
            val = estimate_probs(resolved_circuit, params)
            return (val for x in range(amount)) # All same samples are generated


    def print_circuit(self, transpose=True):
        print(self.circuit.to_text_diagram(transpose=transpose))

    def __str__(self):
        return self.circuit.to_text_diagram(transpose=True)

    def __repr__(self):
        return str(self)