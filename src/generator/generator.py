from time import time
from functools import partial
import numpy as np
import cirq
from sympy import Symbol, Matrix
from enum import Enum
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from src.settings import g_settings as s
from src.settings import data_settings as ds

from src.data import get_real_data
from src.enums.initparamtype import InitParamType, get_init_params
from src.enums.trainingtype import TrainingType


####################################
# Loss function
####################################

# Estimate all probabilities of the PQCs distribution.
def estimate_probs(circuit, params):
    # Creating parameter resolve dict by adding state and theta.
    param_mapping = [(f'var_{x}', param) for x, param in enumerate(params)]
    resolver = cirq.ParamResolver(dict(param_mapping))
    resolved_circuit = cirq.resolve_parameters(circuit, resolver)

    if s.n_shots == 0: # Use statevector simulator
        final_state = cirq.final_wavefunction(resolved_circuit)
        return np.array([np.abs(final_state[x])**2 for x in range(len(final_state))])
    else: # Run the circuit
        # Adding measurement at the end.
        resolved_circuit.append(cirq.measure(*resolved_circuit.all_qubits(), key='m'))
        results = cirq.sample(resolved_circuit, repetitions=s.n_shots)
        frequencies = results.histogram(key='m')
        probs = np.zeros(2**s.num_qubits)
        for key, value in frequencies.items():
            probs[key] = value / s.n_shots
    return probs


def cost_to_optimize(generator, discriminator, params):
    # params -> generate fake data (hier)
    data = list(generator.gen_synthetics(ds.train_synthetic_size, params=params))
    labels = discriminator.predict(data) 
    return mean_squared_error(labels, [1 for x in range(len(labels))])


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
# Training method Control
####################################

def train(circuit, paramlist, generator, discriminator):
    step = [0]
    tracking_cost = []
    def callback(x, *args, **kwargs):
        step[0] += 1
        # tracking_cost.append(loss_ansatz(x))
        # print(f'step = {step[0]}, loss = {loss_ansatz(x):.5}')
        print(f'step = {step[0]}')

    cto = partial(cost_to_optimize, generator, discriminator)
    if s.trainingtype == TrainingType.ADAM:
        from climin import Adam
        optimizer = Adam(wrt=paramlist, 
                        # fprime=partial(gradient, circuit=circuit, target=get_real_data(), kernel_matrix=kernel_matrix),
                        fprime=cto,
                        step_rate=s.step_rate)
        for info in optimizer:
            callback(paramlist)
            if step[0] == s.max_iter:
                break
        # return loss_ansatz(paramlist), paramlist
        return None, paramlist

    else:
        methodname = s.trainingtype.get_scipy_name()
        
        res = minimize(cto,
                       paramlist, 
                       method=methodname, 
                       # jac=partial(gradient, circuit=circuit, target=get_real_data(), kernel_matrix=kernel_matrix),
                       tol=10**-4, 
                       options={'maxiter':s.max_iter, 'disp': 0, 'gtol':1e-10, 'ftol':0}, 
                       callback=callback)
        return res.fun, res.x


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
        loss, params_final = train(self.circuit, self.params, self, discriminator)
        self.params = params_final
        return loss, params_final


    # Generate synthetic data, used to train discriminator
    def gen_synthetics(self, amount, params=None):
        if s.n_shots > 0:
            if not params is None:
                return (estimate_probs(self.circuit, params) for x in range(amount))
            else:
                return (estimate_probs(self.circuit, self.params) for x in range(amount))
        else:
            raise NotImplementedError('All same samples would be generated')


    def print_circuit(self, transpose=True):
        print(self.circuit.to_text_diagram(transpose=transpose))