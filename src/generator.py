from time import time
from functools import partial
import numpy as np
import cirq
from sympy import Symbol, Matrix
from enum import Enum
import matplotlib.pyplot as plt

from src.settings import g_settings as s
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


# Function that computes the kernel for the MMD loss
def multi_rbf_kernel(x, y, sigma_list=[0.25, 4]):
    # multi-RBF kernel. Args: 
    #     x,y -> 1-or-2darray: collection of samples A (x) or B (y)
    #     sigma_list (list): a list of bandwidths.
    # Returns: 2darray: kernel matrix.
    ndim = x.ndim
    assert ndim == 1 or ndim == 2
    exponent = np.abs(x[:, None] - y[None, :])**2 if ndim == 1 else ((x[:, None, :] - y[None, :, :])**2).sum(axis=2)
    return np.sum((np.exp(-(1.0 / (2*x))*exponent) for x in sigma_list))


# Function that computes expectation of kernel in MMD loss
def kernel_expectation(px, py, kernel_matrix):
    return px.dot(kernel_matrix).dot(py)


# Function that computes the squared MMD loss related to the given kernel_matrix.
def squared_MMD_loss(probs, target, kernel_matrix):
    dif_probs = probs - target
    return kernel_expectation(dif_probs, dif_probs, kernel_matrix)


# The loss function that we aim to minimize.
def loss(theta, circuit, target, kernel_matrix):
    probs = estimate_probs(circuit, theta)
    return squared_MMD_loss(probs, target, kernel_matrix)


# Cheat and get gradient
def gradient(theta, circuit, target, kernel_matrix):
    prob = estimate_probs(circuit, theta)
    grad = []
    for i in range(len(theta)):
        # pi/2 phase
        theta[i] += np.pi/2.
        prob_pos = estimate_probs(circuit, theta)
        # -pi/2 phase
        theta[i] -= np.pi
        prob_neg = estimate_probs(circuit, theta)
        # recover
        theta[i] += np.pi/2.
        grad_pos = kernel_expectation(prob, prob_pos, kernel_matrix) - kernel_expectation(prob, prob_neg, kernel_matrix)
        grad_neg = kernel_expectation(target, prob_pos, kernel_matrix) - kernel_expectation(target, prob_neg, kernel_matrix)
        grad.append(grad_pos - grad_neg)
    return np.array(grad)


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

def train(circuit, paramlist):
    step = [0]
    tracking_cost = []
    def callback(x, *args, **kwargs):
        step[0] += 1
        tracking_cost.append(loss_ansatz(x))
        print(f'step = {step[0]}, loss = {loss_ansatz(x):.5}')

    # MMD kernel
    kernel_matrix = multi_rbf_kernel(np.arange(2**s.num_qubits), np.arange(2**s.num_qubits))
    loss_ansatz = partial(loss, circuit=circuit, target=get_real_data(), kernel_matrix=kernel_matrix)


    if s.trainingtype == TrainingType.ADAM:
        from climin import Adam
        optimizer = Adam(wrt=paramlist, fprime=partial(gradient, circuit=circuit, target=get_real_data(), kernel_matrix=kernel_matrix),step_rate=s.step_rate)
        for info in optimizer:
            callback(paramlist)
            if step[0] == s.max_iter:
                break
        return loss_ansatz(paramlist), paramlist
    else:
        methodname = s.trainingtype.get_scipy_name()
        from scipy.optimize import minimize
        res = minimize(loss_ansatz,
                       paramlist, 
                       method=methodname, 
                       jac=partial(gradient, circuit=circuit, target=get_real_data(), kernel_matrix=kernel_matrix),
                       tol=10**-4, 
                       options={'maxiter':s.max_iter, 'disp': 0, 'gtol':1e-10, 'ftol':0}, 
                       callback=callback)
        return res.fun, res.x


####################################
# Main Control
####################################

# def run_generator(plot=False):
#     # Training the QCBM.
#     start_time = time()
#     loss, params_final = train(circuit, params_init)
#     end_time = time()
#     print(end_time-start_time)

#     if plot:
#         plt.plot(get_real_data())
#         plt.plot(estimate_probs(circuit, params_final))
#         plt.plot(estimate_probs(circuit, params_final)) #Only different if we use n_shots > 0
#         plt.legend(['Data', 'QCBM0','QCBM1'])
#         plt.show()
#     # return estimate_probs(circuit, params_final) for x in range(num_fakes)


class Generator(object):
    def __init__(self):
        self.params = get_init_params(s.paramtype, 2*s.depth*s.num_qubits)
        self.circuit, self.qubits = construct_base_circuit()
        ansatz = Ansatz()
        self.circuit.append(ansatz.get_circuit(self.qubits))


    # Train this thing to become better at providing synthetic data
    def train(self):
        loss, params_final = train(self.circuit, self.params)
        self.params = params_final
        return loss, params_final


    # Generate synthetic data, used to train discriminator
    def gen_synthetics(self, amount):
        if s.n_shots > 0:
            return (estimate_probs(self.circuit, self.params) for x in range(amount))
        else:
            raise NotImplementedError('All same samples would be generated')


    def print_circuit(self, transpose=True):
        print(self.circuit.to_text_diagram(transpose=transpose))