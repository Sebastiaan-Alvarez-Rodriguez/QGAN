from time import time
from functools import partial
import numpy as np
import cirq
from sympy import Symbol, Matrix
from enum import Enum
import matplotlib.pyplot as plt

####################################
# Sampling trainingdata
####################################

def gaussian_pdf(num_bit, mu, sigma):
    x = np.arange(2**num_bit)
    pl = 1. / np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2 / (2.*sigma**2))
    return pl/pl.sum()

####################################
# Loss function
####################################

# Estimate all probabilities of the PQCs distribution.
def estimate_probs(circuit, num_qubits, params, n_shots):
    # Creating parameter resolve dict by adding state and theta.
    param_mapping = [(f'var_{x}', param) for x, param in enumerate(params)]
    resolver = cirq.ParamResolver(dict(param_mapping))
    resolved_circuit = cirq.resolve_parameters(circuit, resolver)

    if n_shots == 0: # Use statevector simulator
        final_state = cirq.final_wavefunction(resolved_circuit)
        return np.array([np.abs(final_state[x])**2 for x in range(len(final_state))])
    else: # Run the circuit
        # Adding measurement at the end.
        resolved_circuit.append(cirq.measure(*resolved_circuit.all_qubits(), key='m'))
        results = cirq.sample(resolved_circuit, repetitions=n_shots)
        frequencies = results.histogram(key='m')
        probs = np.zeros(2**num_qubits)
        for key, value in frequencies.items():
            probs[key] = value / n_shots
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
def loss(theta, circuit, num_qubits, target, kernel_matrix, n_shots):
    probs = estimate_probs(circuit, num_qubits, theta, n_shots)
    return squared_MMD_loss(probs, target, kernel_matrix)


# Cheat and get gradient
def gradient(theta, circuit, num_qubits, target, kernel_matrix, n_shots):
    prob = estimate_probs(circuit, num_qubits, theta, n_shots)
    grad = []
    for i in range(len(theta)):
        # pi/2 phase
        theta[i] += np.pi/2.
        prob_pos = estimate_probs(circuit, num_qubits, theta, n_shots)
        # -pi/2 phase
        theta[i] -= np.pi
        prob_neg = estimate_probs(circuit, num_qubits, theta, n_shots)
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
    return (cirq.H(ancilla), *(cirq.X(q) for q in qubits[:len(qubits)//2]))

# Constructs a base circuit, with inputs set ready
def construct_base_circuit(num_qubits):
    qubits = [cirq.LineQubit(x) for x in range(num_qubits)]
    circuit = cirq.Circuit()
    # circuit.append(get_initial_states(qubits))
    return circuit, qubits


####################################
# Ansatz
####################################
class Ansatz(object):
    '''Class containing functions to generate circuit and parameters for ansatz'''
    def __init__(self):
        self.params = None

    def gen_params(self, num_qubits, depth):
        return (Symbol(f'var_{x}') for x in range(2*depth*num_qubits))

    # Return lazy layer of single qubit z rotations
    def rot_z_layer(self, num_qubits, parameters):
        return (cirq.rz(2*parameters[x])(cirq.LineQubit(x)) for x in range(num_qubits))

    # Return lazy layer of single qubit y rotations
    def rot_y_layer(self, num_qubits, parameters):
        return (cirq.ry(parameters[x])(cirq.LineQubit(x)) for x in range(num_qubits))

    # Return (1 item or) lazy Layer of entangling CZ(i,i+1 % num_qubits) gates
    def entangling_layer(self, num_qubits):
        if num_qubits == 1:
            raise ValueError('A controlled-Z rotation needs at least 2 qubits')
        return cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(1)) if num_qubits == 2 else (cirq.CZ(cirq.LineQubit(x), cirq.LineQubit((x+1)%num_qubits)) for x in range(num_qubits))

    def get_circuit(self, qubits, depth=2):
        num_qubits = len(qubits)
        self.params = list(self.gen_params(num_qubits, depth))
        for d in range(depth):
            # Adding single qubit rotations
            yield self.rot_z_layer(num_qubits, self.params[d*2*num_qubits : (d+1)*2*num_qubits : 2])
            yield self.rot_y_layer(num_qubits, self.params[d*2*num_qubits+1 : (d+1)*2*num_qubits+1 : 2])
            # Adding entangling layer
            yield self.entangling_layer(num_qubits)


####################################
# Initial Parameter Control
####################################

class InitParamType(Enum):
    NORMAL = 1,
    RANDOM = 2,
    UNIFORM = 3


def get_init_params(paramtype: InitParamType, num_params):
    if paramtype == InitParamType.NORMAL:
        pass
    elif paramtype == InitParamType.RANDOM:
        return np.random.random(num_params)*2*np.pi
    else:
        pass


####################################
# Training method Control
####################################

class TrainingType(Enum):
    ADAM = 1,
    BFGS = 2,


    def get_scipy_name(self):
        if self == TrainingType.ADAM:
            raise ValueError('Adam is not in scipy')
        elif self == TrainingType.BFGS:
            return 'L-BFGS-B'


def train(circuit, num_qubits, n_shots, paramlist, method: TrainingType, max_iter, step_rate):
    step = [0]
    tracking_cost = []
    def callback(x, *args, **kwargs):
        step[0] += 1
        tracking_cost.append(loss_ansatz(x))
        print(f'step = {step[0]}, loss = {loss_ansatz(x):.5}')

    # MMD kernel
    kernel_matrix = multi_rbf_kernel(np.arange(2**num_qubits), np.arange(2**num_qubits))
    pg = gaussian_pdf(num_qubits, mu=2**(num_qubits-1)-0.5, sigma=2**(num_qubits-2))
    loss_ansatz = partial(loss, circuit=circuit, num_qubits=num_qubits, target=pg, kernel_matrix=kernel_matrix, n_shots=n_shots)


    if method == TrainingType.ADAM:
        from climin import Adam
        optimizer = Adam(wrt=paramlist, fprime=partial(gradient, circuit=circuit, num_qubits=num_qubits, target=pg, kernel_matrix=kernel_matrix, n_shots=n_shots),step_rate=step_rate)
        for info in optimizer:
            callback(paramlist)
            if step[0] == max_iter:
                break
        return loss_ansatz(paramlist), paramlist
    else:
        methodname = method.get_scipy_name()
        from scipy.optimize import minimize
        res = minimize(loss_ansatz,
                       paramlist, 
                       method=methodname, 
                       jac=partial(gradient, circuit=circuit, num_qubits=num_qubits, target=pg, kernel_matrix=kernel_matrix, n_shots=n_shots),
                       tol=10**-4, 
                       options={'maxiter':max_iter, 'disp': 0, 'gtol':1e-10, 'ftol':0}, 
                       callback=callback)
        return res.fun, res.x

####################################
# Main Control
####################################

def run(num_qubits, depth, n_shots, initparamtype: InitParamType, trainingtype: TrainingType, max_iter=50, step_rate=0.1, plot=False):
    circuit, qubits = construct_base_circuit(num_qubits)
    ansatz = Ansatz()
    circuit.append(ansatz.get_circuit(qubits))
    params = ansatz.params
    # print(circuit.to_text_diagram(transpose=True))
    

    params_init = get_init_params(initparamtype, 2*depth*num_qubits)
    
    # Training the QCBM.
    start_time = time()
    loss, params_final = train(circuit, num_qubits, n_shots, params_init, trainingtype, max_iter, step_rate)
    end_time = time()
    print(end_time-start_time)

    if plot:
        plt.plot(gaussian_pdf(num_qubits, mu=2**(num_qubits-1)-0.5, sigma=2**(num_qubits-2)))
        plt.plot(estimate_probs(circuit, num_qubits, params_final, n_shots))
        plt.legend(['Data', 'Quantum Circuit Born Machine'])
        plt.show()

def main():
    num_qubits = 3
    depth = 3
    n_shots = 0
    run(num_qubits, depth, n_shots, InitParamType.RANDOM, TrainingType.ADAM, max_iter=50, plot=True)


if __name__ == '__main__':
    main()