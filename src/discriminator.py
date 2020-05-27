import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.optimize import minimize
from functools import partial

from src.settings import d_settings as s
import cirq


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
# Simulator generation
####################################

# Constructs a base simulator and some qubits
def construct_base_simulator():
    qubits = [cirq.LineQubit(x) for x in range(s.num_qubits)]
    simulator = cirq.Simulator()
    return simulator, qubits


# def pqc(qubits, params):
#     for l in range(s.num_layers):
#         yield (cirq.rx(params[(l+1)*j]).on(qubits[j]) for j in range(len(qubits)))
#         yield (cirq.CZ(qubits[j],qubits[j+1]) for j in range(len(qubits)-1))


# def qml_classifier_circuit(qubits, params, x_i):
#     dp = data_preparation(qubits, x_i), 
#     p = pqc(qubits, params)
#     return cirq.Circuit(dp, p, cirq.measure(*qubits, key='x')) if s.n_shots > 0 else cirq.Circuit(dp, p)


####################################
# Measuring
####################################
# Note, we measure chance of all 1's on all qubits below if s.n_shots > 0
def run_circuit(simulator, qubits, params, datapoint):
    if s.n_shots > 0:
        circuit = cirq.Circuit(qml_classifier_circuit(qubits, params, datapoint), cirq.measure(*qubits, key='x'))
        results = simulator.run(circuit, repetitions=s.n_shots)
        counter_measurements = results.histogram(key='x')
        return float(counter_measurements[2**s.num_qubits-1]) / s.n_shots if 2**s.num_qubits-1 in counter_measurements.keys() else 0
    else:
        circuit = cirq.Circuit(qml_classifier_circuit(qubits, params, datapoint))
        results = simulator.simulate(circuit)
        return abs(results.final_state[-1])


####################################
# Optimize
####################################
def sweep_data(simulator, qubits, params, X):
    probas = [run_circuit(simulator, qubits, params, X[i,:]) for i in range(X.shape[0])]
    return probas

# will save by iterations the current cost, and accuracies
tracking_cost = []
def cost_to_optimize(simulator, qubits, data, labels, params):
    cost = mean_squared_error(labels, sweep_data(simulator, qubits, params, data))
    tracking_cost.append(cost)
    return cost


####################################
# Measurements
####################################
def predict(simulator, qubits, params, data):
    return np.array([1 if p > .5 else 0 for p in sweep_data(simulator, qubits, params, data)])

####################################
# Main Control
####################################

def run_discriminator(data, labels):
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.5, random_state=42, stratify=labels)

    simulator, qubits = construct_base_simulator()
    num_params = s.num_qubits * s.num_layers
    # Set a random seed
    np.random.seed(42)
    params_init = np.random.uniform(-2*np.pi, 2*np.pi, size=num_params)
    run_circuit(simulator, qubits, params_init, data[0])

    cto = partial(cost_to_optimize, simulator, qubits, data_train, labels_train)

    print(f'Accuracy pre (training): {accuracy_score(labels_train, predict(simulator, qubits, params_init, data_train))}')
    print(f'Accuracy pre (testing):  {accuracy_score(labels_test, predict(simulator, qubits, params_init, data_test))}')

    from time import time
    start_time = time()
    final_params = minimize(cto, params_init, method="COBYLA", options={"maxiter":s.max_iter})
    end_time = time()
    print(end_time-start_time)

    print(f'Accuracy post (training): {accuracy_score(labels_train, predict(simulator, qubits, final_params.x, data_train))}')
    print(f'Accuracy post (testing):  {accuracy_score(labels_test, predict(simulator, qubits, final_params.x, data_test))}')