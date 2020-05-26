import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.optimize import minimize
from functools import partial

from src.settings import d_settings as s
import cirq

####################################
# Data generation
####################################
# Make a dataset of points inside and outside of a circle
def circle(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
    Xvals, yvals = [], []

    for i in range(samples):
        x = 2 * (np.random.rand(2)) - 1
        y = 0
        if np.linalg.norm(x - center) < radius:
            y = 1
        Xvals.append(x)
        yvals.append(y)
    return np.array(Xvals), np.array(yvals)


def obtain_data(samples=200):
    X, Y = circle(samples)
    data = np.empty((X.shape[0],X.shape[1]+1))
    data[:,:2] = X
    data[:,-1] = X[:,0]*X[:,1]
    return data, Y


####################################
# Circuit generation
####################################
# Constructs a base circuit, with inputs set ready
def construct_base_simulator():
    qubits = [cirq.LineQubit(x) for x in range(s.num_qubits)]
    simulator = cirq.Simulator()
    # circuit.append(get_initial_states(qubits))
    return simulator, qubits


def data_preparation(qubits, x_i):
    # input x_i as angles of RY operations
    yield (cirq.rz(x_i[j]).on(qubits[j]) for j in range(len(x_i)))
    yield (cirq.ry(x_i[j]).on(qubits[j]) for j in range(len(x_i)))
    yield (cirq.rz(x_i[j]).on(qubits[j]) for j in range(len(x_i)))


def pqc(qubits, params):
    for l in range(s.num_layers):
        yield (cirq.rx(params[(l+1)*j]).on(qubits[j]) for j in range(len(qubits)))
        yield (cirq.CZ(qubits[j],qubits[j+1]) for j in range(len(qubits)-1))


def qml_classifier_circuit(qubits, params, x_i):
    dp = data_preparation(qubits, x_i), 
    p = pqc(qubits, params)
    return cirq.Circuit(dp, p, cirq.measure(*qubits, key='x')) if s.n_shots > 0 else cirq.Circuit(dp, p)


####################################
# Measuring
####################################
def run_without_measurements(simulator, circuit):
    results = simulator.simulate(circuit)
    return abs(results.final_state[-1])

def run_with_measurements(simulator, circuit):
    results = simulator.run(circuit, repetitions=s.n_shots)
    counter_measurements = results.histogram(key='x')
    return float(counter_measurements[7]) / s.n_shots if 7 in counter_measurements.keys() else 0

def run_circuit(simulator, qubits, params, datapoint):
    circuit = qml_classifier_circuit(qubits, params, datapoint)
    return run_with_measurements(simulator, circuit) if s.n_shots > 0 else run_without_measurements(simulator, circuit)


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

def run_discriminator():
    data, labels = obtain_data()
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
    final_params = minimize(cto, params_init, method="COBYLA", options={"maxiter":80})
    end_time = time()
    print(end_time-start_time)

    print(f'Accuracy post (training): {accuracy_score(labels_train, predict(simulator, qubits, final_params.x, data_train))}')
    print(f'Accuracy post (testing):  {accuracy_score(labels_test, predict(simulator, qubits, final_params.x, data_test))}')