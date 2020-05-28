import numpy as np
from scipy.stats import lognorm

from src.settings import data_settings as s
from src.settings import g_settings as gs

# Our true distribution, authentic/real data, however you want to call it
def gaussian_pdf_exact():
    x = np.arange(ds.num_qubits)
    pl = 1. / np.sqrt(2*np.pi*s.sigma**2) * np.exp(-(x-s.mu)**2 / (2.*s.sigma**2))
    return pl/pl.sum()


def get_real_data():
    return gaussian_pdf_exact()


def normal_pdf(amount):
    datapoints = 2**gs.num_qubits
    data = np.empty(0)
    while len(data) != amount:
        data = np.append(data, lognorm.rvs(s=s.mu, scale=np.exp(s.sigma), size=amount-len(data)))
        data = np.array([round(i) for i in data[(data<=datapoints-.5) & (-.5<=data)]])
    return data

def get_dist(data):
    return np.array([len(data[data==x])/len(data) for x in range(2**gs.num_qubits)])


def get_real_samples(amount):
    return (get_dist(normal_pdf(s.batch_size)) for x in range(amount))


def in_real_data(datapoint):
    return datapoint in gaussian_pdf()