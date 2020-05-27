import numpy as np
from scipy.stats import lognorm

from src.settings import data_settings as s

# Our true distribution, authentic/real data, however you want to call it
def gaussian_pdf_exact():
    x = np.arange(2**s.num_qubits)
    pl = 1. / np.sqrt(2*np.pi*s.sigma**2) * np.exp(-(x-s.mu)**2 / (2.*s.sigma**2))
    return pl/pl.sum()

def normal_pdf(amount):
    datapoints = 2**s.num_qubits
    data = np.empty(0)
    while True:
        data = np.append(data, lognorm.rvs(s=s.mu, scale=exp(s.sigma), size=amount-len(data)))
        data = data[data<=datapoints-.5]
        data = data[data>=-.5]
        data = np.array([round(i) for i in data])
        if len(data) == amount: 
            return data

def get_real_data():
    return gaussian_pdf_exact()

def get_real_samples(amount):
    return normal_pdf(amount)


def in_real_data(datapoint):
    return datapoint in gaussian_pdf()