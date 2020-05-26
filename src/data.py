import numpy as np

from src.settings import data_settings as s

# Our true distribution, authentic/real data, however you want to call it
def gaussian_pdf():
    x = np.arange(2**s.num_qubits)
    pl = 1. / np.sqrt(2*np.pi*s.sigma**2) * np.exp(-(x-s.mu)**2 / (2.*s.sigma**2))
    return pl/pl.sum()

def get_real_data():
    return gaussian_pdf()

def in_real_data(datapoint):
    return datapoint in gaussian_pdf()