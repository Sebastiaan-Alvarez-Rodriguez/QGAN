from itertools import chain
from time import time
import logging
import sys
import matplotlib.pyplot as plt
import numpy as np

from src.enums.trainingtype import TrainingType
from src.discriminator.type1 import Discriminator as Discriminator1
from src.discriminator.type2 import Discriminator as Discriminator2
from src.generator.generator import Generator

from src.data import get_real_samples

from src.settings import g_settings as gs
from src.settings import d_settings as ds
from src.settings import t_settings as ts
from src.settings import data_settings as das

from src.settings import settings_init

'''
The plan:
1 train the discriminator to do what I propose above 
  (starting with the easiest thing to train, i.e., feeding quantum state
  with amplitudes given by square root of probs of Gaussian or quantum state
  produced by QCBM and training discriminator to tell the difference).
2 train the QCBM to fool the generator
 (i.e., minimize the loss function: discriminator says fake,
  i.e., QCBM, data is fake).
3 repeat and alternate the above two steps.
4 if it works, try harder to train encodings of the real/fake data.
'''

class QGAN(object):
    def __init__(self):
        self.d = Discriminator1() if ds.type == 1 else Discriminator2()
        self.g = Generator()

    # Returns dataset of [[point0, point1..., point7], ...], and list of 0 (fake) and 1 (real)
    def generate_dataset(self):
        f, r = self.g.gen_synthetics(das.synthetic_size), get_real_samples(das.real_size)
        labels = list(chain((0 for x in range(das.synthetic_size)), (1 for x in range(das.real_size))))
        return list(chain(f, r)), labels
            

    def _train_discriminator(self):
        dataset, labels = self.generate_dataset()
        logging.info(f'New trainingset generated')

        if ts.print_accuracy:
            logging.info(f'Discriminator mean squared error pre: {self.d.test(dataset, labels)}')
        d_start_time = time()
        self.d.train(dataset, labels)
        d_end_time = time()
        logging.info(f'Discriminator training completed in {round(d_end_time-d_start_time, 2)} seconds')
        if ts.print_accuracy:
            logging.info(f'Discriminator mean squared error post: {self.d.test(dataset, labels)}')


    def _train_generator(self):
        if ts.print_accuracy:
            logging.info(f'Generator mean squared error pre: {self.g.test(self.d)}')
        g_start_time = time()
        self.g.train(self.d)
        g_end_time = time()
        logging.info(f'Generator training completed in {round(g_end_time-g_start_time, 2)} seconds')
        if ts.print_accuracy:
            logging.info(f'Generator mean squared error post: {self.g.test(self.d)}')


    def train(self):
        total_start_time = time()
        for idx, x in enumerate(range(ts.repeats)):
            logging.info(f'Starting training iteration {idx}')
            it_start_time = time()

            self._train_discriminator()
            self._train_generator()

            it_end_time = time()
            logging.info(f'COMPLETED in {round(it_end_time-it_start_time, 2)} seconds')
           
        total_end_time = time()
        logging.info(f'FINISHED in {round(total_end_time-total_start_time, 2)} seconds')


    def _test_generator(self):
        print(f'Final generator squared error (can be high if discriminator is trained well): {self.g.test(self.d)}')

        logging.getLogger('matplotlib').setLevel(logging.WARNING) #shut up matplotlib
        plt.plot(next(get_real_samples(1)))
        for dist in self.g.gen_synthetics(4):
            plt.plot(dist)
        legend = ['Data']
        legend.extend(list(f'gen{x}' for x in range(4)))
        plt.legend(legend)
        if ts.show_figs:
            plt.show()
        plt.savefig('gen.pdf')


    def _test_discriminator(self):
        dataset, labels = self.generate_dataset()
        accuracy, details = self.d.test2(dataset, labels)
        logging.info(f'Final discriminator mean squared error (on test): {accuracy}')
        TP, FP, TN, FN = details
        logging.info(f'TP: {TP}')
        logging.info(f'FP: {FP}')
        logging.info(f'TN: {TN}')
        logging.info(f'FN: {FN}')
        logging.info(f'Total: {das.synthetic_size+das.real_size}')


    def test(self):
        self._test_generator()
        self._test_discriminator()


def parameter_prelude():
    logging.info(f'''
Training network for {ts.repeats} repeats, using
Generator:
    initial param type {gs.paramtype.name}
    training type {gs.trainingtype.name}
    using {gs.num_qubits} qubits
    depth {gs.depth} ({2*gs.depth*gs.num_qubits} params to optimize)
    shots {gs.n_shots} (used to estimate probabilities on hardware if > 0)
    maximal {gs.max_iter} iterations per repeat
    learning step rate {gs.step_rate} (used only if training type is ADAM)
Discriminator:
    initial param type {ds.paramtype.name}
    training type {ds.trainingtype.name}
    type {ds.type} (using {ds.num_qubits} qubits)
    depth {ds.depth} ({2*ds.depth*ds.num_qubits} params to optimize)
    shots {ds.n_shots} (used to estimate probabilities on hardware if > 0)
    maximal {ds.max_iter} iterations per repeat
Distribution:
    mu {das.mu}
    sigma {das.sigma}
    batch size {das.batch_size} (higher means better log-normal estimation)
Data:
    discriminator trainingset size {das.synthetic_size+das.real_size} (with {das.synthetic_size} synthetic and {das.real_size} real)
    generator trainingset size {das.gen_size}
Training:
    repeats {ts.repeats}
    printing accuracy {ts.print_accuracy} (more info during training for a small slowdown)
    showing figures is set to {ts.show_figs}
''')

def main():
    parameter_prelude()
    qgan = QGAN()
    qgan.train()
    qgan.test()


if __name__ == '__main__':
    default_excited_state = False
    settings_init()
    main()