from src.discriminator import Discriminator
from src.generator import Generator

from src.data import get_real_samples

from src.settings import g_settings as gs
from src.settings import settings_init

'''
The plan:
1 train the generator to do what I propose above 
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
        self.d = Discriminator()
        self.g = Generator()


    def generate_dataset(self, size_fake, size_real):
        return self.g.gen_synthetics(size_fake), get_real_samples(size_real)
        #Should both return a dataset of [[point0, point1..., point7], ...]


    def train(self, repeats=1, fakes=10, reals=10):
        for x in range(repeats):
            f, r = self.generate_dataset(fakes, reals)
            for x in f:
                print(f'Fake: {x}')
            for x in r:
                print(f'Real: {x}')


            # TODO: Train discriminator

            # TODO: Train generator (using loss function discriminator?)


def main():
    qgan = QGAN()
    qgan.train()


if __name__ == '__main__':
    default_excited_state = False
    settings_init()
    main()