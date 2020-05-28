from itertools import chain

from src.discriminator.type1 import Discriminator1
from src.discriminator.type2 import Discriminator2
from src.generator.generator import Generator

from src.data import get_real_samples

from src.settings import g_settings as gs
from src.settings import d_settings as ds
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


    def generate_dataset(self):
        return self.g.gen_synthetics(das.items_synthetic_size), get_real_samples(das.items_real_size)
        #Should both return a dataset of [[point0, point1..., point7], ...]

# Andere manieren om real/fake data in de discriminator
# Nu: 8 qubits, losse kans op 1 qubits (optie 2)
# (optie 1): meer representatie
#   Geen encoding
#   Hardcode data in quantum state (3-qubit state, 8 computational basis state)
#   a|x> for x in range(8), kan afhankelijk zijn van a
#   a_x = sqrt(point(x)) (point -> 1 van entries )
#   a_0|0> + a_1|1> + a_2|2> + ... -> in variational circuit van discriminator, zelfde loss function
#   
# (optie 3): we hebben probabilities niet, alleen functie voor probs
#   Bepaal kansen door repeats
#   Schatting van kansen, en dan zelfde optie 2
# 
# quantum RAM problem (opzoeken)


    def train(self, repeats=1):
        for x in range(repeats):
            f, r = self.generate_dataset()
            d_dataset =list(chain(f, r))
            d_labels = list((0 for x in range(das.items_synthetic_size)))
            d_labels.extend((1 for x in range(das.items_real_size)))
            self.d.train(d_dataset, d_labels, printing=True)

            self.g.train(self.d)
            
            # cost-to-minimize:
            #  neemt parameters van de generator
            #  aan de hand van params genereer ik data
            #  op data laat je discriminator los (aka genereer labels)
            #  mean_squared_loss van labels en lijst met 1'en


            # TODO: Train generator (using loss function discriminator?)


def main():
    qgan = QGAN()
    qgan.train()


if __name__ == '__main__':
    default_excited_state = False
    settings_init()
    main()