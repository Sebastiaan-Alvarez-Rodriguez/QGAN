from numpy.random import random
from numpy import pi
from enum import Enum

class InitParamType(Enum):
    NORMAL = 1,
    RANDOM = 2,
    UNIFORM = 3


def get_init_params(paramtype, num_params):
    if paramtype == InitParamType.NORMAL:
        pass
    elif paramtype == InitParamType.RANDOM:
        return random(num_params)*2*pi
    else:
        pass