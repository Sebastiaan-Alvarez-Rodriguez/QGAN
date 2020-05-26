from enum import Enum

class TrainingType(Enum):
    ADAM = 1,
    BFGS = 2,

    def get_scipy_name(self):
        if self == TrainingType.ADAM:
            raise ValueError('Adam is not in scipy')
        elif self == TrainingType.BFGS:
            return 'L-BFGS-B'