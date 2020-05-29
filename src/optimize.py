from scipy.optimize import minimize

from src.enums.trainingtype import TrainingType

def optimize(c: callable, paramlist, strategy: TrainingType, max_iter, step_rate):
    if strategy == TrainingType.ADAM:
        raise NotImplementedError('Do not use Adam without a gradient function (we have no gradient)... Pick another training type')
        from climin import Adam
        optimizer = Adam(wrt=paramlist,
                        # fprime=, # Need gradient of callable here, not callable itself!
                        step_rate=step_rate)
        for idx, info in enumerate(optimizer):
            if idx == max_iter:
                break
        return None, paramlist
    else: # Scipy optimizer type
        methodname = strategy.get_scipy_name()
        res = minimize(c, paramlist, method=methodname, tol=10**-4, options={'maxiter':max_iter})
        return res.fun, res.x
