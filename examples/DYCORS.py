from pySOT.optimization_problems import Ackley
from pySOT2.optim import optimize

# Serial DYCORS as the default optimizor
ackley = Ackley(dim=10)
optimize(ackley)

# asynchronous optimization
optimize(ackley, run='asynchronous')

# synchronous optimization
optimize(ackley, run='synchronous')