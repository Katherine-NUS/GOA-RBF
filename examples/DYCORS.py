from pySOT.optimization_problems import Ackley
from pySOT2.optimize import surrogate_optimization

# Serial DYCORS as the default optimizor
ackley = Ackley(dim=10)
surrogate_optimization.optimize(ackley)

# asynchronous optimization
#surrogate_optimization.optimize(ackley, run='asynchronous')

# synchronous optimization
#surrogate_optimization.optimize(ackley, run='synchronous')