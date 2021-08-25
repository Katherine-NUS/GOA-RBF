from pySOT.optimization_problems import Ackley
from pySOT2.Optimize import Optim

# Serial DYCORS as the default optimizor
ackley = Ackley(dim=10)
Optim.optimize(ackley)

# asynchronous optimization
#Optim.optimize(ackley, run='asynchronous')

# synchronous optimization
#Optim.optimize(ackley, run='synchronous')