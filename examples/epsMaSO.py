from pySOT2.epsMaSO2.multiobjective_problems import DTLZ2
from pySOT2.Optimize import epsMOoptim

data = DTLZ2(dim=10, nobj=2)
# Serial
epsMOoptim.epsMOoptimize(data)

# Parallel
epsMOoptim.epsMOoptimize(data, run='asynchronous')
