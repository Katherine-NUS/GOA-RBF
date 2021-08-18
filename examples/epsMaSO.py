from epsMaSO2.multiobjective_problems import DTLZ2
from epsMaSO2.epsMOoptim import epsMOoptimize

data = DTLZ2(dim=10, nobj=2)
# Serial
epsMOoptimize(data)

# Parallel
epsMOoptimize(data, run='asynchronous')
