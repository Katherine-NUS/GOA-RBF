from GOArbf.epsMaSO2.multiobjective_problems import DTLZ2
from GOArbf.optimize import many_objective

data = DTLZ2(dim=10, nobj=2)
# Serial
many_objective.optimize(data)

# Parallel
# many_objective.optimize(data, run='asynchronous')
