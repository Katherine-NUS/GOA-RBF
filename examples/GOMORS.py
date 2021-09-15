from pySOT2.optimize import multi_objective
from pySOT2.GOMORS2.test_problems import DTLZ4

data = DTLZ4(nobj=2)
# Serial GOMORS as the default optimizor
multi_objective.optimize(data)

# Parallel GOMORS
# multi_objective.optimize(data, run='asynchronous')