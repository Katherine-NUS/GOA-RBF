from pySOT2.Optimize import MOoptim
from pySOT2.GOMORS2.test_problems import DTLZ4

data = DTLZ4(nobj=2)
# Serial GOMORS as the default optimizor
MOoptim.MOoptimize(data)

# Parallel GOMORS
MOoptim.MOoptimize(data, run='asynchronous')