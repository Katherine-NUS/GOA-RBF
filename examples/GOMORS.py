from GOMORS2.MOoptim import MOoptimize
from GOMORS2.test_problems import DTLZ4

data = DTLZ4(nobj=2)
# Serial GOMORS as the default optimizor
MOoptimize(data)

# Parallel GOMORS
MOoptimize(data, run='asynchronous')

#new