from GOArbf.optimize import parallel_surrogate
from GOArbf.GOPS2.BBOB import BBOB

data = BBOB(id=15, instance=0, dim=40)
parallel_surrogate.optimize(data)