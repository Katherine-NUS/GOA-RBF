from pySOT2.optimize import parallel_surrogate
from pySOT2.GOPS2.BBOB import BBOB

data = BBOB(id=15, instance=0, dim=40)
parallel_surrogate.optimize(data)