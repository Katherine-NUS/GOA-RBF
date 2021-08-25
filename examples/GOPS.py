from pySOT2.Optimize import GOPSoptim
from pySOT2.GOPS2.BBOB import BBOB

data = BBOB(id=15, instance=0, dim=40)
GOPSoptim.GOPSoptimize(data)