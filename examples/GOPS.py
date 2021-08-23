from GOPS2.GOPSoptim import GOPSoptimize
from GOPS2.BBOB import BBOB

data = BBOB(id=15, instance=0, dim=40)
GOPSoptimize(data)