from pySOT2.Optimize import MFoptim
from pySOT2.MRSO2.MFO_main import function_object

data = function_object(1, 5, [])
MFoptim.MFoptimize(data)