from pySOT2.optimize import multi_fidelity
from pySOT2.MRSO2.MFO_main import function_object

data = function_object(1, 5, [])
multi_fidelity.optimize(data)