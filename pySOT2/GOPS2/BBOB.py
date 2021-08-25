
import numpy as np
import time
from .bbobbenchmarks import instantiate


class BBOB(object):
    def __init__(self, id=15, instance=None, dim=10):
        self.xlow = -5 * np.ones(dim)
        self.xup = 5 * np.ones(dim)
        self.dim = dim
        self.id = id
        self.instance = instance
        self.info = str(dim)+"-dimensional BBOB F" + str(id) + " function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.init()

    def init(self):
        if self.instance is not None:
            self.func, self.fopt = instantiate(self.id,self.instance)
        else:
            self.func, self.fopt = instantiate(self.id)
        self.func.evaluate([0]*self.dim)

    def objfunction(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        return self.func.evaluate(x)

print(BBOB(15))