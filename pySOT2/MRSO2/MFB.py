
"""
.. module:: MFB
   :synopsis: Multi-fidelity benchmark functions

.. moduleauthor:: YI JIN <iseyij@nus.edu.sg>

:Module: MFB
:Author: YI JIN <iseyij@nus.edu.sg>


"""

import os
import sys
import random
from time import time
import numpy as np
from ..pySOT1.utils import check_opt_prob
from os.path import join
import scipy.io

# import matlab
# import matlab.engine

class MFO_1:
    """MFO-1 function
    """

    def __init__(self, dim=10):
        self.xlow = -1 * np.ones(dim)
        self.xup = 1 * np.ones(dim)
        self.dim = dim
        self.info = str(dim)+"-dimensional Rosenbrock function \n" +\
                             "Global optimum: Unknown"
        self.min = 0
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.search_type = None
        self.sv = np.zeros([self.dim])
        for i in range(self.dim):
            if i % 2 == 0:
                self.sv[i] = (self.xup[i]-self.xlow[i]) / 4
            else:
                self.sv[i] = -(self.xup[i] - self.xlow[i]) / 4
        check_opt_prob(self)

    def objfunction_LF(self, z):
        """Evaluate the LF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c=self.dim
        phi=1000
        x=np.asarray(x)
        f = c + np.sum(x**2 - np.cos(10 * np.pi * x))
        theta = 1 - 0.0001 * phi
        a = theta * np.ones([n, c])
        w = 10 * np.pi * theta * np.ones([n, c])
        b = 0.5 * np.pi * theta * np.ones([n, c])
        e = np.sum(a* np.cos(w * x+b + np.pi))
        obj = f + e
        cost = np.ones([n, 1]) * phi

        return obj

    def objfunction_HF(self, z):
        """Evaluate the HF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c = self.dim
        phi = 10000
        x = np.asarray(x)
        f = c + np.sum(x ** 2 - np.cos(10 * np.pi * x))
        theta = 1 - 0.0001 * phi
        a = theta * np.ones([n, c])
        w = 10 * np.pi * theta * np.ones([n, c])
        b = 0.5 * np.pi * theta * np.ones([n, c])
        e = np.sum(a * np.cos(w * x + b + np.pi))
        obj = f + e
        cost = np.ones([n, 1]) * phi
        return obj


class MFO_2:
    """MFO-2 function
    """

    def __init__(self, dim=10):
        self.xlow = -1 * np.ones(dim)
        self.xup = 1 * np.ones(dim)
        self.dim = dim
        self.info = str(dim) + "-dimensional Rosenbrock function \n" + \
                    "Global optimum: Unknown"
        self.min = 0
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.search_type = None
        self.sv = np.zeros([self.dim])
        for i in range(self.dim):
            if i % 2 == 0:
                self.sv[i] = (self.xup[i]-self.xlow[i]) / 4
            else:
                self.sv[i] = -(self.xup[i] - self.xlow[i]) / 4
        check_opt_prob(self)

    def objfunction_LF(self, z):
        """Evaluate the LF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c = self.dim
        phi = 1000
        x = np.asarray(x)
        f = c + np.sum(x ** 2 - np.cos(10 * np.pi * x))
        theta = np.exp(-0.00025*phi)
        a = theta * np.ones([n, c])
        w = 10 * np.pi * theta * np.ones([n, c])
        b = 0.5 * np.pi * theta * np.ones([n, c])
        e = np.sum(a * np.cos(w * x + b + np.pi))
        obj = f + e
        cost = np.ones([n, 1]) * phi

        return obj

    def objfunction_HF(self, z):
        """Evaluate the HF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c = self.dim
        phi = 10000
        x = np.asarray(x)
        f = c + np.sum(x ** 2 - np.cos(10 * np.pi * x))
        theta = np.exp(-0.00025*phi)
        a = theta * np.ones([n, c])
        w = 10 * np.pi * theta * np.ones([n, c])
        b = 0.5 * np.pi * theta * np.ones([n, c])
        e = np.sum(a * np.cos(w * x + b + np.pi))
        obj = f + e
        cost = np.ones([n, 1]) * phi
        return obj

class MFO_3:
    """MFO-3 function
    """

    def __init__(self, dim=10):
        self.xlow = -1 * np.ones(dim)
        self.xup = 1 * np.ones(dim)
        self.dim = dim
        self.info = str(dim) + "-dimensional Rosenbrock function \n" + \
                    "Global optimum: Unknown"
        self.min = 0
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.search_type = None
        self.sv = np.zeros([self.dim])
        for i in range(self.dim):
            if i % 2 == 0:
                self.sv[i] = (self.xup[i]-self.xlow[i]) / 4
            else:
                self.sv[i] = -(self.xup[i] - self.xlow[i]) / 4
        check_opt_prob(self)

    def objfunction_LF(self, z):
        """Evaluate the LF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c = self.dim
        phi = 1000
        x = np.asarray(x)
        f = c + np.sum(x ** 2 - np.cos(10 * np.pi * x))
        if phi < 1000:
            theta = 1 - 0.0002 * phi
        elif phi < 2000:
            theta = 0.8
        elif phi < 3000:
            theta = 1.2 - 0.0002 * phi
        elif phi < 4000:
            theta = 0.6
        elif phi < 5000:
            theta = 1.4 - 0.0002 * phi
        elif phi < 6000:
            theta = 0.4
        elif phi < 7000:
            theta = 1.6 - 0.0002 * phi
        elif phi < 8000:
            theta = 0.2
        elif phi < 9000:
            theta = 1.8 - 0.0002 * phi
        else:
            theta = 0
        a = theta * np.ones([n, c])
        w = 10 * np.pi * theta * np.ones([n, c])
        b = 0.5 * np.pi * theta * np.ones([n, c])
        e = np.sum(a * np.cos(w * x + b + np.pi))
        obj = f + e
        cost = np.ones([n, 1]) * phi

        return obj

    def objfunction_HF(self, z):
        """Evaluate the HF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c = self.dim
        phi = 10000
        x = np.asarray(x)
        f = c + np.sum(x ** 2 - np.cos(10 * np.pi * x))
        if phi < 1000:
            theta = 1 - 0.0002 * phi
        elif phi < 2000:
            theta = 0.8
        elif phi < 3000:
            theta = 1.2 - 0.0002 * phi
        elif phi < 4000:
            theta = 0.6
        elif phi < 5000:
            theta = 1.4 - 0.0002 * phi
        elif phi < 6000:
            theta = 0.4
        elif phi < 7000:
            theta = 1.6 - 0.0002 * phi
        elif phi < 8000:
            theta = 0.2
        elif phi < 9000:
            theta = 1.8 - 0.0002 * phi
        else:
            theta = 0
        a = theta * np.ones([n, c])
        w = 10 * np.pi * theta * np.ones([n, c])
        b = 0.5 * np.pi * theta * np.ones([n, c])
        e = np.sum(a * np.cos(w * x + b + np.pi))
        obj = f + e
        cost = np.ones([n, 1]) * phi
        return obj


class MFO_4:
    """MFO-4 function from MFB-7
    """

    def __init__(self, dim=10):
        self.xlow = -1 * np.ones(dim)
        self.xup = 1 * np.ones(dim)
        self.dim = dim
        self.info = str(dim)+"-dimensional Rosenbrock function \n" +\
                             "Global optimum: Unknown"
        self.min = 0
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.search_type = None
        self.sv = np.zeros([self.dim])
        for i in range(self.dim):
            if i % 2 == 0:
                self.sv[i] = (self.xup[i]-self.xlow[i]) / 4
            else:
                self.sv[i] = -(self.xup[i] - self.xlow[i]) / 4
        check_opt_prob(self)

    def objfunction_LF(self, z):
        """Evaluate the LF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c=self.dim
        phi=1000
        x=np.asarray(x)
        f = c + np.sum(x**2 - np.cos(10 * np.pi * x))
        theta = 1 - 0.0001 * phi
        psi = 1 - np.abs(x)
        a = theta * np.ones([n, c]) * psi
        w = 10 * np.pi * theta * np.ones([n, c])
        b = 0.5 * np.pi * theta * np.ones([n, c])
        e = np.sum(a* np.cos(w * x+b + np.pi))
        obj = f + e
        cost = np.ones([n, 1]) * (0.001*phi)**4

        return obj

    def objfunction_HF(self, z):
        """Evaluate the HF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c = self.dim
        phi = 10000
        x = np.asarray(x)
        f = c + np.sum(x ** 2 - np.cos(10 * np.pi * x))
        theta = 1 - 0.0001 * phi
        a = theta * np.ones([n, c])
        w = 10 * np.pi * theta * np.ones([n, c])
        b = 0.5 * np.pi * theta * np.ones([n, c])
        e = np.sum(a * np.cos(w * x + b + np.pi))
        obj = f + e
        cost = np.ones([n, 1]) * (0.001*phi)**4
        return obj


class MFO_5:
    """MFO-5 function
    """

    def __init__(self, dim=10):
        self.xlow = -1 * np.ones(dim)
        self.xup = 1 * np.ones(dim)
        self.dim = dim
        self.info = str(dim)+"-dimensional Rosenbrock function \n" +\
                             "Global optimum: Unknown"
        self.min = 0
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.search_type = None
        self.sv = np.zeros([self.dim])
        for i in range(self.dim):
            if i % 2 == 0:
                self.sv[i] = (self.xup[i]-self.xlow[i]) / 4
            else:
                self.sv[i] = -(self.xup[i] - self.xlow[i]) / 4
        check_opt_prob(self)

    def objfunction_LF(self, z):
        """Evaluate the LF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c=self.dim
        phi=1000
        x=np.asarray(x)
        f = c + np.sum(x**2 - np.cos(10 * np.pi * x))
        sigma = (1 - 0.0001 * phi) * c * 0.1
        mu = 0
        e = np.random.normal(0,1) * sigma + mu
        obj = f + e
        cost = np.ones([n, 1]) * phi

        return obj

    def objfunction_HF(self, z):
        """Evaluate the HF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c = self.dim
        phi = 10000
        x = np.asarray(x)
        f = c + np.sum(x ** 2 - np.cos(10 * np.pi * x))
        sigma = (1 - 0.0001 * phi) * c * 0.1
        mu = 0
        e = np.random.normal(0, 1) * sigma + mu
        obj = f + e
        cost = np.ones([n, 1]) * phi
        return obj

class MFO_6:
    """MFO-6 function
    """

    def __init__(self, dim=10):
        self.xlow = -1 * np.ones(dim)
        self.xup = 1 * np.ones(dim)
        self.dim = dim
        self.info = str(dim)+"-dimensional Rosenbrock function \n" +\
                             "Global optimum: Unknown"
        self.min = 0
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.search_type = None
        self.sv = np.zeros([self.dim])
        for i in range(self.dim):
            if i % 2 == 0:
                self.sv[i] = (self.xup[i]-self.xlow[i]) / 4
            else:
                self.sv[i] = -(self.xup[i] - self.xlow[i]) / 4
        check_opt_prob(self)

    def objfunction_LF(self, z):
        """Evaluate the LF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c=self.dim
        phi=1000
        x=np.asarray(x)
        f = c + np.sum(x**2 - np.cos(10 * np.pi * x))
        sigma = c * np.exp(-0.0005 * phi) * 0.1
        mu = 0
        e = np.random.normal(0, 1) * sigma + mu
        obj = f + e
        cost = np.ones([n, 1]) * phi

        return obj

    def objfunction_HF(self, z):
        """Evaluate the HF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c = self.dim
        phi = 10000
        x = np.asarray(x)
        f = c + np.sum(x ** 2 - np.cos(10 * np.pi * x))
        sigma = c * np.exp(-0.0005 * phi) * 0.1
        mu = 0
        e = np.random.normal(0, 1) * sigma + mu
        obj = f + e
        cost = np.ones([n, 1]) * phi
        return obj


class MFO_7:
    """MFO-7 function
    """

    def __init__(self, dim=10):
        self.xlow = -1 * np.ones(dim)
        self.xup = 1 * np.ones(dim)
        self.dim = dim
        self.info = str(dim)+"-dimensional Rosenbrock function \n" +\
                             "Global optimum: Unknown"
        self.min = 0
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.search_type = None
        self.sv = np.zeros([self.dim])
        for i in range(self.dim):
            if i % 2 == 0:
                self.sv[i] = (self.xup[i]-self.xlow[i]) / 4
            else:
                self.sv[i] = -(self.xup[i] - self.xlow[i]) / 4
        check_opt_prob(self)

    def objfunction_LF(self, z):
        """Evaluate the LF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c=self.dim
        phi=1000
        x=np.asarray(x)
        f = c + np.sum(x**2 - np.cos(10 * np.pi * x))
        sigma = (1 - 0.0001 * phi) * 0.1
        mu = np.sum(np.ones([n,c]) - np.abs(x))*sigma
        e = np.random.normal(0, 1) * sigma + mu
        obj = f + e
        cost = np.ones([n, 1]) * phi

        return obj

    def objfunction_HF(self, z):
        """Evaluate the HF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c = self.dim
        phi = 10000
        x = np.asarray(x)
        f = c + np.sum(x ** 2 - np.cos(10 * np.pi * x))
        f = c + np.sum(x ** 2 - np.cos(10 * np.pi * x))
        sigma = (1 - 0.0001 * phi) * 0.1

        mu = np.sum(np.ones([n,c]) - np.abs(x)) * sigma
        e = np.random.normal(0, 1) * sigma + mu
        obj = f + e
        cost = np.ones([n, 1]) * phi
        return obj


class MFO_8:
    """MFO-8 function
    """

    def __init__(self, dim=10):
        self.xlow = -1 * np.ones(dim)
        self.xup = 1 * np.ones(dim)
        self.dim = dim
        self.info = str(dim)+"-dimensional Rosenbrock function \n" +\
                             "Global optimum: Unknown"
        self.min = 0
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.search_type = None
        self.sv = np.zeros([self.dim])
        for i in range(self.dim):
            if i % 2 == 0:
                self.sv[i] = (self.xup[i]-self.xlow[i]) / 4
            else:
                self.sv[i] = -(self.xup[i] - self.xlow[i]) / 4
        check_opt_prob(self)

    def objfunction_LF(self, z):
        """Evaluate the LF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c=self.dim
        phi=1000
        x=np.asarray(x)
        f = c + np.sum(x**2 - np.cos(10 * np.pi * x))
        sigma = np.exp(-0.0005 * phi) * 0.1
        mu = np.sum(1 - np.abs(x))*sigma
        e = np.random.normal(0, 1) * sigma + mu
        obj = f + e
        cost = np.ones([n, 1]) * phi

        return obj

    def objfunction_HF(self, z):
        """Evaluate the HF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c = self.dim
        phi = 10000
        x = np.asarray(x)
        f = c + np.sum(x ** 2 - np.cos(10 * np.pi * x))
        sigma = np.exp(-0.0005 * phi) * 0.1
        mu = np.sum(1 - np.abs(x)) * sigma
        e = np.random.normal(0, 1) * sigma + mu
        obj = f + e
        cost = np.ones([n, 1]) * phi
        return obj

class MFO_9:
    """MFO-9 function
    """

    def __init__(self, dim=10):
        self.xlow = -1 * np.ones(dim)
        self.xup = 1 * np.ones(dim)
        self.dim = dim
        self.info = str(dim)+"-dimensional Rosenbrock function \n" +\
                             "Global optimum: Unknown"
        self.min = 0
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.search_type = None
        self.sv = np.zeros([self.dim])
        for i in range(self.dim):
            if i % 2 == 0:
                self.sv[i] = (self.xup[i]-self.xlow[i]) / 4
            else:
                self.sv[i] = -(self.xup[i] - self.xlow[i]) / 4
        check_opt_prob(self)

    def objfunction_LF(self, z):
        """Evaluate the LF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c=self.dim
        phi=1000
        x=np.asarray(x)
        f = c + np.sum(x ** 2 - np.cos(10 * np.pi * x))
        p = 0.1 * (1 - 0.0001 * phi)
        e = 0
        if np.random.random() <= p:
            e = 10 * c
        obj = f + e
        cost = np.ones([n, 1]) * phi

        return obj

    def objfunction_HF(self, z):
        """Evaluate the HF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c = self.dim
        phi = 10000
        x = np.asarray(x)
        f = c + np.sum(x ** 2 - np.cos(10 * np.pi * x))
        p = 0.1 * (1 - 0.0001 * phi)
        e = 0
        if np.random.random() <= p:
            e = 10 * c
        obj = f + e
        cost = np.ones([n, 1]) * phi
        return obj


class MFO_10:
    """MFO-10 function
    """

    def __init__(self, dim=10):
        self.xlow = -1 * np.ones(dim)
        self.xup = 1 * np.ones(dim)
        self.dim = dim
        self.info = str(dim)+"-dimensional Rosenbrock function \n" +\
                             "Global optimum: Unknown"
        self.min = 0
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.search_type = None
        self.sv = np.zeros([self.dim])
        for i in range(self.dim):
            if i % 2 == 0:
                self.sv[i] = (self.xup[i]-self.xlow[i]) / 4
            else:
                self.sv[i] = -(self.xup[i] - self.xlow[i]) / 4
        check_opt_prob(self)

    def objfunction_LF(self, z):
        """Evaluate the LF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c=self.dim
        phi=1000
        x=np.asarray(x)
        f = c + np.sum(x**2 - np.cos(10 * np.pi * x))
        p=np.exp(-0.001 * (phi + 100))
        e = 0
        if np.random.random() <= p:
            e = 10 * c
        obj = f + e
        cost = np.ones([n, 1]) * phi
        return obj

    def objfunction_HF(self, z):
        """Evaluate the HF function  at x
        """
        x = z - self.sv
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = 1
        c = self.dim
        phi = 10000
        x = np.asarray(x)
        f = c + np.sum(x ** 2 - np.cos(10 * np.pi * x))
        p=np.exp(-0.001 * (phi + 100))
        e=0
        if np.random.random() <= p:
            e = 10 * c
        obj = f + e
        cost = np.ones([n, 1]) * phi
        return obj



class Styblinski_Tang:
    """Styblinski-Tang function
    """

    def __init__(self, dim=10):
        self.xlow = -np.ones(dim)
        self.xup = np.ones(dim)
        self.dim = dim
        self.info = str(dim)+"-dimensional Styblinski-Tang function \n" +\
                             "Global optimum: Unknown"
        self.min = 0
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.search_type = None
        check_opt_prob(self)

    def objfunction_LF(self, x):
        """Evaluate the LF function  at x
        """
        # if len(x) != self.dim:
        #     raise ValueError('Dimension mismatch')
        x = np.asarray(x).reshape(-1, self.dim)
        if x.shape[0] > 1:
            obj = np.zeros([x.shape[0]])
            for h in range(x.shape[0]):
                n = 1
                c = self.dim
                phi = 1000
                z = 5*np.asarray(x[h, :])
                f = c + 0.5*np.sum(z**4 - 16*z**2+5*z)
                e = np.random.uniform(0.9, 1.1, 1)
                obj[h] = f*float(e)
        else:
            n = 1
            c = self.dim
            phi = 1000
            x = 5 * np.asarray(x[0])
            f = c + 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x)
            e = np.random.uniform(0.8, 1.2, 1)
            obj = f * float(e)
        return obj

    def objfunction_HF(self, x):
        """Evaluate the HF function  at x
        """

        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        x = np.asarray(x).reshape(-1, self.dim)
        if x.shape[0] > 1:
            obj = np.zeros([x.shape[0]])
            for h in range(x.shape[0]):
                n = 1
                c = self.dim
                phi = 10000
                z = 5 * np.asarray(x[h, :])
                f = c + 0.5 * np.sum(z ** 4 - 16 * z ** 2 + 5 * z)
                obj[h] = f
        else:
            n = 1
            c = self.dim
            phi = 10000
            x = 5 * np.asarray(x[0])
            f = c + 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x)
            obj = f
        return obj






#
# class Capacity_planning_1:
#     """Capacity_planning function
#     """
#
#     def __init__(self, dim=10, eng=None):
#         self.dim = dim
#         if self.dim != 12:
#             raise ValueError('Dimension mismatch')
#         self.eng = eng
#         self.xlow = np.asarray([5.05,5.05,5.05,10.1,15.15,15.15,15.15,5.05,10.1,5.05,5.05,5.05])
#         self.xup = 3 * np.asarray([5.05,5.05,5.05,10.1,15.15,15.15,15.15,5.05,10.1,5.05,5.05,5.05])
#         self.info = str(dim)+"-dimensional Capacity_planning function \n" +\
#                              "Global optimum: Unknown"
#         self.min = 0
#         self.integer = []
#         self.continuous = np.arange(0, dim)
#         self.search_type = None
#         check_opt_prob(self)
#
#     def objfunction_LF(self, x):
#         """Evaluate the LF function  at x
#         """
#
#         params = list(x)
#         self.eng.workspace['y'] = matlab.double(params)
#         result = self.eng.eval('exact_LowFidelity12D(y)')
#
#
#
#
#         return result
#
#     def objfunction_HF(self, x):
#         """Evaluate the HF function  at x
#         """
#
#         params = list(x)
#         self.eng.workspace['y'] = matlab.double(params)
#         result = self.eng.eval('HighFidelity12D(y)')
#
#         return result
#
#
#
#
#
#
# class Capacity_planning_2:
#     """Capacity_planning function
#     """
#
#     def __init__(self, dim=10, eng=None):
#         self.dim = dim
#         if self.dim != 12:
#             raise ValueError('Dimension mismatch')
#         self.eng = eng
#         self.xlow = np.asarray([5.05, 5.05, 5.05, 10.1, 15.15, 15.15, 15.15, 5.05, 10.1, 5.05, 5.05, 5.05])
#         self.xup = 3 * np.asarray([5.05, 5.05, 5.05, 10.1, 15.15, 15.15, 15.15, 5.05, 10.1, 5.05, 5.05, 5.05])
#         self.info = str(dim)+"-dimensional Capacity_planning function \n" +\
#                              "Global optimum: Unknown"
#         self.min = 0
#         self.integer = []
#         self.continuous = np.arange(0, dim)
#         self.search_type = None
#         check_opt_prob(self)
#
#     def objfunction_LF(self, x):
#         """Evaluate the LF function  at x
#         """
#         #dir = 'F:/YiJin_Work/Bi-layer-RBF-DYCORS/program/Python_Code/MFO_RBF-DYCORS/simulation_problem/12D/'
#
#
#         params = list(x)
#         self.eng.workspace['y'] = matlab.double(params)
#         result = self.eng.eval('sim_LowFidelity12D(y)')
#
#
#
#
#         return result
#
#     def objfunction_HF(self, x):
#         """Evaluate the HF function  at x
#         """
#
#         params = list(x)
#         self.eng.workspace['y'] = matlab.double(params)
#         result = self.eng.eval('HighFidelity12D(y)')
#
#         return result







if __name__=="__main__":


    eng = matlab.engine.start_matlab()
    data = Capacity_planning_1(dim=12, eng=eng)
    x = matlab.double(3*[5.05, 5.05, 5.05, 10.1, 15.15, 15.15, 15.15, 5.05, 10.1, 5.05, 5.05, 5.05])
    print('HF value is: ', data.objfunction_HF(x), 'LF value is:', data.objfunction_LF(x))
