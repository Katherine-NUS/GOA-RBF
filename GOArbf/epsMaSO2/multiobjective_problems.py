import time
import math
from .mo_utils import *
from pySOT.optimization_problems import OptimizationProblem


class MOEADDE_F1(OptimizationProblem):
    def __init__(self, dim = 10, nobj = 2):
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = self.paretofront()

    def eval(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        F = np.zeros(2)
        count_even = 0.0
        count_odd = 0.0

        F[0] = 0.0
        F[1] = 0.0

        i = 2
        while(i <= self.dim):
            y = x[i - 1] - np.power(x[0], 0.5 * (1.0 + 3.0 * float(i - 2) / float(self.dim - 2)))
            if i % 2 == 0:
                F[1] += np.power(y, 2.0)
                count_even += 1
            else:
                F[0] += np.power(y, 2.0)
                count_odd += 1
            i += 1


        F[0] = F[0] * 2.0 / float(count_odd)
        F[1] = F[1] * 2.0 / float(count_even)

        F[0] += x[0]
        F[1] += 1.0 - np.sqrt(x[0])
        return F

    def paretofront(self):
        F = np.zeros([1001, 2])
        for i in range(1001):
            F[i, 0] = np.double(i) / 1000
            F[i, 1] = 1 - F[i, 0] ** 0.5
        return F


class MOEADDE_F2(OptimizationProblem):
    def __init__(self, dim = 10, nobj = 2):
        self.dim = dim
        self.nobj = nobj

        self.lb = np.ones(dim) * (-1)
        self.ub = np.ones(dim) * (1)
        self.lb[0] = 0

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = self.paretofront()

    def eval(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        F = np.zeros(2)
        count_even = 0.0
        count_odd = 0.0

        F[0] = 0.0
        F[1] = 0.0

        i = 2
        while(i <= self.dim):
            y = x[i - 1] - np.sin(6.0 * np.pi * x[0] + float(i) * np.pi / float(self.dim))
            if i % 2 == 0:
                F[1] += np.power(y, 2.0)
                count_even += 1
            else:
                F[0] += np.power(y, 2.0)
                count_odd += 1
            i += 1


        F[0] = F[0] * 2.0 / float(count_odd)
        F[1] = F[1] * 2.0 / float(count_even)

        F[0] += x[0]
        F[1] += 1.0 - np.sqrt(x[0])
        return F

    def paretofront(self, exp_points = None):
        F = np.zeros([1001, 2])
        for i in range(1001):
            F[i, 0] = np.double(i) / 1000
            F[i, 1] = 1 - F[i, 0] ** 0.5
        return F


class MOEADDE_F7(OptimizationProblem):
    def __init__(self, dim = 10, nobj = 2):
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = self.paretofront()

    def eval(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        F = np.zeros(2)
        count_even = 0.0
        count_odd = 0.0

        F[0] = 0.0
        F[1] = 0.0

        i = 2
        y = np.copy(x)
        while(i <= self.dim):
            y[i - 1] = y[i - 1] - np.power(x[0], 0.5 * (1.0 + 3.0 * float(i - 2) / float(self.dim - 2)))
            if i % 2 == 0:
                F[1] += (4.0 * np.power(y[i - 1], 2.0) - np.cos(8.0 * np.pi * y[i - 1]) + 1.0)
                count_even += 1
            else:
                F[0] += (4.0 * np.power(y[i - 1], 2.0) - np.cos(8.0 * np.pi * y[i - 1]) + 1.0)
                count_odd += 1
            i += 1


        F[0] = F[0] * 2.0 / float(count_odd)
        F[1] = F[1] * 2.0 / float(count_even)

        F[0] += x[0]
        F[1] += 1.0 - np.sqrt(x[0])
        return F

    def paretofront(self):
        F = np.zeros([1001, 2])
        for i in range(1001):
            F[i, 0] = np.double(i) / 1000
            F[i, 1] = 1 - F[i, 0] ** 0.5
        return F


class ZDT4(OptimizationProblem):
    def __init__(self, dim = 8, nobj = 2):
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim) * 4.0
        self.lb[0] = 0.0
        self.ub[0] = 1.0

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = self.paretofront()

    def eval(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        F = np.zeros(2)
        F[0] = x[0]

        g = 1 + 10 * (self.dim - 1)
        for i in range(1, self.dim):
            g += ((x[i] - 2.0) ** 2.0 - 10 * np.cos(4 * np.pi * (x[i] - 2.0)))

        F[1] = g * (1 - (F[0] / g) ** 0.5)
        return F

    def paretofront(self, exp_points = None):
        g = 1
        F = np.zeros([1001, 2])
        for i in range(1001): # true pareto front if known
            F[i, 0] = np.double(i) / 1000
            F[i, 1] = g * (1 - (F[i, 0] / g) ** 0.5)
        return F


class ZDT1(OptimizationProblem):
    def __init__(self, dim = 8, nobj = 2):
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = self.paretofront()

    def eval(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension dismatch')
        F = np.zeros(2)
        F[0] = x[0]
        t = 0
        for i in range(1, self.dim):
            t += x[i]
        g = 1 + 9 * (t / (self.dim - 1))
        F[1] = g * (1 - np.sqrt(F[0] / g))
        return F

    def paretofront(self):
        g = 1
        F = np.zeros([1001, 2])
        for i in range(1001):
            F[i, 0] = np.double(i) / 1000
            F[i, 1] = g * (1 - np.sqrt(F[i, 0] / g))
        return F


class ZDT2(OptimizationProblem):
    def __init__(self, dim = 8, nobj = 2):
        self.dim = dim
        self.nobj = nobj

        self.lb =  np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = self.paretofront()

    def eval(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        F = np.zeros(2)
        F[0] = x[0]
        t = 0
        for i in range(1, self.dim):
            t += x[i]
        g = 1 + 9 * (t / (self.dim - 1))
        F[1] = g * (1 - (F[0] / g) ** 2)
        return F

    def paretofront(self):
        g = 1
        F = np.zeros([1001, 2])
        for i in range(1001): # true pareto front if known
            F[i, 0] = np.double(i) / 1000
            F[i, 1] = g * (1 - (F[i, 0] / g) ** 2)
        return F


class LZF1(OptimizationProblem):
    def __init__(self, dim = 8, nobj = 2):
        self.dim = dim
        self.nobj = nobj

        self.lb =  np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = self.paretofront()

    def eval(self, x):
        #time.sleep(np.random.uniform(3, 6))

        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        F = np.zeros(2)
        t_even = 0
        count_even = 0
        t_odd = 0
        count_odd = 0
        i = 2
        xnew = np.copy(x)
        while(i <= self.dim):
            xnew[i - 1] = -1 + 2 * xnew[i - 1]
            if((i % 2) == 0):
                t_even = t_even + (xnew[i - 1] - np.sin(6 * np.pi * xnew[0] + i * np.pi / self.dim)) ** 2
                count_even = count_even + 1
            else:
                t_odd = t_odd + (xnew[i - 1] - np.sin(6 * np.pi * xnew[0] + i * np.pi / self.dim)) ** 2
                count_odd = count_odd + 1
            i = i + 1
        F[0] = xnew[0] + 2 * t_odd / count_odd
        F[1] = 1 - np.sqrt(xnew[0]) + 2 * t_even / float(count_even)
        return F

    def paretofront(self):
        F = np.zeros([1001, 2])
        for i in range(1001):
            F[i, 0] = np.double(i) / 1000
            F[i, 1] = 1 - np.sqrt(F[i, 0])
        return F


class LZF4(OptimizationProblem):
    def __init__(self, dim = 8, nobj = 2):
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = self.paretofront()

    def eval(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        F = np.zeros(2)
        sum_even = 0
        count_even = 0
        sum_odd = 0
        count_odd = 0
        i = 2
        xnew = np.copy(x)
        while(i <= self.dim):
            xnew[i - 1] = -2 + 4 * xnew[i - 1]
            if (i % 2) == 0:
                y = xnew[i - 1] - np.sin(6 * np.pi * xnew[0] + i * np.pi / self.dim)
                h = np.abs(y) / (1.0 + np.exp(2 * np.abs(y)))
                sum_even = sum_even + h
                count_even = count_even + 1
            else:
                y = xnew[i - 1] - np.sin(6 * np.pi * xnew[0] + i * np.pi / self.dim)
                h = np.abs(y) / (1.0 + np.exp(2 * np.abs(y)))
                sum_odd = sum_odd + h
                count_odd = count_odd + 1
            i += 1
        F[0] = xnew[0] + 2 * sum_odd / float(count_odd)
        F[1] = 1 - xnew[0] ** 2 + 2 * sum_even / float(count_even)
        return F

    def paretofront(self):
        F = np.zeros([1001, 2])
        for i in range(1001):
            F[i, 0] = np.double(i) / 1000
            F[i, 1] = 1 - F[i, 0] ** 2
        return F


class DTLZ1(OptimizationProblem):
    def __init__(self, nobj = 2, dim = None):
        if dim is None:
            dim = nobj + 4
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = self.paretofront(5000)

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = 100.0 * (k + sum(
            [math.pow(x - 0.5, 2.0) - math.cos(20.0 * math.pi * (x - 0.5)) for x in solution[self.dim - k:]]))
        f = [0.5 * (1.0 + g)] * self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj - i - 1):
                f[i] *= solution[j]
            if i > 0:
                f[i] *= 1 - solution[self.nobj-i-1]
        f = np.asarray(f)
        return f

    def paretofront(self, exp_npoints):
        J = uniform_points(exp_npoints, self.nobj)
        for points in J:
            points /= 2.0
        return J


class DTLZ2(OptimizationProblem):
    def __init__(self, nobj = 2, dim = None):
        if dim is None:
            dim = nobj + 9
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = self.paretofront(5000)

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = sum([math.pow(x - 0.5, 2.0) for x in solution[self.dim - k:]])
        f = [1.0 + g] * self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj - i - 1):
                f[i] *= math.cos(0.5 * math.pi * solution[j])
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi * solution[self.nobj - i - 1])
        f = np.asarray(f)
        return f

    def paretofront(self, exp_npoints):
        J = uniform_points(exp_npoints, self.nobj)
        for points in J:
            temp = math.sqrt(sum([point ** 2 for point in points]))
            points /= float(temp)
        return J


class DTLZ3(OptimizationProblem):
    def __init__(self, nobj = 2, dim = None):
        if dim is None:
            dim = nobj + 9
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = self.paretofront(5000)

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = 100.0 * (k + sum(
            [math.pow(x - 0.5, 2.0) - math.cos(20.0 * math.pi * (x - 0.5)) for x in solution[self.dim - k:]]))
        f = [1.0 + g] * self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj - i - 1):
                f[i] *= math.cos(0.5 * math.pi * solution[j])
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi * solution[self.nobj - i - 1])
        f = np.asarray(f)
        return f

    def paretofront(self, exp_npoints):
        J = uniform_points(exp_npoints, self.nobj)
        for points in J:
            temp = math.sqrt(sum([point ** 2 for point in points]))
            points /= float(temp)
        return J


class DTLZ4(OptimizationProblem):
    def __init__(self, nobj = 2, dim = None):
        if dim is None:
            dim = nobj + 9
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = self.paretofront(5000)

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        alpha = 100.0
        g = sum([math.pow(x - 0.5, 2.0) for x in solution[self.dim - k:]])
        f = [1.0 + g] * self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj - i - 1):
                f[i] *= math.cos(0.5 * math.pi * math.pow(solution[j], alpha))
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi * math.pow(solution[self.nobj - i - 1], alpha))
        f = np.asarray(f)
        return f

    def paretofront(self, exp_npoints):
        J = uniform_points(exp_npoints, self.nobj)
        for points in J:
            temp = math.sqrt(sum([point ** 2 for point in points]))
            points /= float(temp)
        return J


class DTLZ5(OptimizationProblem):
    def __init__(self, nobj = 2, dim = None):
        if dim is None:
            dim = nobj + 9
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = self.paretofront(5000)

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = sum([math.pow(x - 0.5, 2.0) for x in solution[self.dim-k:]])
        f = [1.0 + g]*self.nobj

        for i in range(self.nobj):
            for j in range(1, self.nobj-i-1):
                f[i] *= math.cos(0.5 * math.pi / (2.0 * (1.0 + g)) * (1.0 + 2.0 * g * solution[j]))

            if i > 0:
                if self.nobj-i-1 != 0:
                    f[i] *= math.sin(0.5 * math.pi / (2.0 * (1.0 + g)) * (1.0 + 2.0 * g * solution[self.nobj-i-1]))
                else:
                    f[i] *= math.sin(0.5 * math.pi * solution[0])

            if self.nobj - i - 1 != 0:
                f[i] *= math.cos(0.5 * math.pi * solution[0])

        f = np.asarray(f)
        return f

    def paretofront(self, exp_npoints):
        P = [list(np.linspace(0, 1, exp_npoints)), list(np.linspace(1, 0, exp_npoints))]
        P = np.asarray(P).transpose()

        P = [[row[0] / np.sqrt(row[0] ** 2 + row[1] ** 2), row[1] / np.sqrt(row[0] ** 2 + row[1] ** 2)] for row in P]
        newP = []
        for row in P:
            newrow = []
            for _ in range(self.nobj - 1):
                newrow.append(row[0])
            newrow.append(row[1])
            newP.append(newrow)

        temp = list(range(self.nobj - 1, -1, -1))
        temp[0] -= 1
        P = np.divide(newP, np.power(np.sqrt(2.0), npmat.repmat(temp, exp_npoints, 1)))
        return P


class DTLZ6(OptimizationProblem):
    def __init__(self, nobj = 2, dim = None):
        if dim is None:
            dim = nobj + 9
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = self.paretofront(5000)

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = sum([math.pow(x, 0.1) for x in solution[self.dim - k:]])
        f = [1.0 + g] * self.nobj

        for i in range(self.nobj):
            for j in range(1, self.nobj - i - 1):
                f[i] *= math.cos(0.5 * math.pi / (2.0 * (1.0 + g)) * (1.0 + 2.0 * g * solution[j]))

            if i > 0:
                if self.nobj - i - 1 != 0:
                    f[i] *= math.sin(0.5 * math.pi / (2.0 * (1.0 + g)) * (1.0 + 2.0 * g * solution[self.nobj - i - 1]))
                else:
                    f[i] *= math.sin(0.5 * math.pi * solution[0])

            if self.nobj - i - 1 != 0:
                f[i] *= math.cos(0.5 * math.pi * solution[0])

        f = np.asarray(f)
        return f

    def paretofront(self, exp_npoints):
        P = [list(np.linspace(0, 1, exp_npoints)), list(np.linspace(1, 0, exp_npoints))]
        P = np.asarray(P).transpose()

        P = [[row[0] / np.sqrt(row[0] ** 2 + row[1] ** 2), row[1] / np.sqrt(row[0] ** 2 + row[1] ** 2)] for row in P]
        newP = []
        for row in P:
            newrow = []
            for _ in range(self.nobj - 1):
                newrow.append(row[0])
            newrow.append(row[1])
            newP.append(newrow)

        temp = list(range(self.nobj - 1, -1, -1))
        temp[0] -= 1
        P = np.divide(newP, np.power(np.sqrt(2.0), npmat.repmat(temp, exp_npoints, 1)))
        return P


class DTLZ7(OptimizationProblem):
    def __init__(self, nobj = 2, dim = None):
        if dim is None:
            dim = nobj + 19
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = self.paretofront(5000)

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = 1.0 + (sum([x for x in solution[self.dim-k:]])) * 9.0 / float(k)
        f = [1.0]*self.nobj

        for i in range(self.nobj):
            if i < self.nobj - 1:
                f[i] = solution[i]
            else:
                h = 0
                for j in range(self.nobj - 1):
                    h += f[j] / (1.0 + g) * (1.0 + np.sin(3.0 * np.pi * f[j]))
                h = self.nobj - h
                f[i] = (1.0 + g) * h

        f = np.asarray(f)
        return f

    def paretofront(self, exp_npoints):
        interval = [0, 0.251412, 0.631627, 0.859401]
        median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])

        if self.nobj > 2:
            exp_npoints = int(np.ceil(exp_npoints ** (1.0 / (self.nobj - 1))) ** (self.nobj - 1))
            num = exp_npoints ** (1.0 / (self.nobj - 1))
            Gap = list(np.linspace(0, 1, int(num)))
            num = min([num, len(Gap)])
            exp_npoints = int(num ** (self.nobj - 1))

            label = [0] * (self.nobj - 1)
            X = []

            for i in range(exp_npoints):
                row = []
                for j in range(self.nobj - 1):
                    row.append(Gap[int(label[j])])
                X.append(row)

                label[self.nobj - 2] += 1
                for j in range(self.nobj - 2, 0, -1):
                    label[j - 1] += label[j] // num
                    label[j] = label[j] % num

            for i in range(len(X)):
                for j in range(len(X[0])):
                    if X[i][j] > median:
                        X[i][j] = (X[i][j] - median) * (interval[3] - interval[2]) / (1 - median) + interval[2]
                    else:
                        X[i][j] = X[i][j] * (interval[1] - interval[0]) / median + interval[0]

            Xnew = [[2.0 * (self.nobj - sum([x / 2.0 * (1.0 + np.sin(3.0 * np.pi * x)) for x in row]))] for row in X]
            P = np.hstack((X, Xnew))
        else:
            X = list(np.linspace(0, 1, exp_npoints))
            for i in range(len(X)):
                if X[i] > median:
                    X[i] = (X[i] - median) * (interval[3] - interval[2]) / (1 - median) + interval[2]
                else:
                    X[i] = X[i] * (interval[1] - interval[0]) / median + interval[0]
            Xnew = [2.0 * (self.nobj - x / 2.0 * (1 + np.sin(3 * np.pi * x))) for x in X]
            P = np.vstack((X, Xnew))
            P = P.transpose()

        return P


class MaF1(OptimizationProblem):
    def __init__(self, nobj = 2):
        dim = nobj + 9
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = None

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = sum([math.pow(x - 0.5, 2.0) for x in solution[self.dim - k:]])
        f = [1.0 + g] * self.nobj

        for i in range(self.nobj):
            temp = 1.0
            for j in range(self.nobj - i - 1):
                temp *= solution[j]
            if i > 0:
                temp *= 1 - solution[self.nobj-i-1]
            f[i] = (1 - temp) * f[i]
        f = np.asarray(f)
        return f

    def paretofront(self, exp_npoints):
        pass


class MaF2:
    def __init__(self, nobj = 2):
        dim = nobj + 9
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = None

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = [0.0] * self.nobj
        f = [0.0] * self.nobj

        for i in range(self.nobj):
            if i < self.nobj - 1:
                for j in range(int(self.nobj + i * np.floor(float(k) / self.nobj) - 1), int(self.nobj + (i + 1) * np.floor(float(k) / self.nobj) - 1)):
                    g[i] += math.pow((0.5 * solution[j] - 0.25), 2.0)
            else:
                for j in range(int(self.nobj + i * np.floor(float(k) / self.nobj) - 1), int(self.dim)):
                    g[i] += math.pow((0.5 * solution[j] - 0.25), 2.0)
            f[i] = 1.0 + g[i]

        for i in range(self.nobj):
            for j in range(self.nobj-i-1):
                f[i] *= math.cos(0.5 * math.pi * (0.5 * solution[j] + 0.25))
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi * (0.5 * solution[self.nobj-i-1] + 0.25))
        f = np.asarray(f)

        return f

    def paretofront(self, exp_npoints):
        pass


class MaF3:
    def __init__(self, nobj = 2):
        dim = nobj + 9
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.pf = None

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = 100.0 * (k + sum([math.pow(x - 0.5, 2.0) - math.cos(20.0 * math.pi * (x - 0.5)) for x in solution[self.dim-k:]]))
        f = [1.0 + g]*self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj-i-1):
                f[i] *= math.cos(0.5 * math.pi * solution[j])
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi *solution[self.nobj-i-1])

            if i < self.nobj - 1:
                f[i] = math.pow(f[i], 4.0)
            else:
                f[i] = math.pow(f[i], 2.0)

        f = np.asarray(f)
        return f

    def paretofront(self, exp_npoints):
        pass


if __name__ == '__main__':
    mo_problem = DTLZ7(nobj = 2)
    print(mo_problem.eval(
        [0.727168, 0.940400, 0.400722, 0.651523, 0.491361, 0.436997, 0.009458, 0.447353, 0.389612, 0.354411, 0.164718,
         0.004920, 0.988386, 0.291270, 0.545449, 0.496552, 0.291778, 0.696243, 0.110069, 0.640723, 0.674940]))
    print(mo_problem.ub)
    print(mo_problem.ub.shape)
    print(mo_problem.pf)
