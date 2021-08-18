import math
import random
import operator
import functools
from platypus.core import Problem, Solution, EPSILON, Generator
from platypus.types import Real, Binary
from abc import ABCMeta
import numpy as np
import scipy.stats as stats


class GlobalProblem(Problem):

    def __init__(self, nvars, nobjs, fhat):
        super(GlobalProblem, self).__init__(nvars, nobjs)
        for i in range(nvars):
            self.types[i] = Real(0, 1)
        self.fhat = fhat

    def evaluate(self, solution):
        x = np.asarray(solution.variables[:])
        f = []
        for fhat in self.fhat:
            f.append(float(fhat.predict(x)))
        solution.objectives[:] = f


class GapProblem(Problem):

    def __init__(self, nvars, nobjs, fhat, xgap, rgap):
        super(GapProblem, self).__init__(nvars, nobjs)
        self.fhat = fhat
        self.set_bounds(xgap, nvars, rgap)

    def set_bounds(self, xgap, nvars, rgap):
        for i in range(nvars):
            minval = max(0, xgap[i] - rgap)
            maxval = min(1, xgap[i] + rgap)
            self.types[i] = Real(minval, maxval)

    def evaluate(self, solution):
        x = np.asarray(solution.variables[:])
        f = []
        for fhat in self.fhat:
            f.append(fhat.eval(x))
        solution.objectives[:] = f


class CustomGenerator(Generator):

    def __init__(self, popsize):
        super(CustomGenerator, self).__init__()
        self.popsize = popsize
        self.iter = 0
        self.solutions = []

    def create(self, problem, nd_solutions = None, prev_pop = None):
        N = M = 0
        if nd_solutions is not None: (N, l) = nd_solutions.shape
        if prev_pop is not None: (M, l) = prev_pop.shape

        print('N = {}, M = {}'.format(N, M))

        if N >= self.popsize:
            indices = np.random.choice(N, self.popsize, replace=False)
            for i in indices:
                solution = Solution(problem)
                solution.variables = list(nd_solutions[i,:])
                self.solutions.append(solution)
        else:
            for i in range(N):
                solution = Solution(problem)
                solution.variables = list(nd_solutions[i,:])
                self.solutions.append(solution)

            if M >= self.popsize - N:
                indices = np.random.choice(M, self.popsize - N, replace = False)
                for i in indices:
                    solution = Solution(problem)
                    solution.variables = list(prev_pop[i, :])
                    self.solutions.append(solution)
            else:
                for i in range(M):
                    solution = Solution(problem)
                    solution.variables = list(prev_pop[i, :])
                    self.solutions.append(solution)

                for i in range(N + M, self.popsize):
                    solution = Solution(problem)
                    solution.variables = [x.rand() for x in problem.types]
                    self.solutions.append(solution)

                '''
                solution = Solution(problem)
                new_solution = []
                for j in range(len(nd_solutions[(i - N) % N])):
                    if ddsprob > random.uniform(0, 1):
                        x = nd_solutions[(i - N) % N, j]
                        lower, upper = 0, 1
                        new_solution.append(float(stats.truncnorm.rvs(lower - x / 0.2, (upper - x) / 0.2, loc = x, scale = 0.2, size = 1)))
                    else:
                        new_solution.append(float(x = nd_solutions[(i - N) % N, j]))
                solution.variables = new_solution
                self.solutions.append(solution)
                '''

    def generate(self, problem):
        solution = self.solutions[self.iter]
        self.iter += 1
        return solution
