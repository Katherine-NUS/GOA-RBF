import math
from nsga2 import seq
from nsga2.problems.problem_definitions import ProblemDefinitions
from modified_adaptive_sampling import Candidate_multiobjective
import scipy.spatial as scp
import numpy as np
class prediction_distanceDefinitions(ProblemDefinitions):

    def __init__(self,cand):
        super(prediction_distanceDefinitions, self).__init__()
        self.cand=cand
        self.n = self.cand.data.dim

    def f1(self, individual):

        return self.cand.fhat.eval(individual.features).ravel()

    def f2(self, individual):
        #print((np.asarray(individual.features).shape))
        dists = scp.distance.cdist(np.asarray(individual.features).reshape(1,self.n), self.cand.proposed_points)
        self.dmerit = np.amin(np.asmatrix(dists), axis=1)
        return -self.dmerit

    def perfect_pareto_front(self):
        step = 0.01
        domain = seq(0, 0.0830015349, step) \
                 + seq(0.1822287280, 0.2577623634, step) \
                 + seq(0.4093136748, 0.4538821041, step) \
                 + seq(0.6183967944, 0.6525117038, step) \
                 + seq(0.8233317983, 0.8518328654, step)
        return domain, map(lambda x1: 1 - math.sqrt(x1) - x1*math.sin(10*math.pi*x1), domain)
