"""
.. module:: adaptive_sampling
   :synopsis: Ways of finding the next point to evaluate in the adaptive phase

.. moduleauthor:: David Eriksson <dme65@cornell.edu>,
                David Bindel <bindel@cornell.edu>

:Module: adaptive_sampling
:Author: David Eriksson <dme65@cornell.edu>,
        David Bindel <bindel@cornell.edu>

"""

import math
from ..pySOT1.utils import *
import scipy.spatial as scp
from scipy.stats import norm
from .heuristic_methods import GeneticAlgorithm as GA
from scipy.optimize import minimize
import scipy.stats as stats
from ..pySOT1.merit_functions import *
from .modified_merit_functions import *
#import modified_gp_regression
import types
from .de import DiffEvolOptimizer
from numpy.random import RandomState
from .push_to_boundary import push_to_boundary
def __fix_docs(cls):
    """Help function for stealing docs from the parent"""
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
    return cls


class MultiSampling(object):
    """Maintains a list of adaptive sampling methods

    A collection of adaptive sampling methods and weights so that the user
    can use multiple adaptive sampling methods for the same optimization
    problem. This object keeps an internal list of proposed points
    in order to be able to compute the minimum distance from a point
    to all proposed evaluations. This list has to be reset each time
    the optimization algorithm restarts

    :param strategy_list: List of adaptive sampling methods to use
    :type strategy_list: list
    :param cycle: List of integers that specifies the sampling order, e.g., [0, 0, 1] uses
        method1, method1, method2, method1, method1, method2, ...
    :type cycle: list
    :raise ValueError: If cycle is incorrect

    :ivar sampling_strategies: List of adaptive sampling methods to use
    :ivar cycle: List that specifies the sampling order
    :ivar nstrats: Number of adaptive sampling strategies
    :ivar current_strat: The next adaptive sampling strategy to be used
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar budget: Remaining evaluation budget

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def __init__(self, strategy_list, cycle):
        if cycle is None:
            cycle = range(len(strategy_list))
        if (not all(isinstance(i, int) for i in cycle)) or \
                np.min(cycle) < 0 or \
                np.max(cycle) > len(strategy_list)-1:
            raise ValueError("Incorrect cycle!!")
        self.sampling_strategies = strategy_list
        self.nstrats = len(strategy_list)
        self.cycle = cycle
        self.current_strat= 0
        self.proposed_points = None
        self.data = strategy_list[0].data
        self.fhat = None
        self.budget = None
        self.n0 = None

    def init(self, start_sample, fhat, budget):
        """Initialize the sampling method after the initial phase

        This initializes the list of sampling methods after the initial phase
        has finished and the experimental design has been evaluated. The user
        provides the points in the experimental design, the surrogate model,
        and the remaining evaluation budget.

        :param start_sample: Points in the experimental design
        :type start_sample: numpy.array
        :param fhat: Surrogate model
        :type fhat: Object
        :param budget: Evaluation budget
        :type budget: int
        """

        self.proposed_points = start_sample
        self.fhat = fhat
        self.n0 = start_sample.shape[0]
        for i in range(self.nstrats):
            self.sampling_strategies[i].init(self.proposed_points, fhat, budget)

    def remove_point(self, x):
        """Remove x from proposed_points

        This removes x from the list of proposed points in the case where the optimization
        strategy decides to not evaluate x.

        :param x: Point to be removed
        :type x: numpy.array
        :return: True if points was removed, False otherwise
        :type: bool
        """

        idx = np.sum(np.abs(self.proposed_points - x), axis=1).argmin()
        if np.sum(np.abs(self.proposed_points[idx, :] - x)) < 1e-10:
            self.proposed_points = np.delete(self.proposed_points, idx, axis=0)
            for i in range(self.nstrats):
                self.sampling_strategies[i].remove_point(x)
            return True
        return False

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_merit_weighted_distance):
        """Proposes npts new points to evaluate

        :param npts: Number of points to select
        :type npts: int
        :param xbest: Best solution found so far
        :type xbest: numpy.array
        :param sigma: Current sampling radius w.r.t the unit box
        :type sigma: float
        :param subset: Coordinates to perturb
        :type subset: numpy.array
        :param proj_fun: Routine for projecting infeasible points onto the feasible region
        :type proj_fun: Object
        :param merit: Merit function for selecting candidate points
        :type merit: Object

        :return: Points selected for evaluation, of size npts x dim
        :rtype: numpy.array

        .. todo:: Change the merit function from being hard-coded
        """

        new_points = np.zeros((npts, self.data.dim))

        # Figure out what we need to generate
        npoints = np.zeros((self.nstrats,), dtype=int)
        for i in range(npts):
            npoints[self.cycle[self.current_strat]] += 1
            self.current_strat = (self.current_strat + 1) % len(self.cycle)

        # Now generate the points from one strategy at the time
        count = 0
        for i in range(self.nstrats):
            if npoints[i] > 0:
                new_points[count:count+npoints[i], :] = \
                    self.sampling_strategies[i].make_points(npts=npoints[i], xbest=xbest,
                                                            sigma=sigma, subset=subset,
                                                            proj_fun=proj_fun,
                                                            merit=merit)

                count += npoints[i]
                # Update list of proposed points
                for j in range(self.nstrats):
                    if j != i:
                        self.sampling_strategies[j].proposed_points = \
                            self.sampling_strategies[i].proposed_points

        return new_points


class CandidateSRBF(object):
    """An implementation of Stochastic RBF

    This is an implementation of the candidate points method that is
    proposed in the first SRBF paper. Candidate points are generated
    by making normally distributed perturbations with standard
    deviation sigma around the best solution. The candidate point that
    minimizes a specified merit function is selected as the next
    point to evaluate.

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def __init__(self, data, numcand=None, weights=None):
        self.data = data
        self.fhat = None
        self.xrange = self.data.xup - self.data.xlow
        self.dtol = 1e-3 * math.sqrt(data.dim)
        self.weights = weights
        if self.weights is None:
            self.weights = [0.2, 0.4, 0.6, 0.8]
        self.proposed_points = None
        self.dmerit = None
        self.xcand = None
        self.fhvals = None
        self.shvals = None
        self.fbest = None
        self.next_weight = 0
        self.numcand = numcand
        if self.numcand is None:
            self.numcand = min([5000, 100*data.dim])
        self.budget = None

        # Check that the inputs make sense
        if not(isinstance(self.numcand, int) and self.numcand > 0):
            raise ValueError("The number of candidate points has to be a positive integer")
        if not((isinstance(self.weights, np.ndarray) or isinstance(self.weights, list))
               and max(self.weights) <= 1 and min(self.weights) >= 0):
            raise ValueError("Incorrect weights")

    def init(self, start_sample, fhat, budget):
        """Initialize the sampling method after the initial phase

        This initializes the list of sampling methods after the initial phase
        has finished and the experimental design has been evaluated. The user
        provides the points in the experimental design, the surrogate model,
        and the remaining evaluation budget.

        :param start_sample: Points in the experimental design
        :type start_sample: numpy.array
        :param fhat: Surrogate model
        :type fhat: Object
        :param budget: Evaluation budget
        :type budget: int
        """

        self.proposed_points = start_sample
        self.budget = budget
        self.fhat = fhat
        if self.data.search_type == 'exploration':
            self.weights = [0.1, 0.3, 0.5, 0.7, 0.9]
        elif self.data.search_type == 'exploitation':
            self.weights = [0.1, 0.3, 0.5, 0.7, 0.9]


    def remove_point(self, x):
        """Remove x from proposed_points

        This removes x from the list of proposed points in the case where the optimization
        strategy decides to not evaluate x.

        :param x: Point to be removed
        :type x: numpy.array
        :return: True if points was removed, False otherwise
        :type: bool
        """

        idx = np.sum(np.abs(self.proposed_points - x), axis=1).argmin()
        if np.sum(np.abs(self.proposed_points[idx, :] - x)) < 1e-10:
            self.proposed_points = np.delete(self.proposed_points, idx, axis=0)
            return True
        return False

    def __generate_cand__(self, scalefactors, xbest, subset):
        self.xbest = xbest
        self.xcand = np.ones((self.numcand,  self.data.dim)) * xbest
        for i in subset:
            lower, upper = self.data.xlow[i], self.data.xup[i]
            ssigma = scalefactors[i]
            self.xcand[:, i] = stats.truncnorm.rvs(
                (lower - xbest[i]) / ssigma, (upper - xbest[i]) / ssigma,
                loc=xbest[i], scale=ssigma, size=self.numcand)

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_merit_weighted_distance):
        """Proposes npts new points to evaluate

        :param npts: Number of points to select
        :type npts: int
        :param xbest: Best solution found so far
        :type xbest: numpy.array
        :param sigma: Current sampling radius w.r.t the unit box
        :type sigma: float
        :param subset: Coordinates to perturb, the others are fixed
        :type subset: numpy.array
        :param proj_fun: Routine for projecting infeasible points onto the feasible region
        :type proj_fun: Object
        :param merit: Merit function for selecting candidate points
        :type merit: Object

        :return: Points selected for evaluation, of size npts x dim
        :rtype: numpy.array

        .. todo:: Change the merit function from being hard-coded
        """

        if subset is None:
            subset = np.arange(0, self.data.dim)
        scalefactors = sigma * self.xrange

        # Make sure that the scale factors are correct for
        # the integer variables (at least 1)
        ind = np.intersect1d(self.data.integer, subset)
        if len(ind) > 0:
            scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

        # Generate candidate points
        self.__generate_cand__(scalefactors, xbest, subset)
        if proj_fun is not None:
            self.xcand = proj_fun(self.xcand)

        dists = scp.distance.cdist(self.xcand, self.proposed_points)
        fhvals = self.fhat.evals(self.xcand)
        #print('fhvals =',fhvals)
        #print('self.xcand =',self.xcand)
        # freal = np.zeros([self.xcand.shape[0]])
        # for k in range(self.xcand.shape[0]):
        #     freal[k] = self.data.objfunction(np.copy(self.xcand[k,:]))
        # #print('freal =',freal)
        # adaptive_error = 0
        # for k in range(self.xcand.shape[0]):
        #     adaptive_error += np.sum((fhvals[k]-freal[k])*(fhvals[k]-freal[k]))
        #print('error =',adaptive_error)
        # x = np.array([[ 1.49337981,  0.54288656, -0.4832504 ,  2.69711966,  0.34385379],
        # [-4.89933104,  1.5926705 ,  4.88103025, -1.40422214,  4.43621966]])
        # fx = self.fhat.evals(x)
        # f = np.zeros([2,1])
        # f[0] = self.data.objfunction(x[0,:])
        # f[1] = self.data.objfunction(x[1,:])
        # print('inner fitness difference = ',fx-f)
        # fhvals2 = self.make_prediction(self.xcand)
        #
        # adaptive_difference = 0
        # for k in range(self.xcand.shape[0]):
        #     adaptive_difference += np.sum((fhvals[k]-fhvals2[k])*(fhvals[k]-fhvals2[k]))
        # print('adaptive differene =',adaptive_difference)
        self.dmerit = np.amin(np.asmatrix(dists), axis=1)
        self.fhvals = unit_rescale(fhvals)

        xnew = merit(self, npts)
        self.proposed_points = np.vstack((self.proposed_points,
                                          np.asmatrix(xnew)))
        return xnew

    def make_prediction(self, eval_points, proj_fun=None):
        if proj_fun is not None:
            eval_points = proj_fun(eval_points)

        fhvals = self.fhat.evals(eval_points)
        return fhvals

@__fix_docs

class Candidate_EI(CandidateSRBF):
    """Candidate points are generated by perturbing ONLY the discrete
    variables using the SRBF strategy
    
    """
#    def __generate_cand__(self, scalefactors, xbest, subset):
#        self.xbest = xbest
#        #self.xcand = np.ones((self.numcand, self.data.dim)) * xbest
#       # self.xcand[:, subset] = np.random.uniform(
#            #self.data.xlow[subset], self.data.xup[subset], (self.numcand, len(subset)))
#        self.xcand = np.random.uniform(
#            self.data.xlow, self.data.xup, (self.numcand, self.data.dim))
    
    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_EI):
                        
        if subset is None:
            subset = np.arange(0, self.data.dim)
        scalefactors = sigma * self.xrange

        # Make sure that the scale factors are correct for
        # the integer variables (at least 1)
        ind = np.intersect1d(self.data.integer, subset)
        if len(ind) > 0:
            scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

        # Generate candidate points
        self.__generate_cand__(scalefactors, xbest, subset)
        if proj_fun is not None:
            self.xcand = proj_fun(self.xcand)


#        fhvals = self.fhat.eval(self.xcand)
#        shvals = self.fhat.deriv(self.xcand)
        
        fhvals , shvals= self.fhat.eval(self.xcand)
        fbest, sbest = self.fhat.eval(self.xbest.reshape(1,-1))
        
#        fbest = self.fhat.eval(xbest.reshape(1,-1))

        self.fhvals = fhvals
        self.shvals = shvals
        self.fbest = fbest

        #self.fhvals = unit_rescale(fhvals)
        #self.fbest = unit_rescale(fbest)

        xnew = merit(self, npts)
        
        self.proposed_points = np.vstack((self.proposed_points,
                                          np.asmatrix(xnew)))
        return xnew

class CandidateUniform(CandidateSRBF):
    """Create Candidate points by sampling uniformly in the domain

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def init(self, start_sample, fhat, budget):
        CandidateSRBF.init(self, start_sample, fhat, budget)

    def remove_point(self, x):
        return CandidateSRBF.remove_point(self, x)

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_merit_weighted_distance):
        return CandidateSRBF.make_points(self, npts, xbest, sigma, subset, proj_fun, merit)

    def __generate_cand__(self, scalefactors, xbest, subset):
        self.xcand = np.ones((self.numcand, self.data.dim)) * xbest
        self.xcand[:, subset] = np.random.uniform(
            self.data.xlow[subset], self.data.xup[subset], (self.numcand, len(subset)))


@__fix_docs
class CandidateDYCORS(CandidateSRBF):
    """An implementation of the DYCORS method

    The DYCORS method only perturbs a subset of the dimensions when
    perturbing the best solution. The probability for a dimension
    to be perturbed decreases after each evaluation and is capped
    in order to guarantee global convergence.

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar minprob: Smallest allowed perturbation probability
    :ivar n0: Evaluations spent when the initial phase ended
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def __init__(self, data, numcand=None, weights=None):
        CandidateSRBF.__init__(self, data, numcand=numcand, weights=weights)
        self.minprob = np.min([1.0, 1.0/self.data.dim])
        self.n0 = None

        if data.dim < 1:
            raise ValueError("You can't use DYCORS on a 1d problem")

        def probfun(numevals, budget):
            if budget < 2:
                return 0
            return min([20.0/data.dim, 1.0]) * (1.0 - (np.log(numevals + 1.0) / np.log(budget)))
        self.probfun = probfun

    def init(self, start_sample, fhat, budget):
        CandidateSRBF.init(self, start_sample, fhat, budget)
        self.n0 = start_sample.shape[0]

    def remove_point(self, x):
        return CandidateSRBF.remove_point(self, x)

    def make_prediction(self, eval_points, proj_fun=None):
        return CandidateSRBF.make_prediction(self, eval_points, proj_fun=None)

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_merit_weighted_distance):

        if self.data.dim ==1:
            return CandidateSRBF.make_points(self, npts, xbest, sigma, subset, proj_fun, merit),self.xcand
        else:
            return CandidateSRBF.make_points(self, npts, xbest, sigma, subset, proj_fun, merit)

    def __generate_cand__(self, scalefactors, xbest, subset):
        ddsprob = self.probfun(self.proposed_points.shape[0] - self.n0, self.budget - self.n0)
        ddsprob = np.max([self.minprob, ddsprob])

        nlen = len(subset)

        # Fix when nlen is 1
        # Todo: Use SRBF instead
        if nlen == 1:
            ar = np.ones((self.numcand, 1))
        else:
            ar = (np.random.rand(self.numcand, nlen) < ddsprob)
            ind = np.where(np.sum(ar, axis=1) == 0)[0]
            ar[ind, np.random.randint(0, nlen - 1, size=len(ind))] = 1

        self.xcand = np.ones((self.numcand, self.data.dim)) * xbest
        for i in range(nlen):
            lower, upper = self.data.xlow[i], self.data.xup[i]
            ssigma = scalefactors[subset[i]]
            ind = np.where(ar[:, i] == 1)[0]
            self.xcand[ind, subset[i]] = stats.truncnorm.rvs(
                (lower - xbest[subset[i]]) / ssigma, (upper - xbest[subset[i]]) / ssigma,
                loc=xbest[subset[i]], scale=ssigma, size=len(ind))


class CandidateDYCORS_boundary(CandidateSRBF):
    """An implementation of the enhanced DYCORS method, another set of points gerated around boundaries.


    The DYCORS method only perturbs a subset of the dimensions when
    perturbing the best solution. The probability for a dimension
    to be perturbed decreases after each evaluation and is capped
    in order to guarantee global convergence.

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar minprob: Smallest allowed perturbation probability
    :ivar n0: Evaluations spent when the initial phase ended
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def __init__(self, data, numcand=None, weights=None):
        CandidateSRBF.__init__(self, data, numcand=numcand, weights=weights)
        self.minprob = np.min([1.0, 1.0/self.data.dim])
        self.n0 = None

        if data.dim < 1:
            raise ValueError("You can't use DYCORS on a 1d problem")

        def probfun(numevals, budget):
            if budget < 2:
                return 0
            return min([20.0/data.dim, 1.0]) * (1.0 - (np.log(numevals + 1.0) / np.log(budget)))
        self.probfun = probfun

    def init(self, start_sample, fhat, budget):
        CandidateSRBF.init(self, start_sample, fhat, budget)
        self.n0 = start_sample.shape[0]

    def remove_point(self, x):
        return CandidateSRBF.remove_point(self, x)

    def make_prediction(self, eval_points, proj_fun=None):
        return CandidateSRBF.make_prediction(self, eval_points, proj_fun=None)

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_merit_weighted_distance):

        if self.data.dim ==1:
            return CandidateSRBF.make_points(self, npts, xbest, sigma, subset, proj_fun, merit),self.xcand
        else:
            return CandidateSRBF.make_points(self, npts, xbest, sigma, subset, proj_fun, merit)

    def __generate_cand__(self, scalefactors, xbest, subset):
        ddsprob = self.probfun(self.proposed_points.shape[0] - self.n0, self.budget - self.n0)
        ddsprob = np.max([self.minprob, ddsprob])

        nlen = len(subset)

        # Fix when nlen is 1
        # Todo: Use SRBF instead
        if nlen == 1:
            ar = np.ones((self.numcand, 1))
        else:
            ar = (np.random.rand(self.numcand, nlen) < ddsprob)
            ind = np.where(np.sum(ar, axis=1) == 0)[0]
            ar[ind, np.random.randint(0, nlen - 1, size=len(ind))] = 1

        xcand = np.ones((self.numcand, self.data.dim)) * xbest
        for i in range(nlen):
            lower, upper = self.data.xlow[i], self.data.xup[i]
            ssigma = scalefactors[subset[i]]
            ind = np.where(ar[:, i] == 1)[0]
            xcand[ind, subset[i]] = stats.truncnorm.rvs(
                (lower - xbest[subset[i]]) / ssigma, (upper - xbest[subset[i]]) / ssigma,
                loc=xbest[subset[i]], scale=ssigma, size=len(ind))

        #generate another set of candidate points

        grid_size=1000
        bp = push_to_boundary(self, xbest, grid_size)
        # grid = sobol_grid.generate(self.data.dim, grid_size=grid_size,
        #                                 grid_seed=0)
        # grid = np.matlib.repmat(self.data.xlow, grid_size, 1)+np.multiply(grid,np.matlib.repmat((self.data.xup-self.data.xlow),grid_size, 1))

        # for k in range(grid_size):
        #     for h in range(self.data.dim):
        #         grid[k,h]=self.data.xlow[h]+grid[k,h]*(self.data.xup[h]-self.data.xlow[h])
        self.xcand=np.concatenate((xcand,bp),axis=0)


class CandidateGradient(CandidateSRBF):
    """An implementation of the DYCORS-gradient method

    The DYCORS method only perturbs a subset of the dimensions when
    perturbing the best solution. The probability for a dimension
    to be perturbed decreases after each evaluation and is capped
    in order to guarantee global convergence.

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar minprob: Smallest allowed perturbation probability
    :ivar n0: Evaluations spent when the initial phase ended
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def __init__(self, data, numcand=None, weights=None):
        CandidateSRBF.__init__(self, data, numcand=numcand, weights=weights)
        self.minprob = np.min([1.0, 1.0/self.data.dim])
        self.n0 = None

        if data.dim <= 1:
            raise ValueError("You can't use DYCORS on a 1d problem")

        def probfun(numevals, budget):
            if budget < 2:
                return 0
            return min([20.0/data.dim, 1.0]) * (1.0 - (np.log(numevals + 1.0) / np.log(budget)))
        self.probfun = probfun

    def init(self, start_sample, fhat, budget):
        CandidateSRBF.init(self, start_sample, fhat, budget)
        self.n0 = start_sample.shape[0]

    def remove_point(self, x):
        return CandidateSRBF.remove_point(self, x)

    def make_prediction(self, eval_points, proj_fun=None):
        return CandidateSRBF.make_prediction(self, eval_points, proj_fun=None)

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_top_points):
        def eval(x):
            return self.fhat.eval(x).ravel()

        def deriv(x):
            return self.fhat.deriv(x).ravel()

        xtop = CandidateSRBF.make_points(self, npts, xbest, sigma, subset, proj_fun, merit)
        top_num = xtop.shape[0]
        ytop = np.inf * np.ones([top_num, 1])
        bounds = np.array([self.data.xlow, self.data.xup]).T
        ybest = np.inf
        for h in range(top_num):
            res = minimize(eval, xtop[h, :].reshape(1, -1), method="L-BFGS-B",
                           jac=deriv, bounds=bounds)
            if ybest > res.fun:
                ybest = res.fun
                xnew = res.x

        xnew = xnew.reshape(1, self.data.dim)
        self.proposed_points = np.vstack((self.proposed_points,
                                          np.asmatrix(xnew)))
        return xnew

    def __generate_cand__(self, scalefactors, xbest, subset):
        ddsprob = self.probfun(self.proposed_points.shape[0] - self.n0, self.budget - self.n0)
        ddsprob = np.max([self.minprob, ddsprob])

        nlen = len(subset)

        # Fix when nlen is 1
        # Todo: Use SRBF instead
        if nlen == 1:
            ar = np.ones((self.numcand, 1))
        else:
            ar = (np.random.rand(self.numcand, nlen) < ddsprob)
            ind = np.where(np.sum(ar, axis=1) == 0)[0]
            ar[ind, np.random.randint(0, nlen - 1, size=len(ind))] = 1

        self.xcand = np.ones((self.numcand, self.data.dim)) * xbest
        for i in range(nlen):
            lower, upper = self.data.xlow[i], self.data.xup[i]
            ssigma = scalefactors[subset[i]]
            ind = np.where(ar[:, i] == 1)[0]
            self.xcand[ind, subset[i]] = stats.truncnorm.rvs(
                (lower - xbest[subset[i]]) / ssigma, (upper - xbest[subset[i]]) / ssigma,
                loc=xbest[subset[i]], scale=ssigma, size=len(ind))


class Candidate_multiobjective(CandidateSRBF):
    """An implementation of the DYCORS-multiobjective method

    The DYCORS method only perturbs a subset of the dimensions when
    perturbing the best solution. The probability for a dimension
    to be perturbed decreases after each evaluation and is capped
    in order to guarantee global convergence.

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar minprob: Smallest allowed perturbation probability
    :ivar n0: Evaluations spent when the initial phase ended
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def __init__(self, data, numcand=None, weights=None):
        CandidateSRBF.__init__(self, data, numcand=numcand, weights=weights)
        self.minprob = np.min([1.0, 1.0/self.data.dim])
        self.n0 = None

        if data.dim <= 1:
            raise ValueError("You can't use DYCORS on a 1d problem")

        def probfun(numevals, budget):
            if budget < 2:
                return 0
            return min([20.0/data.dim, 1.0]) * (1.0 - (np.log(numevals + 1.0) / np.log(budget)))
        self.probfun = probfun

    def init(self, start_sample, fhat, budget):
        CandidateSRBF.init(self, start_sample, fhat, budget)
        self.n0 = start_sample.shape[0]

    def remove_point(self, x):
        return CandidateSRBF.remove_point(self, x)

    def make_prediction(self, eval_points, proj_fun=None):
        return CandidateSRBF.make_prediction(self, eval_points, proj_fun=None)

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_top_points):


        # def eval(x):
        #     return self.fhat.eval(x).ravel()
        #
        # def deriv(x):
        #     return self.fhat.deriv(x).ravel()
        #
        # xtop = CandidateSRBF.make_points(self, npts, xbest, sigma, subset, proj_fun, merit)
        # top_num = xtop.shape[0]
        # ytop = np.inf * np.ones([top_num, 1])
        # bounds = np.array([self.data.xlow, self.data.xup]).T
        # ybest = np.inf
        # for h in range(top_num):
        #     res = minimize(eval, xtop[h, :].reshape(1, -1), method="L-BFGS-B",
        #                    jac=deriv, bounds=bounds)
        #     if ybest > res.fun:
        #         ybest = res.fun
        #         xnew = res.x
        # do multi-objective optimization using NSGA-2
        #from metrics.problems.zdt import ZDT3Metrics
        from nsga2.evolution import Evolution
        from PDT import PDT
        from prediction_distance_definitions import prediction_distanceDefinitions

        zdt_definitions = prediction_distanceDefinitions(self)

        problem = PDT(zdt_definitions,self.data.xlow,self.data.xup)
        evolution = Evolution(problem, 100, 100)
        pareto_front = evolution.evolve()


        xnew = xnew.reshape(1, self.data.dim)
        self.proposed_points = np.vstack((self.proposed_points,
                                          np.asmatrix(xnew)))
        return xnew

    def __generate_cand__(self, scalefactors, xbest, subset):
        ddsprob = self.probfun(self.proposed_points.shape[0] - self.n0, self.budget - self.n0)
        ddsprob = np.max([self.minprob, ddsprob])

        nlen = len(subset)

        # Fix when nlen is 1
        # Todo: Use SRBF instead
        if nlen == 1:
            ar = np.ones((self.numcand, 1))
        else:
            ar = (np.random.rand(self.numcand, nlen) < ddsprob)
            ind = np.where(np.sum(ar, axis=1) == 0)[0]
            ar[ind, np.random.randint(0, nlen - 1, size=len(ind))] = 1

        self.xcand = np.ones((self.numcand, self.data.dim)) * xbest
        for i in range(nlen):
            lower, upper = self.data.xlow[i], self.data.xup[i]
            ssigma = scalefactors[subset[i]]
            ind = np.where(ar[:, i] == 1)[0]
            self.xcand[ind, subset[i]] = stats.truncnorm.rvs(
                (lower - xbest[subset[i]]) / ssigma, (upper - xbest[subset[i]]) / ssigma,
                loc=xbest[subset[i]], scale=ssigma, size=len(ind))




class CandidateSpearmint(CandidateSRBF):
    """An implementation of the spearmint way to select the next evaluation point



    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar minprob: Smallest allowed perturbation probability
    :ivar n0: Evaluations spent when the initial phase ended
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def __init__(self, data, numcand=None, weights=None):
        CandidateSRBF.__init__(self, data, numcand=numcand, weights=weights)


    def init(self, start_sample, fhat, budget):
        CandidateSRBF.init(self, start_sample, fhat, budget)
        self.n0 = start_sample.shape[0]

    def remove_point(self, x):
        return CandidateSRBF.remove_point(self, x)

    def make_prediction(self, eval_points, proj_fun=None):
        return CandidateSRBF.make_prediction(self, eval_points, proj_fun=None)

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_spearmint):


        def eval(x):
            return self.fhat.eval(x).ravel()

        def deriv(x):
            return self.fhat.deriv(x).ravel()



        # Generate candidate points
        self.__generate_cand__()
        if proj_fun is not None:
            self.xcand = proj_fun(self.xcand)

        dists = scp.distance.cdist(self.xcand, self.proposed_points)
        fhvals = self.fhat.evals(self.xcand)


        self.dmerit = np.amin(np.asmatrix(dists), axis=1)
        self.fhvals = unit_rescale(fhvals)
        top_num = 20
        xtop = merit(self, npts,top_num)

        ytop = np.inf*np.ones([top_num,1])
        bounds = np.array([self.data.xlow,self.data.xup]).T
        ybest = np.inf
        for h in range(top_num):
            res = minimize(eval, xtop[h,:].reshape(1,-1), method="L-BFGS-B",
                           jac=deriv, bounds=bounds)
            if ybest > res.fun:
               ybest=res.fun
               xnew=res.x

        xnew=xnew.reshape(1,self.data.dim)
        self.proposed_points = np.vstack((self.proposed_points,
                                          np.asmatrix(xnew)))
        return xnew

    def __generate_cand__(self):
        self.xcand = np.random.RandomState(0).uniform(self.data.xlow,self.data.xup,[20000,self.data.dim])








@__fix_docs
class CandidateDDS(CandidateDYCORS):
    """An implementation of the DDS candidate points method

    Only a few candidate points are generated
    and the candidate point with the lowest value predicted
    by the surrogate model is selected. The DDS method only
    perturbs a subset of the dimensions when perturbing the
    best solution. The probability for a dimension to be
    perturbed decreases after each evaluation and is capped
    in order to guarantee global convergence.

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def __init__(self, data, numcand=None, weights=None):
        CandidateDYCORS.__init__(self, data, numcand=numcand, weights=weights)
        self.weights = np.array([1.0])
        self.numcand = max([0.5*data.dim, 2])

        def probfun(numevals, budget):
            return 1.0-(np.log(numevals + 1.0) / np.log(budget))
        self.probfun = probfun

    def init(self, start_sample, fhat, budget):
        CandidateDYCORS.init(self, start_sample, fhat, budget)
        self.n0 = start_sample.shape[0]

    def remove_point(self, x):
        return CandidateDYCORS.remove_point(self, x)

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_merit_weighted_distance):

        new_points = np.zeros((npts, self.data.dim))
        for i in range(npts):
            new_points[i, :] = CandidateDYCORS.make_points(self, npts=1, xbest=xbest, sigma=0.2,
                                                           subset=subset, proj_fun=proj_fun)
        return new_points


@__fix_docs
class CandidateSRBF_INT(CandidateSRBF):
    """CandidateSRBF where only the the integer variables are perturbed

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def init(self, start_sample, fhat, budget):
        CandidateSRBF.init(self, start_sample, fhat, budget)

    def remove_point(self, x):
        return CandidateSRBF.remove_point(self, x)

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_merit_weighted_distance):
        if len(self.data.integer) > 0:
            return CandidateSRBF.make_points(self, npts=npts, xbest=xbest, sigma=sigma,
                                             subset=self.data.integer, proj_fun=proj_fun,
                                             merit=merit)
        else:
            return CandidateSRBF.make_points(self, npts=npts, xbest=xbest, sigma=sigma,
                                             subset=self.data.continuous, proj_fun=proj_fun,
                                             merit=merit)


@__fix_docs
class CandidateDYCORS_INT(CandidateDYCORS):
    """CandidateDYCORS where only the the integer variables are perturbed

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def init(self, start_sample, fhat, budget):
        CandidateDYCORS.init(self, start_sample, fhat, budget)

    def remove_point(self, x):
        return CandidateDYCORS.remove_point(self, x)

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_merit_weighted_distance):
        if len(self.data.integer) > 0:
            return CandidateDYCORS.make_points(self, npts=npts, xbest=xbest, sigma=sigma,
                                               subset=self.data.integer, proj_fun=proj_fun,
                                               merit=merit)
        else:
            return CandidateDYCORS.make_points(self, npts=npts, xbest=xbest, sigma=sigma,
                                               subset=self.data.continuous, proj_fun=proj_fun,
                                               merit=merit)


@__fix_docs
class CandidateDDS_INT(CandidateDDS):
    """CandidateDDS where only the the integer variables are perturbed

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def init(self, start_sample, fhat, budget):
        CandidateDDS.init(self, start_sample, fhat, budget)

    def remove_point(self, x):
        return CandidateDDS.remove_point(self, x)

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_merit_weighted_distance):
        if len(self.data.integer) > 0:
            return CandidateDDS.make_points(self, npts=npts, xbest=xbest, sigma=sigma,
                                            subset=self.data.integer, proj_fun=proj_fun,
                                            merit=merit)
        else:
            return CandidateDDS.make_points(self, npts=npts, xbest=xbest, sigma=sigma,
                                            subset=self.data.continuous, proj_fun=proj_fun,
                                            merit=merit)


@__fix_docs
class CandidateUniform_INT(CandidateUniform):
    """CandidateUniform where only the the integer variables are perturbed

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def init(self, start_sample, fhat, budget):
        CandidateUniform.init(self, start_sample, fhat, budget)

    def remove_point(self, x):
        return CandidateUniform.remove_point(self, x)

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_merit_weighted_distance):
        if len(self.data.integer) > 0:
            return CandidateUniform.make_points(self, npts=npts, xbest=xbest, sigma=sigma,
                                                subset=self.data.integer, proj_fun=proj_fun,
                                                merit=merit)
        else:
            return CandidateUniform.make_points(self, npts=npts, xbest=xbest, sigma=sigma,
                                                subset=self.data.continuous, proj_fun=proj_fun,
                                                merit=merit)


@__fix_docs
class CandidateSRBF_CONT(CandidateSRBF):
    """CandidateSRBF where only the the continuous variables are perturbed

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def init(self, start_sample, fhat, budget):
        CandidateSRBF.init(self, start_sample, fhat, budget)

    def remove_point(self, x):
        return CandidateSRBF.remove_point(self, x)

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_merit_weighted_distance):
        if len(self.data.continuous) > 0:
            return CandidateSRBF.make_points(self, npts=npts, xbest=xbest, sigma=sigma,
                                             subset=self.data.continuous, proj_fun=proj_fun,
                                             merit=merit)
        else:
            return CandidateSRBF.make_points(self, npts=npts, xbest=xbest, sigma=sigma,
                                             subset=self.data.integer, proj_fun=proj_fun,
                                             merit=merit)


@__fix_docs
class CandidateDYCORS_CONT(CandidateDYCORS):
    """CandidateDYCORS where only the the continuous variables are perturbed

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def init(self, start_sample, fhat, budget):
        CandidateDYCORS.init(self, start_sample, fhat, budget)

    def remove_point(self, x):
        return CandidateDYCORS.remove_point(self, x)

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_merit_weighted_distance):
        if len(self.data.continuous) > 0:
            return CandidateDYCORS.make_points(self, npts=npts, xbest=xbest, sigma=sigma,
                                               subset=self.data.continuous, proj_fun=proj_fun,
                                               merit=merit)
        else:
            return CandidateDYCORS.make_points(self, npts=npts, xbest=xbest, sigma=sigma,
                                               subset=self.data.integer, proj_fun=proj_fun,
                                               merit=merit)


@__fix_docs
class CandidateDDS_CONT(CandidateDDS):
    """CandidateDDS where only the the continuous variables are perturbed

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def init(self, start_sample, fhat, budget):
        CandidateDDS.init(self, start_sample, fhat, budget)

    def remove_point(self, x):
        return CandidateDDS.remove_point(self, x)

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_merit_weighted_distance):
        if len(self.data.continuous) > 0:
            return CandidateDDS.make_points(self, npts=npts, xbest=xbest, sigma=sigma,
                                            subset=self.data.continuous, proj_fun=proj_fun,
                                            merit=merit)
        else:
            return CandidateDDS.make_points(self, npts=npts, xbest=xbest, sigma=sigma,
                                            subset=self.data.integer, proj_fun=proj_fun,
                                            merit=merit)


@__fix_docs
class CandidateUniform_CONT(CandidateUniform):
    """CandidateUniform where only the the continuous variables are perturbed

    :param data: Optimization problem object
    :type data: Object
    :param numcand: Number of candidate points to be used. Default is min([5000, 100*data.dim])
    :type numcand: int
    :param weights: Weights used for the merit function, to balance exploration vs exploitation
    :type weights: list of numpy.array

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar weights: Weights used for the merit function
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar dmerit: Minimum distance between the points and the proposed points
    :ivar xcand: Candidate points
    :ivar fhvals: Predicted values by the surrogate model
    :ivar next_weight: Index of the next weight to be used
    :ivar numcand: Number of candidate points
    :ivar budget: Remaining evaluation budget
    :ivar probfun: Function that computes the perturbation probability of a given iteration

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. todo:: Get rid of the proposed_points object and replace it by something that is
        controlled by the strategy.
    """

    def init(self, start_sample, fhat, budget):
        CandidateUniform.init(self, start_sample, fhat, budget)

    def remove_point(self, x):
        return CandidateUniform.remove_point(self, x)

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None,
                    merit=candidate_merit_weighted_distance):
        if len(self.data.continuous) > 0:
            return CandidateUniform.make_points(self, npts=npts, xbest=xbest, sigma=sigma,
                                                subset=self.data.continuous, proj_fun=proj_fun,
                                                merit=merit)
        else:
            return CandidateUniform.make_points(self, npts=npts, xbest=xbest, sigma=sigma,
                                                subset=self.data.integer, proj_fun=proj_fun,
                                                merit=merit)


class GeneticAlgorithm(object):
    """Genetic algorithm for minimizing the surrogate model

    :param data: Optimization problem object
    :type data: Object

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar budget: Remaining evaluation budget

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.
    """

    def __init__(self, data):
        self.data = data
        self.fhat = None
        self.dtol = 1e-3 * math.sqrt(data.dim)
        self.proposed_points = None
        self.budget = None

    def init(self, start_sample, fhat, budget):
        """Initialize the sampling method after the initial phase

        This initializes the list of sampling methods after the initial phase
        has finished and the experimental design has been evaluated. The user
        provides the points in the experimental design, the surrogate model,
        and the remaining evaluation budget.

        :param start_sample: Points in the experimental design
        :type start_sample: numpy.array
        :param fhat: Surrogate model
        :type fhat: Object
        :param budget: Evaluation budget
        :type budget: int
        """

        self.proposed_points = start_sample
        self.budget = budget
        self.fhat = fhat

    def remove_point(self, x):
        """Remove x from proposed_points

        This removes x from the list of proposed points in the case where the optimization
        strategy decides to not evaluate x.

        :param x: Point to be removed
        :type x: numpy.array
        :return: True if points was removed, False otherwise
        :type: bool
        """

        idx = np.sum(np.abs(self.proposed_points - x), axis=1).argmin()
        if np.sum(np.abs(self.proposed_points[idx, :] - x)) < 1e-10:
            self.proposed_points = np.delete(self.proposed_points, idx, axis=0)
            return True
        return False

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None, merit=None):
        """Proposes npts new points to evaluate

        :param npts: Number of points to select
        :type npts: int
        :param xbest: Best solution found so far (Ignored)
        :type xbest: numpy.array
        :param sigma: Current sampling radius w.r.t the unit box (Ignored)
        :type sigma: float
        :param subset: Coordinates to perturb (Ignored)
        :type subset: numpy.array
        :param proj_fun: Routine for projecting infeasible points onto the feasible region
        :type proj_fun: Object
        :param merit: Merit function for selecting candidate points (Ignored)
        :type merit: Object

        :return: Points selected for evaluation, of size npts x dim
        :rtype: numpy.array
        """

        new_points = np.zeros((npts, self.data.dim))
        for i in range(npts):
            ga = GA(self.fhat.evals, self.data.dim, self.data.xlow, self.data.xup,
                    popsize=max([2*self.data.dim, 100]), ngen=100, projfun=proj_fun)
            x_min, f_min = ga.optimize()

            dist = np.atleast_2d(np.min(
                scp.distance.cdist(self.proposed_points, np.atleast_2d(x_min)), axis=1)).T

            x_new = x_min
            if np.min(dist) < self.dtol:
                # Perturb the best solution until we satisfy the tolerance
                d = 0.0
                x_new = None
                while d < self.dtol:
                    x_new = x_min + self.dtol * np.random.randn(1, self.data.dim)
                    x_new = np.maximum(x_new, self.data.xlow)
                    x_new = np.minimum(x_new, self.data.xup)
                    d = np.atleast_2d(np.min(
                        scp.distance.cdist(self.proposed_points, np.atleast_2d(x_new)), axis=1)).T.min()

            new_points[i, :] = x_new
            self.proposed_points = np.vstack((self.proposed_points, np.asarray(x_new)))

        return new_points

class GeneticAlgorithm_EI(object):
    """Genetic algorithm for maximizing the EI criterion

    :param data: Optimization problem object
    :type data: Object

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar budget: Remaining evaluation budget

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.
    """

    def __init__(self, data):
        self.data = data
        self.fhat = None
        self.dtol = 1e-3 * math.sqrt(data.dim)
        self.proposed_points = None
        self.budget = None
        self.fbest = None

    def init(self, start_sample, fhat, budget):
        """Initialize the sampling method after the initial phase

        This initializes the list of sampling methods after the initial phase
        has finished and the experimental design has been evaluated. The user
        provides the points in the experimental design, the surrogate model,
        and the remaining evaluation budget.

        :param start_sample: Points in the experimental design
        :type start_sample: numpy.array
        :param fhat: Surrogate model
        :type fhat: Object
        :param budget: Evaluation budget
        :type budget: int
        """

        self.proposed_points = start_sample
        self.budget = budget
        self.fhat = fhat

    def remove_point(self, x):
        """Remove x from proposed_points

        This removes x from the list of proposed points in the case where the optimization
        strategy decides to not evaluate x.

        :param x: Point to be removed
        :type x: numpy.array
        :return: True if points was removed, False otherwise
        :type: bool
        """

        idx = np.sum(np.abs(self.proposed_points - x), axis=1).argmin()
        if np.sum(np.abs(self.proposed_points[idx, :] - x)) < 1e-10:
            self.proposed_points = np.delete(self.proposed_points, idx, axis=0)
            return True
        return False

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None, merit=None):
        """Proposes npts new points to evaluate

        :param npts: Number of points to select
        :type npts: int
        :param xbest: Best solution found so far (Ignored)
        :type xbest: numpy.array
        :param sigma: Current sampling radius w.r.t the unit box (Ignored)
        :type sigma: float
        :param subset: Coordinates to perturb (Ignored)
        :type subset: numpy.array
        :param proj_fun: Routine for projecting infeasible points onto the feasible region
        :type proj_fun: Object
        :param merit: Merit function for selecting candidate points (Ignored)
        :type merit: Object

        :return: Points selected for evaluation, of size npts x dim
        :rtype: numpy.array
        """
        self.xbest = xbest
        self.fhat.get_fbest(self.xbest.reshape(1,-1))
        new_points = np.zeros((npts, self.data.dim))



        # fhat = self.fhat.evals
        # shat = self.fhat.deriv
        # y_n = (y_min-self.fhat.evals)/np.sqrt(self.fhat.deriv)  #normalization
        # ei = (y_min-self.fhat.evals)*norm.cdf(y_n)+np.sqrt(self.fhat.deriv)*norm.pdf(y_n)
        for i in range(npts):
            ga = GA(self.fhat.eval_EI, self.data.dim, self.data.xlow, self.data.xup,
                    popsize=max([2*self.data.dim, 100]), ngen=10*self.data.dim, projfun=proj_fun, start="SLHD")
            x_min, f_min = ga.optimize() # GA's default setting is for minimization, min(-EI)

            new_points[i, :] = x_min
            self.proposed_points = np.vstack((self.proposed_points, np.asarray(x_min)))

        return new_points

class DifferentialEvolution_EI(object):
    """Differential Evolution algorithm for maximizing the EI criterion

    :param data: Optimization problem object
    :type data: Object

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar budget: Remaining evaluation budget

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.
    """

    def __init__(self, data):
        self.data = data
        self.fhat = None
        self.dtol = 1e-3 * math.sqrt(data.dim)
        self.proposed_points = None
        self.budget = None
        self.fbest = None

    def init(self, start_sample, fhat, budget):
        """Initialize the sampling method after the initial phase

        This initializes the list of sampling methods after the initial phase
        has finished and the experimental design has been evaluated. The user
        provides the points in the experimental design, the surrogate model,
        and the remaining evaluation budget.

        :param start_sample: Points in the experimental design
        :type start_sample: numpy.array
        :param fhat: Surrogate model
        :type fhat: Object
        :param budget: Evaluation budget
        :type budget: int
        """

        self.proposed_points = start_sample
        self.budget = budget
        self.fhat = fhat

    def remove_point(self, x):
        """Remove x from proposed_points

        This removes x from the list of proposed points in the case where the optimization
        strategy decides to not evaluate x.

        :param x: Point to be removed
        :type x: numpy.array
        :return: True if points was removed, False otherwise
        :type: bool
        """

        idx = np.sum(np.abs(self.proposed_points - x), axis=1).argmin()
        if np.sum(np.abs(self.proposed_points[idx, :] - x)) < 1e-10:
            self.proposed_points = np.delete(self.proposed_points, idx, axis=0)
            return True
        return False

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None, merit=None):
        """Proposes npts new points to evaluate

        :param npts: Number of points to select
        :type npts: int
        :param xbest: Best solution found so far (Ignored)
        :type xbest: numpy.array
        :param sigma: Current sampling radius w.r.t the unit box (Ignored)
        :type sigma: float
        :param subset: Coordinates to perturb (Ignored)
        :type subset: numpy.array
        :param proj_fun: Routine for projecting infeasible points onto the feasible region
        :type proj_fun: Object
        :param merit: Merit function for selecting candidate points (Ignored)
        :type merit: Object

        :return: Points selected for evaluation, of size npts x dim
        :rtype: numpy.array
        """
        self.xbest = xbest
        self.fhat.get_fbest(self.xbest.reshape(1,-1))
        new_points = np.zeros((npts, self.data.dim))

        bounds = []
        for i in range(self.data.dim):
            bounds.append([self.data.xlow[i],self.data.xup[i]])

        for i in range(npts):
            # ga = GA(self.fhat.eval_EI, self.data.dim, self.data.xlow, self.data.xup,
            #         popsize=max([2*self.data.dim, 100]), ngen=100, projfun=proj_fun)
            # x_min, f_min = ga.optimize() # GA's default setting is for minimization, min(-EI)
            ngen= 100
            npop = 100
            ndim = self.data.dim

            pop = np.zeros([ngen, npop, ndim])
            loc = np.zeros([ngen, ndim])
            de = DiffEvolOptimizer(self.fhat.eval_EI, bounds, npop=npop,ngen= ngen)

            [bestindividual, bestfit] = de.run()


            #print('x=',bestindividual,'y=',  bestfit)
            new_points[i, :] = bestindividual
            self.proposed_points = np.vstack((self.proposed_points, np.asarray(de.location)))

        return new_points


class MultiStartGradient(object):
    """A Multi-Start Gradient method for minimizing the surrogate model

    A wrapper around the scipy.optimize implementation of box-constrained
    gradient based minimization.

    :param data: Optimization problem object
    :type data: Object
    :param method: Optimization method to use. The options are

        - L-BFGS-B
            Quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)
        - TNC
            Truncated Newton algorithm

    :type method: string
    :param num_restarts: Number of restarts for the multi-start gradient
    :type num_restarts: int

    :raise ValueError: If number of candidate points is
        incorrect or if the weights aren't a list in [0, 1]

    :ivar data: Optimization problem object
    :ivar fhat: Response surface object
    :ivar xrange: Variable ranges, xup - xlow
    :ivar dtol: Smallest allowed distance between evaluated points 1e-3 * sqrt(dim)
    :ivar bounds: n x 2 matrix with lower and upper bound constraints
    :ivar proposed_points: List of points proposed to the optimization algorithm
    :ivar budget: Remaining evaluation budget

    .. note:: This object needs to be initialized with the init method. This is done when the
        initial phase has finished.

    .. Note:: SLSQP is supposed to work with bound constraints but for some reason it
        sometimes violates the constraints anyway.
    """

    def __init__(self, data, method='L-BFGS-B', num_restarts=30):
        self.data = data
        self.fhat = None
        self.bounds = np.zeros((self.data.dim, 2))
        self.bounds[:, 0] = self.data.xlow
        self.bounds[:, 1] = self.data.xup
        self.dtol = 1e-3 * math.sqrt(data.dim)
        self.proposed_points = None
        self.budget = None
        self.num_restarts = num_restarts
        if (method == 'TNC') or (method == 'L-BFGS-B'):
            self.method = method
        else:
            self.method = 'L-BFGS-B'

    def init(self, start_sample, fhat, budget):
        """Initialize the sampling method after the initial phase

        This initializes the list of sampling methods after the initial phase
        has finished and the experimental design has been evaluated. The user
        provides the points in the experimental design, the surrogate model,
        and the remaining evaluation budget.

        :param start_sample: Points in the experimental design
        :type start_sample: numpy.array
        :param fhat: Surrogate model
        :type fhat: Object
        :param budget: Evaluation budget
        :type budget: int
        """

        self.proposed_points = start_sample
        self.budget = budget
        self.fhat = fhat

    def remove_point(self, x):
        """Remove x from proposed_points

        This removes x from the list of proposed points in the case where the optimization
        strategy decides to not evaluate x.

        :param x: Point to be removed
        :type x: numpy.array
        :return: True if points was removed, False otherwise
        :type: bool
        """

        idx = np.sum(np.abs(self.proposed_points - x), axis=1).argmin()
        if np.sum(np.abs(self.proposed_points[idx, :] - x)) < 1e-10:
            self.proposed_points = np.delete(self.proposed_points, idx, axis=0)
            return True
        return False

    def make_points(self, npts, xbest, sigma, subset=None, proj_fun=None, merit=None):
        """Proposes npts new points to evaluate

        :param npts: Number of points to select
        :type npts: int
        :param xbest: Best solution found so far (Ignored)
        :type xbest: numpy.array
        :param sigma: Current sampling radius w.r.t the unit box (Ignored)
        :type sigma: float
        :param subset: Coordinates to perturb (Ignored)
        :type subset: numpy.array
        :param proj_fun: Routine for projecting infeasible points onto the feasible region
        :type proj_fun: Object
        :param merit: Merit function for selecting candidate points (Ignored)
        :type merit: Object

        :return: Points selected for evaluation, of size npts x dim
        :rtype: numpy.array
        """

        def eval(x):
            return self.fhat.eval(x).ravel()

        def deriv(x):
            return self.fhat.deriv(x).ravel()

        new_points = np.zeros((npts, self.data.dim))
        for j in range(npts):
            fvals = np.zeros(self.num_restarts)
            xvals = np.zeros((self.num_restarts, self.data.dim))
            dists = np.zeros(self.num_restarts)

            for i in range(self.num_restarts):
                if i == 0 and j == 0:
                    x0 = np.array(xbest)
                else:
                    x0 = np.random.uniform(self.data.xlow, self.data.xup)

                res = minimize(eval, x0, method=self.method,
                               jac=deriv, bounds=self.bounds)

                # Compute the distance to the proposed points
                xx = np.atleast_2d(res.x)
                if proj_fun is not None:
                    xx = proj_fun(xx)
                dist = np.atleast_2d(np.min(
                    scp.distance.cdist(self.proposed_points, xx), axis=1)).T

                fvals[i] = res.fun
                xvals[i, :] = xx
                dists[i] = dist.min()

            if dists.max() > self.dtol:
                x_new = None
                f_new = np.inf
                for i in range(self.num_restarts):
                    if dists[i] > self.dtol and fvals[i] < f_new:
                        x_new = xvals[i, :]
                        f_new = fvals[i]
            else:
                # Perturb the best point
                d = -1.0
                x_new = None
                while d < self.dtol:
                    x_new = xvals[np.argmin(fvals), :] + self.dtol * np.random.randn(1, self.data.dim)
                    x_new = np.maximum(x_new, self.data.xlow)
                    x_new = np.minimum(x_new, self.data.xup)
                    if proj_fun is not None:
                        x_new = proj_fun(x_new)
                    d = np.atleast_2d(np.min(
                        scp.distance.cdist(self.proposed_points, np.atleast_2d(x_new)), axis=1)).T.min()

            new_points[j, :] = x_new
            self.proposed_points = np.vstack((self.proposed_points, np.asarray(x_new)))

        return new_points
