"""
.. module:: sot_sync_strategies
   :synopsis: Parallel synchronous optimization strategy

.. moduleauthor:: David Bindel <bindel@cornell.edu>,
                David Eriksson <dme65@cornell.edu>

:Module: sot_sync_strategies
:Author: David Bindel <bindel@cornell.edu>,
        David Eriksson <dme65@cornell.edu>

"""

from __future__ import print_function
import numpy as np
import math
import logging
from ..pySOT1.experimental_design import SymmetricLatinHypercube, LatinHypercube
from ..pySOT1.adaptive_sampling import CandidateDYCORS
from poap.strategy import BaseStrategy, RetryStrategy
from ..pySOT1.rbf import *
from ..pySOT1.utils import *
from ..pySOT1.rs_wrappers import *
import time
from .gpr import GaussianProcessRegressor
#import matplotlib.pyplot as plt
import os

# Get module-level logger
logger = logging.getLogger(__name__)


class SyncStrategyNoConstraints(BaseStrategy):
    """Parallel synchronous optimization strategy without non-bound constraints.

    This class implements the parallel synchronous SRBF strategy
    described by Regis and Shoemaker.  After the initial experimental
    design (which is embarrassingly parallel), the optimization
    proceeds in phases.  During each phase, we allow nsamples
    simultaneous function evaluations.  We insist that these
    evaluations run to completion -- if one fails for whatever reason,
    we will resubmit it.  Samples are drawn randomly from around the
    current best point, and are sorted according to a merit function
    based on distance to other sample points and predicted function
    values according to the response surface.  After several
    successive significant improvements, we increase the sampling
    radius; after several failures to improve the function value, we
    decrease the sampling radius.  We restart once the sampling radius
    decreases below a threshold.

    :param worker_id: Start ID in a multi-start setting
    :type worker_id: int
    :param data: Problem parameter data structure
    :type data: Object
    :param response_surface: Surrogate model object
    :type response_surface: Object
    :param maxeval: Stopping criterion. If positive, this is an 
                    evaluation budget. If negative, this is a time
                    budget in seconds.
    :type maxeval: int
    :param nsamples: Number of simultaneous fevals allowed
    :type nsamples: int
    :param exp_design: Experimental design
    :type exp_design: Object
    :param sampling_method: Sampling method for finding
        points to evaluate
    :type sampling_method: Object
    :param extra: Points to be added to the experimental design
    :type extra: numpy.array
    :param extra_vals: Values of the points in extra (if known). Use nan for values that are not known.
    :type extra_vals: numpy.array
    """

    def __init__(self, worker_id, data, response_surface, maxeval, nsamples,
                 exp_design=None, sampling_method=None, extra=None, extra_vals=None, evaluated=None):

        # Check stopping criterion
        self.start_time = time.time()
        self.process_start_time = self.start_time
        if maxeval < 0:  # Time budget
            self.maxeval = np.inf
            self.time_budget = np.abs(maxeval)
        else:
            self.maxeval = maxeval
            self.time_budget = np.inf

        # Import problem information
        self.worker_id = worker_id
        self.data = data
        self.fhat = response_surface
        if self.fhat is None:
            self.fhat = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval)
        self.fhat.reset()  # Just to be sure!

        self.nsamples = nsamples
        self.extra = extra
        self.extra_vals = extra_vals
        self.eval_progress = 1
        self.iteration_cost = np.zeros(10)
        self.evaluated = evaluated
        self.trigger = True



        # Default to generate sampling points using Symmetric Latin Hypercube
        self.design = exp_design
        if self.design is None:
            if self.data.dim > 50:
                self.design = LatinHypercube(data.dim, data.dim+1)
            else:
                if self.data.dim==1:
                    s_num=2
                else:
                    s_num=2*(data.dim+1)
                self.design = SymmetricLatinHypercube(data.dim, s_num)

        self.xrange = np.asarray(data.xup - data.xlow)

        # algorithm parameters
        self.sigma_min = 0.005
        self.sigma_max = 0.2
        self.sigma_init = 0.2

        self.failtol = max(5, data.dim)
        self.succtol = 3

        self.numeval = 0
        self.status = 0
        self.sigma = 0
        self.resubmitter = RetryStrategy()
        self.xbest = None
        self.fbest = np.inf
        self.fbest_old = None
        self.MSE = []
        self.MAE = []
        self.R = []
        self.samples = np.zeros([maxeval,self.data.dim])
        self.Tmac = 0 # Computational time for model accuracy calculation
        self.subtime = 0 #record the computational time of model accuracy calculation in every phase
        self.eval_points = None  # self.eval points are used to record the evaluation points



        # Set up search procedures and initialize
        self.sampling = sampling_method
        if self.sampling is None:
            self.sampling = CandidateDYCORS(data)

        self.check_input()

        # Start with first experimental design
        self.sample_initial()

        # check if it already has the accuracy samples
        # if self.accuracy_samples is None:
        #     self.accuracy_samples = np.random.uniform(data.xlow, data.xup, (50*data.dim, data.dim))
        #     self.accuracy_fit = np.zeros(50*data.dim)
        #     for k in range(50*data.dim):
        #         self.accuracy_fit[k] = data.objfunction(self.accuracy_samples[k,:])

    def check_input(self):
        """Checks that the inputs are correct"""

        self.check_common()
        if hasattr(self.data, "eval_ineq_constraints"):
            raise ValueError("Optimization problem has constraints,\n"
                             "SyncStrategyNoConstraints can't handle constraints")
        if hasattr(self.data, "eval_eq_constraints"):
            raise ValueError("Optimization problem has constraints,\n"
                             "SyncStrategyNoConstraints can't handle constraints")

    def check_common(self):
        """Checks that the inputs are correct"""

        # Check evaluation budget
        if self.extra is None:
            if self.maxeval < self.design.npts:
                self.design.npts = self.maxeval - 1
                #raise ValueError("Experimental design is larger than the evaluation budget")
        else:
            # Check the number of unknown extra points
            if self.extra_vals is None:  # All extra point are unknown
                nextra = self.extra.shape[0]
            else:  # We know the values at some extra points so count how many we don't know
                nextra = np.sum(np.isinf(self.extra_vals)) + np.sum(np.isnan(self.extra_vals))

            if self.maxeval < self.design.npts + nextra:
                raise ValueError("Experimental design + extra points "
                                 "exceeds the evaluation budget")

        # Check dimensionality
        if self.design.dim != self.data.dim:
            raise ValueError("Experimental design and optimization "
                             "problem have different dimensions")
        if self.extra is not None:
            if self.data.dim != self.extra.shape[1]:
                raise ValueError("Extra point and optimization problem "
                                 "have different dimensions")
            if self.extra_vals is not None:
                if self.extra.shape[0] != len(self.extra_vals):
                    raise ValueError("Extra point values has the wrong length")

        # Check that the optimization problem makes sense
        check_opt_prob(self.data)

    def proj_fun(self, x):
        """Projects a set of points onto the feasible region

        :param x: Points, of size npts x dim
        :type x: numpy.array
        :return: Projected points
        :rtype: numpy.array
        """

        x = np.atleast_2d(x)
        return round_vars(self.data, x)

    def log_completion(self, record):
        """Record a completed evaluation to the log.

        :param record: Record of the function evaluation
        :type record: Object
        """

        xstr = np.array_str(record.params[0], max_line_width=np.inf,
                            precision=5, suppress_small=True)
        if record.feasible:
            logger.info("{} {:.3e} @ {}".format("True", record.value, xstr))
        else:
            logger.info("{} {:.3e} @ {}".format("False", record.value, xstr))

    def adjust_step(self):
        """Adjust the sampling radius sigma.

        After succtol successful steps, we cut the sampling radius;
        after failtol failed steps, we double the sampling radius.
        """

        # Initialize if this is the first adaptive step
        if self.fbest_old is None:
            self.fbest_old = self.fbest
            return

        # Check if we succeeded at significant improvement
        if self.fbest < self.fbest_old - 1e-3 * math.fabs(self.fbest_old):
            self.status = max(1, self.status + 1)
        else:
            self.status = min(-1, self.status - 1)
        self.fbest_old = self.fbest

        # Check if step needs adjusting
        if self.status <= -self.failtol:
            self.status = 0
            self.sigma /= 2
            logger.info("Reducing sigma")
        if self.status >= self.succtol:
            self.status = 0
            self.sigma = min([2.0 * self.sigma, self.sigma_max])
            logger.info("Increasing sigma")

    def sample_initial(self):
        """Generate and queue an initial experimental design."""

        if self.numeval == 0:
            logger.info("=== Start ===")
        else:
            logger.info("=== Restart ===")
            print("=== Restart ===")
        self.fhat.reset()
        self.sigma = self.sigma_init
        self.status = 0
        self.xbest = None
        self.fbest_old = None
        self.fbest = np.inf
        self.fhat.reset()



        if len(self.evaluated) > 0 and self.trigger:
            print('===Reading from history data===')
            self.trigger = False
            a = len(self.evaluated)

            if a < self.design.npts:
                start_sample = np.zeros([self.design.npts, self.data.dim])
                for k, item in enumerate(self.evaluated):
                    start_sample[k, :] = np.asarray(item["point"])
                experiment_design = LatinHypercube(self.data.dim, self.design.npts - a)
                extra_point=experiment_design.generate_points()
                assert extra_point.shape[1] == self.data.dim, \
                    "Dimension mismatch between problem and experimental design"
                extra_point = from_unit_box(extra_point, self.data)
                for h in range(a, self.design.npts):
                    start_sample[h, :] = extra_point[h-a, :]
            else:
                tmp = min(4 * (self.data.dim + 1), a, self.maxeval-1)
                start_sample = np.zeros([tmp, self.data.dim])
                assert start_sample.shape[1] == self.data.dim
                for k in range(tmp):
                    start_sample[k, :] = np.asarray(self.evaluated[k]["point"])

            for j in range(min(start_sample.shape[0], self.maxeval - self.numeval)):
                start_sample[j, :] = self.proj_fun(start_sample[j, :])  # Project onto feasible region
                proposal = self.propose_eval(np.copy(start_sample[j, :]))
                self.resubmitter.rput(proposal)

            self.sampling.init(start_sample, self.fhat, self.maxeval - self.numeval)
            self.eval_points = start_sample


        else:
            start_sample = self.design.generate_points()
            assert start_sample.shape[1] == self.data.dim, \
                "Dimension mismatch between problem and experimental design"
            start_sample = from_unit_box(start_sample, self.data)

            if self.extra is not None:
                # We know the values if this is a restart, so add the points to the surrogate
                if self.numeval > 0:
                    for i in range(len(self.extra_vals)):
                        xx = self.proj_fun(np.copy(self.extra[i, :]))
                        self.fhat.add_point(np.ravel(xx), self.extra_vals[i])
                else:  # Check if we know the values of the points
                    if self.extra_vals is None:
                        self.extra_vals = np.nan * np.ones((self.extra.shape[0], 1))

                    for i in range(len(self.extra_vals)):
                        xx = self.proj_fun(np.copy(self.extra[i, :]))
                        if np.isnan(self.extra_vals[i]) or np.isinf(self.extra_vals[i]):  # We don't know this value
                            proposal = self.propose_eval(np.ravel(xx))
                            proposal.extra_point_id = i  # Decorate the proposal
                            self.resubmitter.rput(proposal)
                        else:  # We know this value
                            self.fhat.add_point(np.ravel(xx), self.extra_vals[i])

            # Evaluate the experimental design
            for j in range(min(start_sample.shape[0], self.maxeval - self.numeval)):
                start_sample[j, :] = self.proj_fun(start_sample[j, :])  # Project onto feasible region
                proposal = self.propose_eval(np.copy(start_sample[j, :]))
                self.resubmitter.rput(proposal)

            if self.extra is not None:
                self.sampling.init(np.vstack((start_sample, self.extra)), self.fhat, self.maxeval - self.numeval)
            else:
                self.sampling.init(start_sample, self.fhat, self.maxeval - self.numeval)

            self.eval_points = start_sample




    def sample_adapt(self):
        """Generate and queue samples from the search strategy"""

        self.adjust_step()
        nsamples = min(self.nsamples, self.maxeval - self.numeval)
        if self.data.dim == 1:
            new_points, self.xcan = self.sampling.make_points(npts=nsamples, xbest=np.copy(self.xbest), sigma=self.sigma,
                                                   proj_fun=self.proj_fun)
        else:
            new_points = self.sampling.make_points(npts=nsamples, xbest=np.copy(self.xbest), sigma=self.sigma,
                                               proj_fun=self.proj_fun)

        if self.data.dim == 1:
            if self.numeval >= self.design.npts:
                print('current num=', str(self.numeval), 'design number=', str(self.design.npts))
                self.draw_process(new_points)

        for i in range(nsamples):
            proposal = self.propose_eval(np.copy(np.ravel(new_points[i, :])))
            self.resubmitter.rput(proposal)

        self.eval_points = np.copy(new_points) # update current eva points

    def start_batch(self):
        """Generate and queue a new batch of points"""
        #original method restart enable
        # if self.sigma < self.sigma_min:
        #     self.sample_initial()
        # else:
        #     self.sample_adapt()
        # revised method, restart disabled
        if self.numeval == 0 or self.sigma < self.sigma_min:
            self.sample_initial()
        else:
            self.sample_adapt()

    def propose_action(self):
        """Propose an action"""

        current_time = time.time()
        if self.numeval >= self.maxeval or (current_time - self.start_time) >= self.time_budget:
            end_time = time.time()
            self.iteration_cost[self.eval_progress-1] = (end_time-self.process_start_time) - self.subtime
            self.samples=self.fhat.get_x()
            #print(self.samples)
            return self.propose_terminate()
        elif self.resubmitter.num_eval_outstanding == 0:
            self.start_batch()
        elif self.numeval >= 0.1*self.eval_progress*self.maxeval and self.eval_progress < 10:
            end_time = time.time()
            self.iteration_cost[self.eval_progress-1] = (end_time-self.process_start_time)- self.subtime
            self.process_start_time = time.time()
            self.subtime = 0
            self.eval_progress += 1

        return self.resubmitter.get()

    # def accuracy_calculation(self):
    #
    #     f_hat = np.zeros([self.sample_size,1])
    #     error_sum = 0
    #     mean_sum = 0
    #     f_mean = 0
    #     STD = 0
    #     accuracy_samples=self.tp
    #     accuracy_fit=self.ty
    #
    #     sample_num = accuracy_samples.shape[0]
    #     tmp = self.sampling.fhat.evals(accuracy_samples)
    #     for j in range(sample_num):
    #         f_hat[j] = tmp[j,0]
    #     f_mean = np.sum(accuracy_fit)/sample_num
    #
    #     error_sum = np.sum(item**2 for item in (accuracy_fit-f_hat))
    #     mean_sum = np.sum(item**2 for item in (accuracy_fit-f_mean*np.ones([sample_num,1])))
    #     #print('mean sum =',mean_sum)
    #     STD = np.sqrt(mean_sum/(sample_num-1))
    #     MSE = np.sqrt(error_sum/sample_num)/STD
    #
    #
    #     #local accuracy around evaluation points
    #     eva_num = self.eval_points.shape[0]
    #     box_r = 0.05*(self.data.xup-self.data.xlow)
    #     eva_MSE = np.zeros(eva_num)
    #     for i in range(eva_num):
    #         low = np.maximum.reduce([self.eval_points[i,:]-box_r,self.data.xlow])
    #         up = np.minimum.reduce([self.eval_points[i,:]+box_r,self.data.xup])
    #         eva_local = np.random.uniform(low,up,(self.sample_size, self.data.dim))
    #         local_fhat = self.sampling.fhat.evals(eva_local)
    #         local_real = np.zeros([eva_local.shape[0],1])
    #         for k in range(eva_local.shape[0]):
    #             local_real[k] = self.data.objfunction(eva_local[k,:])
    #         #print('diff =',local_real-local_fhat)
    #
    #         error_sum = np.sum(item**2 for item in (local_real-local_fhat))
    #         #print('error_sum =',error_sum)
    #         f_mean = np.sum(local_real)/eva_local.shape[0]
    #
    #         mean_sum = np.sum(item**2 for item in (local_real-f_mean*np.ones([eva_local.shape[0],1])))
    #         STD = np.sqrt(mean_sum/(eva_local.shape[0]-1))
    #         eva_MSE[i] = np.sqrt(error_sum/eva_local.shape[0])/STD
    #
    #     #accuracy around the current best region
    #     box_r = 0.05*(self.data.xup-self.data.xlow)
    #     low = np.maximum.reduce([self.xbest-box_r,self.data.xlow])
    #     up = np.minimum.reduce([self.xbest+box_r,self.data.xup])
    #     eva_gb = np.random.uniform(low,up,(self.sample_size, self.data.dim))
    #     gb_fhat = self.sampling.fhat.evals(eva_gb)
    #     gb_real = np.zeros([eva_gb.shape[0],1])
    #     for k in range(eva_gb.shape[0]):
    #         gb_real[k] = self.data.objfunction(eva_gb[k,:])
    #
    #     error_sum = np.sum(item**2 for item in (gb_real-gb_fhat))
    #     f_mean = np.sum(gb_real)/eva_gb.shape[0]
    #
    #     mean_sum = np.sum(item**2 for item in (gb_real-f_mean*np.ones([eva_gb.shape[0],1])))
    #     STD = np.sqrt(mean_sum/(eva_gb.shape[0]-1))
    #     gb_MSE = np.sqrt(error_sum/eva_gb.shape[0])/STD
    #     self.R.append(gb_MSE)
    #
    #     self.MAE.append(np.mean(eva_MSE))
    #     self.MSE.append(MSE)
        # print("MSE = ",MSE)
        # print("eva_MSE = ",np.mean(eva_MSE))
        # print("gb_MSE = ",gb_MSE)

        # x = np.array([[ 1.49337981,  0.54288656, -0.4832504 ,  2.69711966,  0.34385379],
        # [-4.89933104,  1.5926705 ,  4.88103025, -1.40422214,  4.43621966]])
        # fx = self.sampling.fhat.evals(x)
        # print(type(fx))
        # f = np.zeros([2,1])
        # f[0] = self.data.objfunction(x[0,:])
        # f[1] = self.data.objfunction(x[1,:])
        # print(type(f))
        # print('outside fitness = ',fx-f)



    def on_reply_accept(self, proposal):
        # Transfer the decorations
        if hasattr(proposal, 'extra_point_id'):
            proposal.record.extra_point_id = proposal.extra_point_id

    def on_complete(self, record):
        """Handle completed function evaluation.

        When a function evaluation is completed we need to ask the constraint
        handler if the function value should be modified which is the case for
        say a penalty method. We also need to print the information to the
        logfile, update the best value found so far and notify the GUI that
        an evaluation has completed.

        :param record: Evaluation record
        :type record: Object
        """

        # Check for extra_point decorator
        if hasattr(record, 'extra_point_id'):
            self.extra_vals[record.extra_point_id] = record.value

        self.numeval += 1
        record.worker_id = self.worker_id
        record.worker_numeval = self.numeval
        record.feasible = True
        self.log_completion(record)
        self.fhat.add_point(np.copy(record.params[0]), record.value)
        #print('points=', str(record.params[0]), 'value=', str(record.value))
        if record.value < self.fbest:
            self.xbest = np.copy(record.params[0])
            self.fbest = record.value

        # when experiment complete calculate the model accuracy


        if self.design.npts < 50:
            if np.mod(self.numeval, 200) == 0:
                print('current number of evaluation = ', self.numeval)
                if self.numeval > self.design.npts:
                    t1 = time.time()
                    #self.accuracy_calculation()
                    t2 = time.time()
                    self.Tmac = self.Tmac + (t2- t1)
                    self.subtime = self.subtime + (t2- t1)
        else:
            if self.numeval == self.design.npts:
                assert self.data.dim <= 40
                t1 = time.time()
                #self.accuracy_calculation()
                t2 = time.time()
                self.Tmac = self.Tmac + (t2- t1)
                self.subtime = self.subtime + (t2- t1)
            if np.mod(self.numeval, 200) == 0:
                print('current number of evaluation = ', self.numeval)
                if self.numeval > self.design.npts:
                    t1 = time.time()
                    #self.accuracy_calculation()
                    t2 = time.time()
                    self.Tmac = self.Tmac + (t2- t1)
                    self.subtime = self.subtime + (t2- t1)



class SyncStrategyPenalty(SyncStrategyNoConstraints):
    """Parallel synchronous optimization strategy with non-bound constraints.

    This is an extension of SyncStrategyNoConstraints that also works with
    bound constraints. We currently only allow inequality constraints, since
    the candidate based methods don't work well with equality constraints.
    We also assume that the constraints are cheap to evaluate, i.e., so that
    it is easy to check if a given point is feasible. More strategies that
    can handle expensive constraints will be added.

    We use a penalty method in the sense that we try to minimize:

    .. math::
        f(x) + \\mu \\sum_j (\\max(0, g_j(x))^2

    where :math:`g_j(x) \\leq 0` are cheap inequality constraints. As a
    measure of promising function values we let all infeasible points have
    the value of the feasible candidate point with the worst function value,
    since large penalties makes it impossible to distinguish between feasible
    points.

    When it comes to the value of :math:`\\mu`, just choose a very large value.

    :param worker_id: Start ID in a multi-start setting
    :type worker_id: int
    :param data: Problem parameter data structure
    :type data: Object
    :param response_surface: Surrogate model object
    :type response_surface: Object
    :param maxeval: Function evaluation budget
    :type maxeval: int
    :param nsamples: Number of simultaneous fevals allowed
    :type nsamples: int
    :param exp_design: Experimental design
    :type exp_design: Object
    :param sampling_method: Sampling method for finding
        points to evaluate
    :type sampling_method: Object
    :param extra: Points to be added to the experimental design
    :type extra: numpy.array
    :param penalty: Penalty for violating constraints
    :type penalty: float
    """

    def __init__(self, worker_id, data, response_surface, maxeval, nsamples,
                 exp_design=None, sampling_method=None, extra=None,
                 penalty=1e6):

        # Evals wrapper for penalty method
        def penalty_evals(fhat, xx):
            penalty = self.penalty_fun(xx).T
            vals = fhat.evals(xx)
            if xx.shape[0] > 1:
                ind = (np.where(penalty <= 0.0)[0]).T
                if ind.shape[0] > 1:
                    ind2 = (np.where(penalty > 0.0)[0]).T
                    ind3 = np.argmax(np.squeeze(vals[ind]))
                    vals[ind2] = vals[ind3]
                    return vals
            return vals + penalty

        # Derivs wrapper for penalty method
        def penalty_derivs(fhat, xx):
            x = np.atleast_2d(xx)
            constraints = np.array(self.data.eval_ineq_constraints(x))
            dconstraints = self.data.deriv_ineq_constraints(x)
            constraints[np.where(constraints < 0.0)] = 0.0
            return np.atleast_2d(fhat.deriv(xx)) + \
                2 * self.penalty * np.sum(
                    constraints * np.rollaxis(dconstraints, 2), axis=2).T

        SyncStrategyNoConstraints.__init__(self,  worker_id, data,
                                           RSPenalty(response_surface, penalty_evals, penalty_derivs),
                                           maxeval, nsamples, exp_design,
                                           sampling_method, extra)
        self.penalty = penalty

    def check_input(self):
        """Checks that the inputs are correct"""

        self.check_common()
        if not hasattr(self.data, "eval_ineq_constraints"):
            raise AttributeError("Optimization problem has no inequality constraints")
        if hasattr(self.data, "eval_eq_constraints"):
            raise AttributeError("Optimization problem has equality constraints,\n"
                                 "SyncStrategyPenalty can't handle equality constraints")

    def penalty_fun(self, xx):
        """Computes the penalty for constraint violation

        :param xx: Points to compute the penalty for
        :type xx: numpy.array
        :return: Penalty for constraint violations
        :rtype: numpy.array
        """

        vec = np.array(self.data.eval_ineq_constraints(xx))
        vec[np.where(vec < 0.0)] = 0.0
        vec **= 2
        return self.penalty * np.asmatrix(np.sum(vec, axis=1))

    def on_complete(self, record):
        """Handle completed function evaluation.

        When a function evaluation is completed we need to ask the constraint
        handler if the function value should be modified which is the case for
        say a penalty method. We also need to print the information to the
        logfile, update the best value found so far and notify the GUI that
        an evaluation has completed.

        :param record: Evaluation record
        :type record: Object
        """

        # Check for extra_point decorator
        if hasattr(record, 'extra_point_id'):
            self.extra_vals[record.extra_point_id] = record.value

        x = np.zeros((1, record.params[0].shape[0]))
        x[0, :] = np.copy(record.params[0])
        penalty = self.penalty_fun(x)[0, 0]
        if penalty > 0.0:
            record.feasible = False
        else:
            record.feasible = True
        self.log_completion(record)
        self.numeval += 1
        record.worker_id = self.worker_id
        record.worker_numeval = self.numeval
        self.fhat.add_point(np.copy(record.params[0]), record.value)
        # Check if the penalty function is a new best
        if record.value + penalty < self.fbest:
            self.xbest = np.copy(record.params[0])
            self.fbest = record.value + penalty


class SyncStrategyProjection(SyncStrategyNoConstraints):
    """Parallel synchronous optimization strategy with non-bound constraints.
    It uses a supplied method to project proposed points onto the feasible
    region in order to always evaluate feasible points which is useful in
    situations where it is easy to project onto the feasible region and where
    the objective function is nonsensical for infeasible points.

    This is an extension of SyncStrategyNoConstraints that also works with
    bound constraints.

    :param worker_id: Start ID in a multi-start setting
    :type worker_id: int
    :param data: Problem parameter data structure
    :type data: Object
    :param response_surface: Surrogate model object
    :type response_surface: Object
    :param maxeval: Function evaluation budget
    :type maxeval: int
    :param nsamples: Number of simultaneous fevals allowed
    :type nsamples: int
    :param exp_design: Experimental design
    :type exp_design: Object
    :param sampling_method: Sampling method for finding
        points to evaluate
    :type sampling_method: Object
    :param extra: Points to be added to the experimental design
    :type extra: numpy.array
    :param proj_fun: Function that projects one point onto the feasible region
    :type proj_fun: Object
    """

    def __init__(self, worker_id, data, response_surface, maxeval, nsamples,
                 exp_design=None, sampling_method=None, extra=None,
                 proj_fun=None):

        self.projection = proj_fun
        SyncStrategyNoConstraints.__init__(self,  worker_id, data,
                                           response_surface, maxeval,
                                           nsamples, exp_design,
                                           sampling_method, extra)

    def check_input(self):
        """Checks that the inputs are correct"""

        self.check_common()
        if not (hasattr(self.data, "eval_ineq_constraints") or
                hasattr(self.data, "eval_eq_constraints")):
            raise AttributeError("Optimization problem has no constraints")

    def proj_fun(self, x):
        """Projects a set of points onto the feasible region

        :param x: Points, of size npts x dim
        :type x: numpy.array
        :return: Projected points
        :rtype: numpy.array
        """

        x = np.atleast_2d(x)
        for i in range(x.shape[0]):
            x[i, :] = self.projection(x[i, :])
        return x
