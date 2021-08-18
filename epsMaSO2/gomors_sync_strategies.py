"""
.. module:: gomors_sync_strategies
   :synopsis: Parallel synchronous MO optimization strategy - GOMORS

.. moduleauthor:: David Bindel <bindel@cornell.edu>,
                David Eriksson <dme65@cornell.edu>,
                Taimoor Akhtar <erita@nus.edu.sg>

"""

from __future__ import print_function
import numpy as np
import math
import logging
from pySOT.experimental_design import SymmetricLatinHypercube, LatinHypercube
from poap.strategy import BaseStrategy, RetryStrategy
from pySOT.utils import *
import time
import random

from .gomors_adaptive_sampling import EvolutionaryAlgorithm
from copy import deepcopy
from .mo_utils import *
from .archiving_strategies import MemoryRecord, NonDominatedArchive, EpsilonArchive
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from .selection_rules import *
from diversipy.distance import *
import scipy.spatial as scp

# Get module-level logger
logger = logging.getLogger(__name__)
POSITIVE_INFINITY = float("inf")

class MoSyncStrategyNoConstraints(BaseStrategy):
    """Parallel Multi-Objective synchronous optimization strategy without non-bound constraints. (GOMORS)

    This class implements the GOMORS Framework
    described by Akhtar and Shoemaker (2016).  After the initial experimental
    design (which is embarrassingly parallel), the optimization
    proceeds in phases.  During each phase, we allow nsamples
    simultaneous function evaluations.  We insist that these
    evaluations run to completion -- if one fails for whatever reason,
    we will resubmit it.  Samples are drawn randomly from a multi-rule
    selection strategy that includes i) Global Evolutionary / Candidate
    search with three selection rules a) Hypervolume, b) Max-min Decision
    Space Distance and c) Max-min Objective Space Distance, and,
     ii) Neighborhood Evolutionary / Candidate Search with hv selection.

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
                 exp_design=None, sampling_method=None, archiving_method=None, extra=None, extra_vals=None):

        # Check stopping criterion
        self.start_time = time.time()
        if maxeval < 0:  # Time budget
            self.maxeval = np.inf
            self.time_budget = np.abs(maxeval)
        else:
            self.maxeval = maxeval
            self.time_budget = np.inf

        # Import problem information
        self.worker_id = worker_id
        self.data = data
        self.fhat = []

        for i in range(self.data.nobj):
            response_surface.reset()  # Just to be sure!
            self.fhat.append(deepcopy(response_surface)) #MOPLS ONLY

        self.ncenters = nsamples
        self.nsamples = 1
        self.numinit = None
        self.extra = extra
        self.extra_vals = extra_vals

        # Default to generate sampling points using Symmetric Latin Hypercube
        self.design = exp_design
        if self.design is None:
            if self.data.dim > 50:
                self.design = LatinHypercube(data.dim, data.dim+1)
            else:
                self.design = SymmetricLatinHypercube(data.dim, 2*(data.dim+1))

        self.xrange = np.asarray(data.ub - data.lb)

        # algorithm parameters
        self.sigma_min = 0.005
        self.sigma_max = 0.2
        self.sigma_init = 0.2

        self.failtol = min(max(5, data.dim), 15)
        self.failcount = 0
        self.contol = 5
        self.numeval = 0
        self.status = 0
        self.sigma = 0
        self.resubmitter = RetryStrategy()
        self.xbest = None
        self.fbest = None
        self.fbest_old = None
        self.improvement_prev = 1
        self.successive_fail = False

        # population of centers and long-term archive
        self.nd_archives = []
        self.new_pop = []
        self.supposed_fvals = None

        if archiving_method is None:
            self.memory_archive = NonDominatedArchive(200)
        else:
            self.memory_archive = archiving_method
        self.evals = []
        self.maxfit = min(500, 20*self.data.dim)
        self.d_thresh = 1.0

        # Set up search procedures and initialize
        self.sampling = sampling_method
        if self.sampling is None:
            self.sampling = EvolutionaryAlgorithm(data)

        # self.check_input()

        # Start with first experimental design
        self.sample_initial()

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
            if self.maxeval < self.design.num_pts:
                raise ValueError("Experimental design is larger than the evaluation budget")
        else:
            # Check the number of unknown extra points
            if self.extra_vals is None:  # All extra point are unknown
                nextra = self.extra.shape[0]
            else:  # We know the values at some extra points so count how many we don't know
                nextra = np.sum(np.isinf(self.extra_vals[0])) + np.sum(np.isnan(self.extra_vals[0]))

            if self.maxeval < self.design.num_pts + nextra:
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
        return round_vars(x, self.data.int_var, self.data.lb, self.data.ub)

    def log_completion(self, record):
        """Record a completed evaluation to the log.

        :param record: Record of the function evaluation
        :type record: Object
        """

        xstr = np.array_str(record.params[0], max_line_width=np.inf,
                            precision=5, suppress_small=True)
        fstr = np.array_str(record.value, max_line_width=np.inf,
                            precision=5, suppress_small=True)
        if record.feasible:
            logger.info("{} {} @ {}".format("True", fstr, xstr))
        else:
            logger.info("{} {} @ {}".format("False", fstr, xstr))


    def sample_initial(self):
        """Generate and queue an initial experimental design."""

        for fhat in self.fhat:
            fhat.reset() #MOPLS Only
        self.sigma = self.sigma_init
        self.failcount = 0
        self.xbest = None
        self.fbest_old = None
        self.fbest = None
        for fhat in self.fhat:
            fhat.reset() #MOPLS Only

        if self.numeval == 0:
            #start_sample = diversipy.maximin_reconstruction(num_points=2 * self.data.dim + 2, dimension=self.data.dim, dist_matrix_function=DistanceMatrixFunction(exponent=2,
            #                                          max_dists_per_dim=[1.0] * self.data.dim))
            start_sample = self.design.generate_points()
        else:
            start_sample = self.design.generate_points()


        assert start_sample.shape[1] == self.data.dim, \
            "Dimension mismatch between problem and experimental design"
        start_sample = from_unit_box(start_sample, self.data.lb, self.data.ub)

        if self.extra is not None:
            # We know the values if this is a restart, so add the points to the surrogate
            if self.numeval > 0:
                for i in range(len(self.extra_vals)):
                    xx = self.proj_fun(np.copy(self.extra[i, :]))
                    for j in range(self.data.nobj):
                        self.fhat[j].add_point(np.ravel(xx), self.extra_vals[i][j])
            else:  # Check if we know the values of the points
                if self.extra_vals is None:
                    self.extra_vals = np.nan * np.ones((self.extra.shape[0], self.data.nobj))

                for i in range(len(self.extra_vals)):
                    xx = self.proj_fun(np.copy(self.extra[i, :]))
                    if np.isnan(self.extra_vals[i][0]) or np.isinf(self.extra_vals[i][0]):  # We don't know this value
                        proposal = self.propose_eval(np.ravel(xx))
                        proposal.extra_point_id = i  # Decorate the proposal
                        self.resubmitter.rput(proposal)
                    else:  # We know this value
                        for j in range(self.data.nobj):
                            self.fhat[j].add_point(np.ravel(xx), self.extra_vals[i][j])

        # Evaluate the experimental design
        for j in range(min(start_sample.shape[0], self.maxeval - self.numeval)):
            start_sample[j, :] = self.proj_fun(start_sample[j, :])  # Project onto feasible region
            proposal = self.propose_eval(np.copy(start_sample[j, :]))
            self.resubmitter.rput(proposal)

        if self.extra is not None:
            sample_init = np.vstack((start_sample, self.extra))
        else:
            sample_init = start_sample

        sample_prev = np.copy(sample_init)

        if self.numeval == 0:
            logger.info("=== Start ===")
        elif self.status < self.contol:
            logger.info("=== Connected Start ===")
            print('Connected Restart # ' + str(self.status+1) + ' initiated')
            # Step 1 - Update connected restart count
            self.status += 1
            # Step 2 - Obtain xvals and fvals of ND points
            front = self.memory_archive.contents
            fvals = [rec.fx for rec in front]
            fvals = np.asarray(fvals)
            xvals = [rec.x for rec in front]
            xvals = np.asarray(xvals)
            # Step 3 - Add ND points to the surrogate
            npts, nobj = fvals.shape
            for i in range(npts):
                for j in range(nobj):
                    self.fhat[j].add_points(xvals[i,:], fvals[i, j])
            # Step 4 -  Add points to the set of previously evaluated points for sampling strategy
            all_xvals = [rec.x for rec in self.evals]
            sample_prev = np.vstack((sample_prev, all_xvals))
        else:
            # Step 4 - Store the Front in a separate archive
            front = self.memory_archive.contents
            self.nd_archives.append(front)
            self.status = 0
            if len(self.nd_archives) == 2:
                logger.info("=== Global Connected Restart ===")
                print('GLOBAL Restart Initiated')
                prev_front = self.nd_archives[0]
                for rec in prev_front:
                    self.memory_archive.add(rec)
                # Step 2 - Obtain xvals and fvals of ND points
                front = self.memory_archive.contents
                fvals = [rec.fx for rec in front]
                fvals = np.asarray(fvals)
                xvals = [rec.x for rec in front]
                xvals = np.asarray(xvals)
                # Step 3 - Add ND points to the surrogate
                npts, nobj = fvals.shape
                for i in range(npts):
                    for j in range(nobj):
                        self.fhat[j].add_points(xvals[i,:], fvals[i, j])
                # Step 4 -  Add points to the set of previously evaluated points for sampling strategy
                all_xvals = [rec.x for rec in self.evals]
                sample_prev = np.vstack((sample_prev, all_xvals))
                self.nd_archives = []
                self.failtol = self.failtol*2
            else:
                logger.info("=== Independent Restart ===")
                print('INDEPENDENT Restart Initiated')
                self.memory_archive.reset() #GOMORS only
                self.new_pop = [] #GOMORS Only
                all_xvals = [rec.x for rec in self.evals]
                sample_prev = np.vstack((sample_prev, all_xvals))

        self.sampling.init(sample_init, self.fhat, self.maxeval - self.numeval, sample_prev)

        if self.numinit is None:
            self.numinit = start_sample.shape[0]

        print('Initialization completed successfully')

    def update_archives(self):
        """Update the Tabu list, Tabu Tenure, memory archive and non-dominated front.
        """

        # Step 3 - Add newly Evaluated Points to Memory Archive and update ND_Archives list
        nimprovements = 0

        temp_error = 0
        for rec in self.new_pop:
            rule = None
            if self.sampling.select_rules.new_points is not None:
                for i in range(len(self.sampling.select_rules.new_points)):
                    if (rec.x - self.sampling.select_rules.new_points[i] < 1e-9).all():
                        rule = self.sampling.select_rules.rule_order[i]

            self.memory_archive.add(rec)
            nimprovements += self.memory_archive.improvement

            if rule is not None and self.memory_archive.improvement > 0:
                self.sampling.select_rules.total_contribution[rule] += 1

        print(self.sampling.select_rules.total_contribution)

        self.new_pop = []
        self.memory_archive.compute_fitness()

        # Step 3b - Adjust failure_count if needed
        self.successive_fail = False
        if nimprovements == 0:
            print('No Improvement Registered')
            if self.improvement_prev == 0:
                self.failcount += 1
                self.successive_fail = True
                print('No Improvement - Failure count is: ' + str(self.failcount))
            self.improvement_prev = 0
        else:
            print("Number of Improvements: " + str(nimprovements))
            self.improvement_prev = 1


    def sample_adapt(self):
        """Generate and queue samples from the search strategy"""

        # # Step 1 - Add Newly Evaluated Points to Memory Archive
        # self.save_plot(self.numeval)
        self.update_archives()

        front = self.memory_archive.contents
        fvals = [rec.fx for rec in front]
        fvals = np.asarray(fvals)
        xvals = [rec.x for rec in front]
        xvals = np.asarray(xvals)

        '''
        idx = None
        step_max = -INF
        for i in range(len(xvals)):
            point = np.asmatrix(np.copy(fvals[i, :]))
            boxA = np.copy(np.floor(point / self.sampling.epsilons))

            delta = INF
            for j in range(len(xvals)):
                if j != i:
                    boxB = np.copy(np.floor(fvals[j, :] / self.sampling.epsilons))
                    temp_delta = np.amax(np.subtract(boxB, boxA))
                    if delta > temp_delta:
                        delta = temp_delta

            if delta > step_max:
                step_max = delta
                idx = i
        '''

        idx = None
        Max_distance = -INF
        for i in range(len(xvals)):
            Min_distance = INF
            for j in range(len(xvals)):
                if j != i and scp.distance.euclidean(fvals[i, :], fvals[j, :]) < Min_distance:
                    Min_distance = scp.distance.euclidean(fvals[i, :], fvals[j, :])
            if Min_distance > Max_distance:
                Max_distance = Min_distance
                idx = i

        self.xbest = xvals[idx, :]
        self.fbest = fvals[idx, :]

        #self.interactive_plotting(fvals)
        print('NUMBER OF EVALUATIONS COMPLETED: ' + str(self.numeval))
        start = time.time()
        new_points, new_fhvals, fhvals_nd = self.sampling.make_points(npts = self.ncenters, xbest = self.xbest,
                                                                      xfront = xvals, front = fvals,
                                                                      ac_xfront = None, ac_front = None,
                                                                      proj_fun = self.proj_fun)

        #print(new_points)
        end = time.time()
        totalTime = end - start
        print('CANDIDATE SELECTION TIME: ' + str(totalTime))
        #self.interactive_plotting(fvals, new_fhvals, fhvals_nd)

        for i in range(len(new_points)):
            proposal = self.propose_eval(np.copy(np.ravel(new_points[i,:])))
            self.resubmitter.rput(proposal)

        self.sampling.select_rules.new_points = np.copy(new_points)

        self.supposed_fvals = np.copy(new_fhvals)
        self.supposed_all_fvals = np.copy(fhvals_nd)

    def start_batch(self):
        """Generate and queue a new batch of points"""
        if self.failcount > self.failtol:
            self.sample_initial()
        else:
            self.sample_adapt()

    def propose_action(self):
        """Propose an action
        """
        if self.numeval >= self.maxeval:
            #self.save_plot(self.numeval)

            # Save results to Array and Terminate
            X = np.zeros((self.maxeval, self.data.dim + self.data.nobj))
            all_xvals = [rec.x for rec in self.evals]
            all_xvals = np.asarray(all_xvals)
            X[:,0:self.data.dim] = all_xvals[0:self.maxeval,:]
            all_fvals = [rec.fx for rec in self.evals]
            all_fvals = np.asarray(all_fvals)
            X[:,self.data.dim:self.data.dim + self.data.nobj] = all_fvals[0:self.maxeval,:]
            np.savetxt('final.txt', X)
            return self.propose_terminate()
        elif self.resubmitter.num_eval_outstanding == 0:
            # UPDATE MEMORY ARCHIVE
            self.start_batch()
        return self.resubmitter.get()

    def on_complete(self, record):
        """Handle completed function evaluation.

        When a function evaluation is completed we need to ask the constraint
        handler if the function value should be modified which is the case for
        say a penalty method. We also need to print the information to the
        logfile, update the best value found so far and notify the GUI that
        an evaluation has completed.

        :param record: Evaluation record
        """
        self.numeval += 1
        record.worker_id = self.worker_id
        record.worker_numeval = self.numeval
        record.feasible = True
        self.log_completion(record)

        # 1 - Update Response Surface Model
        i = 0
        for fhat in self.fhat:
            fhat.add_points(np.copy(record.params[0]), record.value[i])
            i +=1

        # 2 - Generate a Memory Record of the New Evaluation
        srec = MemoryRecord(np.copy(record.params[0]),record.value,self.sigma_init)
        self.new_pop.append(srec)
        self.evals.append(srec)

    def interactive_plotting(self, fvals, sel_fhvals, new_fhvals_nd):
        """"If interactive plotting is on,
        """
        maxgen = (self.maxeval - self.numinit)/(self.nsamples*self.ncenters)
        curgen = (self.numeval - self.numinit)/(self.nsamples*self.ncenters) + 1

        plt.show()
        #plt.plot(self.data.pf[:,0], self.data.pf[:,1], 'g')
        all_fvals = [rec.fx for rec in self.evals]
        all_fvals = np.asarray(all_fvals)
        plt.plot(all_fvals[:,0], all_fvals[:,1], 'k+')
        plt.plot(fvals[:,0], fvals[:,1], 'b*')
        plt.plot(self.fbest[0], self.fbest[1], 'y>')
        #plt.plot(new_fhvals_nd[:,0], new_fhvals_nd[:,1], 'ro')
        #plt.plot(sel_fhvals[:,0], sel_fhvals[:,1], 'cd')
        plt.draw()
        if curgen < maxgen:
            plt.pause(0.001)
        else:
            plt.show()

    def save_plot(self, i):
        """"If interactive plotting is on,
        """
        if self.memory_archive.contents != []:
            fig = plt.figure()
            title = 'Number of Evals Completed: ' + str(i)
            front = self.memory_archive.contents
            fvals = [rec.fx for rec in front]
            fvals = np.asarray(fvals)
            all_fvals = [rec.fx for rec in self.evals]
            all_fvals = np.asarray(all_fvals)
            maxgen = (self.maxeval - self.numinit)/(self.nsamples*self.ncenters)
            curgen = (self.numeval - self.numinit)/(self.nsamples*self.ncenters) + 1
            ax = None
            if len(fvals[0]) == 2:
                ax = fig.add_subplot(111)
                if self.data.pf is not None:
                    ax.scatter(self.data.pf[:,0], self.data.pf[:,1], c = 'green', s = 2.0)
                if self.new_pop is not None:
                    for i in range(len(self.new_pop)):
                        ax.scatter(self.new_pop[i].fx[0], self.new_pop[i].fx[1], c = 'yellow', s = 50.0)
                if self.supposed_fvals is not None:
                    for i in range(len(self.supposed_fvals)):
                        ax.scatter(self.supposed_fvals[i][0], self.supposed_fvals[i][1], c='red', s=(i + 1) * 10.0)
                    for i in range(len(self.supposed_all_fvals)):
                        ax.scatter(self.supposed_all_fvals[i][0], self.supposed_all_fvals[i][1], c='green', s=2.0)

                ax.scatter(all_fvals[:,0], all_fvals[:,1], c = 'black', s = 2.0)
                ax.scatter(fvals[:,0], fvals[:,1], c = 'blue', s = 5.0)
                #==================================================================
                # Test for AC_Domination
                #all_fvals = [rec.fx for rec in self.evals]
                #all_fvals = np.asarray(all_fvals)
                #all_xvals = [rec.x for rec in self.evals]
                #all_xvals = np.asarray(all_xvals)
                #(ndf_index, df_index, box_index) = epsilon_ND_front(np.transpose(normalize_objectives(all_fvals)), self.sampling.epsilons, ACD = True)

                #all_fvals = all_fvals[ndf_index, :]
                #all_xvals = all_xvals[ndf_index, :]
                #ax.scatter(all_fvals[:, 0], all_fvals[:, 1], c='brown', s=50.0)
                # ==================================================================
            elif len(fvals[0]) == 3:
                ax = fig.add_subplot(111, projection='3d')
                if self.data.pf is not None:
                    #for i in range(len(self.data.pf)):
                    #    print(self.data.pf[i, 0], self.data.pf[i, 1], self.data.pf[i, 2])
                    ax.scatter3D(self.data.pf[:, 0], self.data.pf[:, 1], self.data.pf[:, 2], c = 'green', s = 2.0)
                if self.new_pop is not None:
                    for i in range(len(self.new_pop)):
                        ax.scatter(self.new_pop[i].fx[0], self.new_pop[i].fx[1], self.new_pop[i].fx[2], c = 'yellow', s = 50.0)
                if self.supposed_fvals is not None:
                    for i in range(len(self.supposed_fvals)):
                        ax.scatter(self.supposed_fvals[i][0], self.supposed_fvals[i][1], self.supposed_fvals[i][2], c='red', s=(i + 1) * 10.0)

                ax.scatter3D(all_fvals[:, 0], all_fvals[:, 1], all_fvals[:, 2], c = 'black', s = 2.0)
                ax.scatter3D(fvals[:, 0], fvals[:, 1], fvals[:, 2], c = 'blue', s = 5.0)

                # ==================================================================
                # Test for AC_Domination
                all_fvals = [rec.fx for rec in self.evals]
                all_fvals = np.asarray(all_fvals)
                all_xvals = [rec.x for rec in self.evals]
                all_xvals = np.asarray(all_xvals)
                (ndf_index, df_index) = ND_Front(np.transpose(normalize_objectives(all_fvals)), ACD=True)

                all_fvals = all_fvals[ndf_index, :]
                all_xvals = all_xvals[ndf_index, :]
                ax.scatter3D(all_fvals[:, 0], all_fvals[:, 1], all_fvals[:, 2], c='brown', s=50.0)
                # ==================================================================
            else:
                return
            ax.set_title(title)
            plt.show()
            #plt.draw()
            #plt.savefig('Final')
            #plt.clf()





