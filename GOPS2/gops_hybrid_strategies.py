from __future__ import print_function
import numpy as np
import math
import logging
from .pySOT1.experimental_design import SymmetricLatinHypercube, LatinHypercube
from .pySOT1.sot_sync_strategies import SyncStrategyNoConstraints
from .pySOT1.adaptive_sampling import CandidateDYCORS
from poap.strategy import BaseStrategy, RetryStrategy
from .pySOT1.rbf import *
from .pySOT1.utils import *
from .pySOT1.rs_wrappers import *
import time
import scipy.spatial as scp
from .sop_utils import *
from poap.strategy import EvalRecord

# Get module-level logger
logger = logging.getLogger(__name__)
POSITIVE_INFINITY = float("inf")

class SopRecord():

    def __init__(self, x, fx, nfail, ntabu, rank, sigma):
        """Initialize the record.

        Args:
            params: Evaluation point for the function
        Kwargs:
            status: Status of the evaluation (default 'pending')
        """
        self.x = x
        self.fx = fx
        self.nfail = nfail
        self.ntabu = ntabu
        self.rank = rank
        self.sigma = sigma

class SopCenter():

    def __init__(self, xc, index, new_points):
        """Initialize the record.

        Args:
            params: Evaluation point for the function
        Kwargs:
            status: Status of the evaluation (default 'pending')
        """
        self.xc = xc
        self.index = index
        self.new_points = new_points
        self.new_indices = []

class SyncGOPSNoConstraints(SyncStrategyNoConstraints):

    def __init__(self, worker_id, data, response_surface, maxeval, ncenters,
                 nsamples=None, exp_design=None, sampling_method=None, extra=None, extra_vals=None, ini_xs=None, ini_vals=None):

        # Check stopping criterion
        self.start_time = time.time()
        self.start = time.time()
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
        self.ini_xs = ini_xs
        self.ini_vals = ini_vals

        # Default to generate sampling points using Symmetric Latin Hypercube
        self.design = exp_design
        if self.design is None:
            if self.data.dim > 50:
                self.design = LatinHypercube(data.dim, data.dim+1)
            else:
                self.design = SymmetricLatinHypercube(data.dim, 2*(data.dim+1))

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

        # Set up search procedures and initialize
        self.sampling = sampling_method
        if self.sampling is None:
            self.sampling = CandidateDYCORS(data)

        self.ncenters = ncenters
        if nsamples is None:
            self.nsamples = 1
        self.evals = []
        self.centers = []
        self.d_thresh = 1.0
        self.numinit = None
        # self.center_type = 'Dynamic Tabu'
        self.center_type =' '

        self.check_input()

        # Start with first experimental design
        logger.info('Optimum for Trial:  ' + str(self.data.fopt))
        self.sample_initial()

        if self.numinit is None:
            self.numinit = self.numeval

    def log_itertime(self, task):
        """Log Time Taken for Task.

        :param Task: Type of task, e.g, function evaluation or RBF Fitting etc
        """
        end = time.time()
        logger.info('Elapsed Time for ' + task + str(end - self.start))
        self.start = time.time()

    def update_archive_obj(self, F):
        """Update 1) Failure count, 2) Tabu List, 3) The sampling radius sigma for each center.

        After succtol successful steps, we cut the sampling radius;
        after failtol failed steps, we double the sampling radius.

        """
        # Initialize if this is the first adaptive step
        if self.fbest_old is None:
            self.fbest_old = self.fbest
            return
        # Check if we succeeded at improving the distance-objective tradeoff
        nevals = len(self.evals)
        nsamples = min(self.nsamples*self.ncenters, self.maxeval - self.numeval)
        for cp in self.centers:
            c_index = cp.index
            check = 0
            for index in cp.new_indices:
                # rank = F[index, self.data.dim+3]
                if F[c_index, self.data.dim] - 1e-3 * math.fabs(F[c_index, self.data.dim]) > F[index, self.data.dim]:
                # if rank == 1:
                    check = 1
                    break
            if check == 0:
                self.evals[c_index].nfail += 1
                self.evals[c_index].sigma = self.evals[c_index].sigma/2
                logger.info("Reducing sigma")

        # Update Tabu Count
        for i in range(nevals-self.nsamples):
            if self.evals[i].ntabu > 0:
                if self.evals[i].ntabu < 5:
                    self.evals[i].ntabu += 1
                else:
                    self.evals[i].ntabu = 0
                    self.evals[i].nfail = 0
                    self.evals[i].sigma = self.sigma_init
                    logger.info("Reset sigma and release Tabu")

        # Update sigma and failure count
        for cp in self.centers:
            index = cp.index
            if self.evals[index].nfail > 3:
                self.evals[index].ntabu = 1
                self.evals[index].nfail = 0
                self.evals[index].sigma = self.sigma_init
                logger.info("Reset sigma and add Tabu")

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

    def update_archive(self, F):
        """Update 1) Failure count, 2) Tabu List, 3) The sampling radius sigma for each center.

        After succtol successful steps, we cut the sampling radius;
        after failtol failed steps, we double the sampling radius.

        """
        # Initialize if this is the first adaptive step
        if self.fbest_old is None:
            self.fbest_old = self.fbest
            return

        # Check if we succeeded at improving the distance-objective tradeoff
        nevals = len(self.evals)
        nsamples = min(self.nsamples*self.ncenters, self.maxeval - self.numeval)
        for cp in self.centers:
            c_index = cp.index
            check = 0
            for index in cp.new_indices:
                rank = F[index, self.data.dim+3]
                if rank == 1:
                    check = 1
                    break
            if check == 0:
                self.evals[c_index].nfail += 1
                self.evals[c_index].sigma = self.evals[c_index].sigma/2

        # Update Tabu Count
        for i in range(nevals-self.nsamples):
            if self.evals[i].ntabu > 0:
                if self.evals[i].ntabu < 5:
                    self.evals[i].ntabu += 1
                else:
                    self.evals[i].ntabu = 0
                    self.evals[i].nfail = 0
                    self.evals[i].sigma = self.sigma_init

        # Update sigma and failure count
        for cp in self.centers:
            index = cp.index
            if self.evals[index].nfail > 3:
                self.evals[index].ntabu = 1
                self.evals[index].nfail = 0
                self.evals[index].sigma = self.sigma_init

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

    def create_db(self):
        """
        create db for only the first half subset of evaluations in order of evaluations function value
        :return:
        """
        nevals = len(self.evals)
        F = np.zeros((nevals,self.data.dim+5))
        for i in range(nevals):
            F[i, 0:self.data.dim] = (self.evals[i].x - self.data.xlow) / (self.data.xup - self.data.xlow)
            F[i, self.data.dim] = self.evals[i].fx
            F[i, self.data.dim+1] = np.inf
            F[i, self.data.dim+2] = self.evals[i].ntabu
            F[i, self.data.dim+3] = self.evals[i].rank
            F[i, self.data.dim+4] = self.evals[i].nfail

        dists = scp.distance.cdist(F[:,0:self.data.dim], F[:,0:self.data.dim])
        for i in range(nevals):
            a = dists[i,:]
            F[i, self.data.dim+1] = -1.0*np.min(a[np.nonzero(a)])

        #Perform ND Sorting
        nevals = len(self.evals)
        best_percent = (1.0 - float(nevals - self.numinit)/float(self.maxeval - self.numinit)) * (0.5 - 0.01) + 0.01 #0.5 and 0.01 can be adjusted
        nmax = 100 #The maximum number of points that may be selected as centers. This is to prevent the nd_softing for all points which is expensive
        ranks = nd_sorting_best_percentage(F[:, self.data.dim:self.data.dim+2].transpose(), nmax, best_percent=best_percent)

        # print("ranks", ranks)
        F[:, self.data.dim+3] = ranks.transpose()
        return F

    def select_centers_unfixed(self, F):
        "This method for selecting centers with a unfixed number"
        nevals = len(self.evals)
        self.d_thresh = 1.0 - float(nevals - self.numinit)/float(self.maxeval - self.numinit)
        # Rank evaluated points according to 1)Tabu Status, 2)ND Rank and 3)Objective Function Value
        ind = np.lexsort((F[:, self.data.dim], F[:, self.data.dim+3], F[:, self.data.dim+2]))
        #print(ind)
        # Put xbest at the top of sorted points (ind_new), regardless of tabu status
        min_index = np.argmin(F[:, self.data.dim])
        if min_index == ind[0]:
            ind_new = np.copy(ind)
        else:
            ind_new = np.copy(ind)
            ind_new[0] = min_index
            check = 0
            i = 1
            while check == 0:
                ind_new[i] = ind[i-1]
                if ind[i]==min_index:
                    check = 1
                i = i + 1

        if self.center_type == 'Random':
            center_index = np.random.choice(nevals, self.ncenters)
        else:
            # center_count = self.ncenters
            # center_index = -1*np.ones((center_count,), dtype=np.int)
            # center_index[0] = ind_new[0]
            center_index = []
            center_index.append(ind_new[0])
            check = 1
            i = 1
            for i in range(1, nevals):
                if F[ind_new[i], self.data.dim+3] > 5: #This is opential, you might don't want to calcate taboo_regions for points on low rank fronts
                    flag = 0
                else:
                    flag = taboo_region(F[ind_new[i], 0:self.data.dim], F[center_index, :], self.sigma_init, self.data.dim, check) #local tabu
                if flag == 1:
                    center_index.append(ind_new[i])
                    check = check + 1
        return center_index

    def sample_adapt(self):
        """Generate and queue samples from the search strategy

        """
        self.d_thresh = 1.0 - float(self.numeval - self.numinit)/float(self.maxeval - self.numinit)
        maxgen = (self.maxeval - self.numinit)/(self.nsamples*self.ncenters)
        curgen = (self.numeval - self.numinit)/(self.nsamples*self.ncenters) + 1
        print('GENERATION NUMBER: ' + str(curgen) + ' OF ' + str(maxgen))
        nsamples = min(self.nsamples*self.ncenters, self.maxeval - self.numeval)
        max_centers = int(math.ceil(self.d_thresh * nsamples))

        # t_0 = time.time()
        F = self.create_db()
        # t_1 = time.time()
        # totalTime = (t_1 - t_0)
        # print('DB CREATION TIME: ' + str(totalTime))

        # t_0 = time.time()
        self.update_archive(F)
        # t_1 = time.time()
        # totalTime = (t_1 - t_0)
        # print('ARCHIVE UPDATE TIME: ' + str(totalTime))

        # Code Change - Taimoor
        # t_0 = time.time()
        new_points = np.zeros((nsamples,self.data.dim))
        num_iter = int(np.floor(float(nsamples)/float(self.nsamples)))
        # The floor of x is the largest integer i, such that i <= x.
        center_index = self.select_centers_unfixed(F)
        # print("selected index", center_index)
        # decide the number of samples for each center
        if len(center_index) > max_centers:
            center_index = center_index[:max_centers]

        nsamples_center = np.ones(len(center_index))

        min_bestcenter = math.ceil((1-self.d_thresh)*nsamples)
        if len(center_index) < nsamples:
            if np.floor_divide(nsamples, len(center_index)) > min_bestcenter:
                nsamples_center = nsamples_center * np.floor_divide(nsamples, len(center_index))
                if np.remainder(self.ncenters, len(center_index)) > 0:
                    for i in range(int(np.remainder(self.ncenters, len(center_index)))):
                        nsamples_center[i] = nsamples_center[i] + 1
            else:
                nsamples_center[0] = min_bestcenter
                if len(center_index)-1 < (nsamples-min_bestcenter):
                    nsamples_center[1:] = nsamples_center[1:] * np.floor_divide(nsamples-min_bestcenter, len(center_index)-1)
                    if np.remainder(nsamples-min_bestcenter, len(center_index)-1) > 0:
                        for i in range(int(np.remainder(nsamples-min_bestcenter, len(center_index)-1))):
                            nsamples_center[i+1] = nsamples_center[i+1] + 1

        elif len(center_index) > nsamples:
            nsamples_center = np.ones(nsamples-min_bestcenter+1)
            nsamples_center[0] = min_bestcenter
            center_index = center_index[:nsamples]
        # t_1 = time.time()
        # totalTime = (t_1 - t_0)
        # print('CENTER SELECTION TIME: ' + str(totalTime))

        # t_0 = time.time()
        self.centers = []
        j = 0
        for i in range(len(center_index)):
            # print(nsamples_center[i])
            xcenter = self.evals[center_index[i]].x
            xsigma = self.evals[center_index[i]].sigma
            new_points[j: j+int(nsamples_center[i]), :] = self.sampling.make_points(npts=int(nsamples_center[i]), xbest=xcenter, sigma=xsigma,
                                                proj_fun=self.proj_fun)
            crec = SopCenter(xcenter, center_index[i], new_points[j: j+int(nsamples_center[i]), :])
            j = j + int(nsamples_center[i])
            self.centers.append(crec)

        self.log_itertime('Find New Point(s):  ')
        for i in range(nsamples):
            proposal = self.propose_eval(np.ravel(np.copy(new_points[i, :])))
            self.resubmitter.rput(proposal)

    def start_batch(self):
        """Generate and queue a new batch of points
        """
       # NOTE: There is no Restart
        self.log_itertime('Expensive Evaluation:  ')
        self.sample_adapt()

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
        self.fhat.add_point(np.copy(record.params[0]), record.value)

        # # 2 - Generate a Memory Record of the New Evaluation
        # srec = MemoryRecord(np.copy(record.params[0]),record.value,self.sigma_init)
        # self.new_pop.append(srec)
        # self.evals.append(srec)
        #  # 3 - Update radius and failure count of center if new point does not improve non-dominated set
        # if self.centers:
        #     self.update_memory(np.copy(record.params[0]), record.value)

        # NEW CODE - Taimoor
        srec = SopRecord(np.copy(record.params[0]), record.value, 0, 0, 0, self.sigma_init)
        if self.centers:
            #nsamples = min(self.nsamples, self.maxeval - self.numeval)
            ncenters = len(self.centers)
            for i in range(ncenters):
                nsamples = self.centers[i].new_points.shape[0]
                for j in range(nsamples):
                    if np.array_equal(np.copy(record.params[0]), self.centers[i].new_points[j, :]):
                        self.centers[i].new_indices.append(self.numeval - 1)
                        break
        self.evals.append(srec)

        if record.value < self.fbest:
            self.xbest = np.copy(record.params[0])
            self.fbest = record.value

    def sample_initial(self):
        """Generate and queue an initial experimental design."""

        if self.numeval == 0:
            logger.info("=== Start ===")
        else:
            logger.info("=== Restart ===")
        self.fhat.reset()
        self.sigma = self.sigma_init
        self.status = 0
        self.xbest = None
        self.fbest_old = None
        self.fbest = np.inf
        self.fhat.reset()
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

        if self.ini_xs is None:
            # Evaluate the experimental design
            for j in range(min(start_sample.shape[0], self.maxeval - self.numeval)):
                start_sample[j, :] = self.proj_fun(start_sample[j, :])  # Project onto feasible region
                proposal = self.propose_eval(np.copy(start_sample[j, :]), )
                self.resubmitter.rput(proposal)
        elif self.ini_vals is None:
            for j in range(min(start_sample.shape[0], self.maxeval - self.numeval)):
                start_sample[j, :] = self.ini_xs[j]  # get the experiments design from given array
                proposal = self.propose_eval(np.copy(start_sample[j, :]), )
                self.resubmitter.rput(proposal)
        else:
            for j in range(min(start_sample.shape[0], self.maxeval - self.numeval)):
                start_sample[j, :] = self.ini_xs[j]
                proposal = self.propose_eval(np.copy(start_sample[j, :]), )
                proposal.record = EvalRecord(proposal.args[0], extra_args=proposal.args[1], status='completed')
                proposal.record.value = self.ini_vals[j]
                proposal.record.worker = 0
                self.on_complete(proposal.record)

        if self.extra is not None:
            self.sampling.init(np.vstack((start_sample, self.extra)), self.fhat, self.maxeval - self.numeval)
        else:
            self.sampling.init(start_sample, self.fhat, self.maxeval - self.numeval)