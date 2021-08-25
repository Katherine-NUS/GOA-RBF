from pySOT2.epsMaSO2.gomors_sync_strategies import MoSyncStrategyNoConstraints
from pySOT2.epsMaSO2.gomors_adaptive_sampling import *
from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant
from pySOT.experimental_design import SymmetricLatinHypercube, LatinHypercube
from poap.controller import SerialController, ThreadController, BasicWorkerThread
from pySOT2.epsMaSO2.archiving_strategies import NonDominatedArchive, EpsilonArchive
import numpy as np


def epsMOoptimize(data, max_evals=200, epsilons=[0.05, 0.05], num_runs=1, num_threads=1, nsamples=1, run='serial',
               surrogate=None, exp_design=None, sampling_method=None, archiving_method=None):

    if surrogate is None:
        surrogate = RBFInterpolant(dim=data.dim, lb=data.lb, ub=data.ub, kernel=CubicKernel(), tail=LinearTail(data.dim))
    if exp_design is None:
        exp_design = SymmetricLatinHypercube(dim=data.dim, num_pts=2*data.dim + 2)
    if sampling_method is None:
        sampling_method = EvolutionaryAlgorithm(data, epsilons=epsilons, cand_flag=1)
    if archiving_method is None:
        archiving_method = EpsilonArchive(size_max=200, epsilon=epsilons)

    if run == 'serial':
        for i in range(num_runs):
            controller = SerialController(objective=data.eval)
            controller.strategy = MoSyncStrategyNoConstraints(
                worker_id=0, data=data, maxeval=max_evals, nsamples=nsamples, exp_design=exp_design,
                response_surface=surrogate, sampling_method=sampling_method, archiving_method=archiving_method)

            def merit(r):
                return r.value[0]

            result = controller.run(merit=merit)
            print("Trial Number:" + str(i))
            print("Best value found: {0}".format(result.value))
            print('Best solution found: {0}\n'.format(
                np.array_str(result.params[0], max_line_width=np.inf,
                             precision=5, suppress_small=True)))
    elif run == 'asynchronous' or 'synchronous':
        for i in range(num_runs):
            controller = ThreadController()
            controller.strategy = \
                MoSyncStrategyNoConstraints(
                    worker_id=0, data=data, maxeval=max_evals, nsamples=nsamples, exp_design=exp_design,
                    response_surface=surrogate, sampling_method=sampling_method, archiving_method=archiving_method)

            # Launch the threads and give them access to the objective function
            for _ in range(num_threads):
                worker = BasicWorkerThread(controller, data.eval)
                controller.launch_worker(worker)

            # Run the optimization strategy
            def merit(r):
                return r.value[0]

            result = controller.run(merit=merit)
            print("Trial Number:" + str(i))
            print("Best value found: {0}".format(result.value))
            print('Best solution found: {0}\n'.format(
                np.array_str(result.params[0], max_line_width=np.inf,
                             precision=5, suppress_small=True)))
    else:
        print("No such method!")

