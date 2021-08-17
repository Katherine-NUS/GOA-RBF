import numpy as np
from GOMORS2.gomors_sync_strategies import MoSyncStrategyNoConstraints
from GOMORS2.gomors_adaptive_sampling import EvolutionaryAlgorithm
from GOMORS2.archiving_strategies import NonDominatedArchive, EpsilonArchive
from GOMORS2.pySOT1.experimental_design import SymmetricLatinHypercube
from GOMORS2.pySOT1.rbf import RBFInterpolant
from GOMORS2.pySOT1.kernels import CubicKernel
from GOMORS2.pySOT1.tails import LinearTail
from poap.controller import SerialController, ThreadController, BasicWorkerThread

# TODO: do multiple runs, decide how to take the final optimal value and point
def MOoptimize(data, max_evals=200, epsilons=[0.05, 0.05], num_runs=1, num_threads=1, nsamples=1, run='serial',
               surrogate=None, exp_design=None, sampling_method=None, archiving_method=None):

    if surrogate is None:
        surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=max_evals)
    if exp_design is None:
        exp_design = SymmetricLatinHypercube(dim=data.dim, npts=2 * (data.dim + 1))
    if sampling_method is None:
        sampling_method = EvolutionaryAlgorithm(data, epsilons=epsilons, cand_flag=1)
    if archiving_method is None:
        archiving_method = EpsilonArchive(size_max=200, epsilon=epsilons)

    if run == 'serial':
        for i in range(num_runs):
            controller = SerialController(objective=data.objfunction)
            controller.strategy = MoSyncStrategyNoConstraints(
                worker_id=0, data=data, maxeval=max_evals, nsamples=nsamples, exp_design=exp_design,
                response_surface=surrogate, sampling_method=sampling_method, archiving_method=archiving_method)

            def merit(r):
                return r.value[0]
            result = controller.run(merit=merit)
            print("Best value found: {0}".format(result.value))
    elif run == 'asynchronous' or 'synchronous':
        # Create a strategy and a controller
        for i in range(num_runs):
            controller = ThreadController()
            controller.strategy = MoSyncStrategyNoConstraints(
                worker_id=0, data=data, maxeval=max_evals, nsamples=nsamples, exp_design=exp_design,
                response_surface=surrogate, sampling_method=sampling_method, archiving_method=archiving_method)

            for _ in range(num_threads):
                worker = BasicWorkerThread(controller, data.objfunction)
                controller.launch_worker(worker)
            # Run the optimization strategy

            def merit(r):
                return r.value[0]
            result = controller.run(merit=merit)
            print("Best value found: {0}".format(result.value))
    else:
        print("No such method!")

