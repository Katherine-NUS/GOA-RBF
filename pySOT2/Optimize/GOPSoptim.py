import numpy as np
from pySOT2.GOPS2.gops_hybrid_strategies import SyncGOPSNoConstraints
from pySOT2.pySOT1.experimental_design import SymmetricLatinHypercube
from pySOT2.pySOT1.rbf import RBFInterpolant
from pySOT2.pySOT1.kernels import CubicKernel
from pySOT2.pySOT1.tails import LinearTail
from pySOT2.pySOT1.adaptive_sampling import CandidateDYCORS
from poap.controller import ThreadController, BasicWorkerThread


# only parallel optimization
def GOPSoptimize(data, max_evals=200, num_runs=1, ncenters=64, nsamples=1,
               surrogate=None, exp_design=None, sampling_method=None):

    if surrogate is None:
        surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=max_evals)
    if exp_design is None:
        exp_design = SymmetricLatinHypercube(dim=data.dim, npts=2 * (data.dim + 1))
    if sampling_method is None:
        sampling_method = CandidateDYCORS(data=data, numcand=100 * data.dim, weights=[1.0])

    num_threads = ncenters * nsamples
    # Create a strategy and a controller
    for i in range(num_runs):
        controller = ThreadController()
        controller.strategy = SyncGOPSNoConstraints(
                worker_id=0, data=data, maxeval=max_evals, ncenters=ncenters, nsamples=nsamples, exp_design=exp_design,
                response_surface=surrogate, sampling_method=sampling_method)

        for _ in range(num_threads):
            worker = BasicWorkerThread(controller, data.objfunction)
            controller.launch_worker(worker)

        result = controller.run()
        print("Trial Number:" + str(i))
        print("Best value found: {0}".format(result.value))
        print('Best solution found: {0}\n'.format(
                np.array_str(result.params[0], max_line_width=np.inf,
                             precision=5, suppress_small=True)))
