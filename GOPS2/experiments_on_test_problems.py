from .gops_hybrid_strategies import SyncGOPSNoConstraints
from .pySOT1.test_problems import Ackley
from .pySOT1.sot_sync_strategies import SyncStrategyNoConstraints
from .pySOT1.experimental_design import SymmetricLatinHypercube
from .pySOT1.rbf import RBFInterpolant
from .pySOT1.kernels import CubicKernel
from .pySOT1.tails import LinearTail
from .pySOT1.adaptive_sampling import CandidateDYCORS
from .pySOT1.heuristic_methods import GeneticAlgorithm
from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path
import logging
from .BBOB import BBOB
import time

def main():

    pname = 'F15'
    pid = 15
    dim = 40
    ninit = 2 * (dim + 1)
    niter = 30
    ncenters = 64
    nsamples = 1
    nthreads = ncenters * nsamples

    # Initiate Log File
    the_filename = './logfiles/' + pname + '_' + str(dim) + '_' + 'GOPS' + '_' + str(ninit) + '_' + str(
        ncenters) + '_' + str(nsamples) + '_' + str(niter)  + '.log'
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists(the_filename):
        os.remove(the_filename)

    log = logging.getLogger()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=the_filename)
    fh.setLevel(logging.INFO)
    log.addHandler(fh)

    maxeval = ninit + ncenters * nsamples * niter
    print("\nNumber of centers: " + str(ncenters))
    print("Maximum number of parallel iterations: " + str(niter))
    print("Search strategy: CandidateDYCORS")
    print("Experimental design: Latin Hypercube")
    print("Ensemble surrogates: Cubic RBF, domain scaled to unit box")

    instance_id = 0
    data = BBOB(id=pid, instance=instance_id, dim=dim)

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncGOPSNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, ncenters=ncenters, nsamples=nsamples,
            exp_design=SymmetricLatinHypercube(dim=data.dim, npts=2 * (data.dim + 1)),
            response_surface=RBFInterpolant(kernel=CubicKernel, tail=LinearTail,
                                            maxp=maxeval),
            sampling_method=CandidateDYCORS(data=data, numcand=100 * data.dim, weights=[1.0]))

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.objfunction)
        controller.launch_worker(worker)

    t_0 = time.time()
    # Run the optimization strategy
    result = controller.run()
    # Stop trial timer
    t_1 = time.time()
    runtime = t_1 - t_0

    opt_diff = result.value - data.fopt
    print('Best value found: {0}'.format(opt_diff))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))
    print('Runtime of trial: {0}\n'.format(runtime))

    return opt_diff
    # Record optimum value for current trial
    print('Optimum is: ' + str(data.fopt))


if __name__ == '__main__':
    main()
