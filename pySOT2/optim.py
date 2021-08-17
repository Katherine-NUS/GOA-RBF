# Serial DYCORS is the default optimization strategy

import numpy as np
from poap.controller import BasicWorkerThread, ThreadController, SerialController

from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import DYCORSStrategy
from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant


def optimize(f, max_evals=200, num_runs=1, num_threads=1, run='serial', surrogate=None, exp_design=None):

    if surrogate is None:
        surrogate = RBFInterpolant(dim=f.dim, lb=f.lb, ub=f.ub,
                                   kernel=CubicKernel(), tail=LinearTail(f.dim))
    if exp_design is None:
        exp_design = SymmetricLatinHypercube(dim=f.dim, num_pts=2 * (f.dim + 1))

    results = np.zeros((max_evals, num_runs))
    if run == 'serial':
        for i in range(num_runs):
            controller = SerialController(objective=f.eval)
            controller.strategy = DYCORSStrategy(
                max_evals=max_evals, opt_prob=f, asynchronous=False,
                exp_design=exp_design, surrogate=surrogate, num_cand=100 * f.dim,
                batch_size=1)
            result = controller.run()
            results[:, i] = np.array(
                [o.value for o in controller.fevals if o.value is not None])
    elif run == 'asynchronous':
        # Create a strategy and a controller
        for i in range(num_runs):
            controller = ThreadController()
            controller.strategy = DYCORSStrategy(
                max_evals=max_evals, opt_prob=f, asynchronous=True,
                exp_design=exp_design, surrogate=surrogate, num_cand=100 * f.dim)

            for _ in range(num_threads):
                worker = BasicWorkerThread(controller, f.eval)
                controller.launch_worker(worker)
            # Run the optimization strategy
            result = controller.run()
            results[:, i] = np.array(
                [o.value for o in controller.fevals if o.value is not None])
    elif run == 'synchronous':
        for i in range(num_runs):
            controller = ThreadController()
            controller.strategy = DYCORSStrategy(
                max_evals=max_evals, opt_prob=f, asynchronous=False,
                exp_design=exp_design, surrogate=surrogate, num_cand=100 * f.dim,
                batch_size=num_threads)

            for _ in range(num_threads):
                worker = BasicWorkerThread(controller, f.eval)
                controller.launch_worker(worker)

            result = controller.run()
            results[:, i] = np.array(
                [o.value for o in controller.fevals if o.value is not None])
    else:
        print("No such method!")

    fvals = np.minimum.accumulate(results)
    # TODO: take mean value of n runs as the final optimal value or the min of n runs? If take mean value, how to decide
    #  the optimal point
    optif = np.mean(fvals[-1, :])
    print("Best value found: {0}".format(optif))

