import logging
import os.path

import numpy as np
from poap.controller import BasicWorkerThread, ThreadController

from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant


def simple(f, max_evals=200, num_threads=1, surrogate=None):

    if surrogate is None:
        surrogate = RBFInterpolant(dim=f.dim, lb=f.lb, ub=f.ub,
                                   kernel=CubicKernel(), tail=LinearTail(f.dim))
    slhd = SymmetricLatinHypercube(dim=f.dim, num_pts=2 * (f.dim + 1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = SRBFStrategy(
        max_evals=max_evals, opt_prob=f, exp_design=slhd, surrogate=surrogate, asynchronous=True
    )

    print("Number of threads: {}".format(num_threads))
    print("Maximum number of evaluations: {}".format(max_evals))
    print("Strategy: {}".format(controller.strategy.__class__.__name__))
    print("Experimental design: {}".format(slhd.__class__.__name__))
    print("Surrogate: {}".format(surrogate.__class__.__name__))

    # Launch the threads and give them access to the objective function
    for _ in range(num_threads):
        worker = BasicWorkerThread(controller, f.eval)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    print("Best value found: {0}".format(result.value))
    print(
        "Best solution found: {0}\n".format(
            np.array_str(
                result.params[0], max_line_width=np.inf, precision=5, suppress_small=True)
        )
    )
