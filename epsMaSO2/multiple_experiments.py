from gomors_sync_strategies import MoSyncStrategyNoConstraints
from gomors_adaptive_sampling import *
from multiobjective_problems import DTLZ2
# from multiobjective_multifidelity_problems import *
from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant
from pySOT.experimental_design import SymmetricLatinHypercube, LatinHypercube
from poap.controller import SerialController, ThreadController, BasicWorkerThread
from archiving_strategies import NonDominatedArchive, EpsilonArchive
import numpy as np
import os.path
import logging
# from Townbrook import TB2
# from Cannonsville import CV2, CV2_Continous

def main():
    for nthreads in [4]:
        for nobj in [2]:
            for dim in [15]:
                data_list = [DTLZ2(dim = 10, nobj = 2)]
                pnames = [type(data).__name__ for data in data_list]
                for (pname, data) in zip(pnames, data_list):
                    epsilon = [0.01] * nobj
                    maxeval = 600
                    num_trials = 20

                    if not os.path.exists("./" + str(nthreads)):
                        os.makedirs(str(nthreads))
                    if not os.path.exists("./" + str(nthreads) + "/" + pname):
                        os.makedirs(str(nthreads) + "/" + pname)
                    if not os.path.exists("./" + str(nthreads) + "/" + pname + "/" + str(nobj)):
                        os.makedirs(str(nthreads) + "/" + pname + "/" + str(nobj))
                    if not os.path.exists("./" + str(nthreads) + "/" + pname + "/" + str(nobj) + "/" + str(dim)):
                        os.makedirs(str(nthreads) + "/" + pname + "/" + str(nobj) + "/" + str(dim))

                    experiment(pname, data, nthreads, maxeval, num_trials, epsilon)


def experiment(pname, data, nthreads, maxeval, num_trials, epsilon):
    print('Problem being solved: ' + pname)
    print('Number of Threads: ' + str(nthreads))
    for i in range(num_trials):
        optimization_trial(pname, data, epsilon, nthreads, maxeval, i + 1)

def optimization_trial(pname, data, epsilon, nthreads, maxeval, num):
    nsamples = nthreads
    print("Trial Number:" + str(num))

    # Create a strategy and a controller
    surrogate = RBFInterpolant(dim=data.dim, lb=data.lb, ub=data.ub, kernel=CubicKernel(), tail=LinearTail(data.dim))
    exp_design = SymmetricLatinHypercube(dim=data.dim, num_pts=2*data.dim + 2)

    controller = ThreadController()
    controller.strategy = \
        MoSyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=exp_design,
            response_surface=surrogate,
            sampling_method=EvolutionaryAlgorithm(data, epsilons=epsilon, cand_flag=1),
            archiving_method=EpsilonArchive(size_max=400, epsilon=epsilon))

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.eval)
        controller.launch_worker(worker)

    # Run the optimization strategy
    def merit(r):
        return r.value[0]
    result = controller.run(merit=merit)

    start = time.time()
    result = controller.run(merit = merit)
    end = time.time()
    print('time = {}'.format(end - start))

    x = np.loadtxt('final.txt')
    # controller.strategy.save_plot(trial_number)
    fname = pname + '_' + str(data.nobj) + '_' + str(data.dim) + '_epsMaSO_' + str(maxeval) + '_' + str(
        num) + '_' + str(nthreads) + '.txt'
    np.savetxt(str(nthreads) + "/" + pname + "/" + str(data.nobj) + "/" + str(data.dim) + "/" + fname, x)


if __name__ == '__main__':
    main()
