# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 12:03:09 2016

@author: mengq
"""

from time import time
import csv
from ..pySOT1 import *
from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import style
import matplotlib.lines as  mlines
import modified_gp_regression as mgp
import modified_adaptive_sampling as mas
import os
from .gpr import GaussianProcessRegressor
from .kernels import (RBF, WhiteKernel, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel as C)
from .gp_extras.kernels import ManifoldKernel
from .modified_test_problems import *
import modified_sot_sync_strategies as msss
from .generate_fixed_sample_points import Sample_points
import sys
#%%%

def gpmain(n_eval,model,maxeval,data,tp,ty,sample_size):


    start_time = time()

    # Define the essential parameters of PySOT
    nthreads = 1 # number of parallel threads / processes to be initiated (1 for serial)
    #maxeval = 50 # maximum number of function evaluations
    nsamples = nthreads # number of simultaneous evaluations in each algorithm iteration (typically set equal to nthreads)

#    n_eval = 2
    bestCost = np.zeros([n_eval, maxeval])
#    averageCost = np.zeros(maxeval)
    computation_cost = np.zeros([n_eval, 10])

    MSE = np.zeros([n_eval, int(maxeval/50)])
    MAE = np.zeros([n_eval, int(maxeval/50)])
    R = np.zeros([n_eval, int(maxeval/50)])
    runtime = np.zeros([n_eval, 1])

    if data.dim==1:
        npts=2
    else:
        npts =2 * data.dim + 1

    #Print setting of PySOT you are using (optional to remember what experimental options you used)
    print("\nNumber of threads: " + str(nthreads))
    print("Maximum number of evaluations: " + str(maxeval))
    print("Search strategy: EI-DE")
    print("Experimental design: Latin Hypercube")
    print("Ensemble surrogates: "+str(model)+", domain scaled to unit box")

    # Create a strategy (defined in the pysot library) and a controller (defined in POAP library)
    for i in range(n_eval):
        t1 = time()
        print('GP:  Start the ',str(i),'th run:')
        controller = ThreadController() # This class instance manages the Parallel framework for assigning obj function evaluations to threads
        controller.strategy = \
            msss.SyncStrategyNoConstraints(
                worker_id=0, data=data,
                maxeval=maxeval, nsamples=nsamples,
                exp_design=LatinHypercube(dim=data.dim, npts=2*data.dim+1),
                response_surface=mgp.GPRegression(gp=model,maxp=maxeval),
                #sampling_method=mas.DifferentialEvolution_EI(data=data)) #this method uses SRBF + EI criterion
                sampling_method=mas.GeneticAlgorithm_EI(data=data),
                tp=tp,
                ty=ty,
                sample_size=sample_size)



        # Launch the threads and give them access to the objective function
        for _ in range(nthreads):
            worker = BasicWorkerThread(controller, data.objfunction)
            controller.launch_worker(worker)

        # Run the surrogate optimization strategy
        result = controller.run()
        t2 = time()
        runtime[i,0] = (t2 - t1)-controller.strategy.Tmac

        bestCost[i,0] = controller.fevals[0].value
        for s in range(10):
            computation_cost[i,s] = controller.strategy.iteration_cost[s]
        # record the model accuracy
        assert len(controller.strategy.MAE) == len(controller.strategy.MSE) == len(controller.strategy.R), \
            "length mismatch between MSE, MAE and R"

        # for g in range(maxeval-len(controller.strategy.MAE)):
        #     MAE[i,g] = nan
        #     MSE[i,g] = nan
        #     R[i,g] = nan
        # #     MAE[i,g] = controller.strategy.MAE[0]
        # #     MSE[i,g] = controller.strategy.MSE[0]
        # #     R[i,g] = controller.strategy.R[0]
        #
        # for g in range(maxeval-len(controller.strategy.MAE),maxeval):
        #     MAE[i,g] = controller.strategy.MAE[g-maxeval+len(controller.strategy.MAE)]
        #     MSE[i,g] = controller.strategy.MSE[g-maxeval+len(controller.strategy.MSE)]
        #     R[i,g] = controller.strategy.R[g-maxeval+len(controller.strategy.R)]



        MAE[i,:] = controller.strategy.MAE
        MSE[i,:] = controller.strategy.MSE
        R[i,:] = controller.strategy.R
        # for g in range(maxeval):
        #             print('MAE = ',MAE[i,g])

        for k in range(1, maxeval):
            if  controller.fevals[k].value < bestCost[i,k-1]:
                bestCost[i,k] = controller.fevals[k].value
            else:
                bestCost[i,k] = bestCost[i,k-1]


    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))

    end_time = time()
    time_cost = np.sum(runtime)/n_eval
    print('runtime = ',runtime)
    print('time_cost = ',time_cost)
    return bestCost,time_cost,computation_cost,MSE,MAE,R
#%%
def DYCORS_main(n_eval,model,maxeval,data,tp,ty,sample_size,sampling_method):

    start_time = time()
    # Define the essential parameters of PySOT
    nthreads = 1 # number of parallel threads / processes to be initiated (1 for serial)
    #maxeval = 50 # maximum number of function evaluations
    nsamples = nthreads # number of simultaneous evaluations in each algorithm iteration (typically set equal to nthreads)

#    n_eval = 2
    bestCost = np.zeros([n_eval, maxeval])
    computation_cost = np.zeros([n_eval, 10])
    MSE = np.zeros([n_eval, int(maxeval/50)])
    MAE = np.zeros([n_eval, int(maxeval/50)])
    R = np.zeros([n_eval, int(maxeval/50)])
    runtime = np.zeros([n_eval, 1])


    if data.dim == 1:
        npts = 2
    else:
        npts = 2 * data.dim + 1

    #Print setting of PySOT you are using (optional to remember what experimental options you used)
    print("\nNumber of threads: " + str(nthreads))
    print("Maximum number of evaluations: " + str(maxeval))
    print("Search strategy: CandidateDYCORS")
    print("Experimental design: Latin Hypercube")
    print("Ensemble surrogates: "+str(model)+", domain scaled to unit box")



    # Create a strategy (defined in the pysot library) and a controller (defined in POAP library)

    for i in range(n_eval):
        t1 = time()
        print('DYCORS: Start the ',str(i),'th run:')
        controller = ThreadController() # This class instance manages the Parallel framework for assigning obj function evaluations to threads
        controller.strategy = \
            msss.SyncStrategyNoConstraints(
                worker_id=0, data=data,
                maxeval=maxeval, nsamples=nsamples,
                exp_design=LatinHypercube(dim=data.dim, npts=npts),
                response_surface=RSUnitbox(RBFInterpolant(kernel=model, maxp=maxeval),data),
                #sampling_method=mas.CandidateGradient(data=data, numcand=100*data.dim),
                sampling_method=sampling_method,
                tp=tp,
                ty=ty,
                sample_size=sample_size)


    # Launch the threads and give them access to the objective function
        for _ in range(nthreads):
            worker = BasicWorkerThread(controller, data.objfunction)
            controller.launch_worker(worker)

    # Run the surrogate optimization strategy
        result = controller.run()
        t2 = time()
        runtime[i,0] = (t2 - t1)-controller.strategy.Tmac

        print('                 run time =',runtime[i,0])
        bestCost[i,0] = controller.fevals[0].value
        for s in range(10):
            computation_cost[i,s] = controller.strategy.iteration_cost[s]

        for k in range(1, maxeval):
            if  controller.fevals[k].value < bestCost[i,k-1]:
                bestCost[i,k] = controller.fevals[k].value
            else:
                bestCost[i,k] = bestCost[i,k-1]

        # record the model accuracy
        assert len(controller.strategy.MAE) == len(controller.strategy.MSE) == len(controller.strategy.R), \
            "length mismatch between MSE, MAE and R"


        #print(controller.strategy.MAE)
        MAE[i,:] = controller.strategy.MAE
        MSE[i,:] = controller.strategy.MSE
        R[i,:] = controller.strategy.R
        # for g in range(maxeval-len(controller.strategy.MAE),maxeval):
        #     MAE[i,g] = controller.strategy.MAE[g-maxeval+len(controller.strategy.MAE)]
        #     MSE[i,g] = controller.strategy.MSE[g-maxeval+len(controller.strategy.MSE)]
        #     R[i,g] = controller.strategy.R[g-maxeval+len(controller.strategy.R)]
        #
        # for g in range(maxeval-len(controller.strategy.MAE)):
        #     MAE[i,g] = nan
        #     MSE[i,g] = nan
        #     R[i,g] = nan



    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))
    end_time = time()
    time_cost = np.sum(runtime)/n_eval
    print('runtime = ',runtime)
    print('time_cost = ',time_cost)
    print('time of DYDCORS =',end_time - start_time)
    return bestCost,time_cost,computation_cost,MSE,MAE,R
    #%%


def function_object(function_index,dim):
    k1 = 20
    k2 = 100
    f1 = np.random.uniform(0,100,k1)
    f2 = np.random.uniform(0,100,k2)
    z1 = np.random.uniform(0,1,[k1,dim])
    z2 = np.random.uniform(0,1,[k2,dim])
    return {
        1: Rastrigin(dim),
        2: Ackley(dim),
        3: Michalewicz(dim),
        4: Levy(dim),
        5: Rosenbrock(dim),
        6: Schwefel(dim),
        7: Sphere(dim),
        8: Exponential(dim),
        9: StyblinskiTang(dim),
        10: Whitley(dim),
        11: SchafferF7(dim),
        12: Schoen20k(dim,f=f1,z=z1),
        13: Schoen100k(dim,f=f2,z=z2)
    }[function_index]


def function_name(function_index):
        return {
            1: 'Rastrigin',
            2: 'Ackley',
            3: 'Michalewicz',
            4: 'Levy',
            5: 'Rosenbrock',
            6: 'Schwefel',
            7: 'Sphere',
            8: 'Exponential',
            9: 'StyblinskiTang',
            10: 'Whitley',
            11: 'SchafferF7',
            12: 'Schoen-20k',
            13: 'Schoen-100k'
        }[function_index]


def  drawGraph(type,averageCost_CRBF,averageCost_TRBF,averageCost_GRBF,averageCost_GPEI_STA,averageCost_GPEI_NONS,sample_file_name,file_dir,maxeval):
        style.use('ggplot')
        x_axis = np.linspace(1,maxeval,maxeval)
        plot(x_axis, averageCost_CRBF, 'r', linestyle='-',linewidth=1.5,marker = 'o',markevery = int(0.05*maxeval),alpha=0.5, label = "CRBF-DYCORS")
        plot(x_axis, averageCost_TRBF, 'b',linestyle='-', linewidth=1.5,marker = 's',markevery = int(0.05*maxeval),alpha=0.5, label = "TRBF-DYCORS")
        plot(x_axis, averageCost_GRBF, 'k',linestyle='-', linewidth=1.5,marker = 'd',markevery = int(0.05*maxeval),alpha=0.5, label = "GRBF-DYCORS")
        plt.plot(x_axis, averageCost_GPEI_STA, 'b',linestyle='-', linewidth=1.5,marker = 'v',markevery = int(0.05*maxeval),alpha=0.5, label = "GPEI_STA")
        plt.plot(x_axis, averageCost_GPEI_NONS, 'r',linestyle='-', linewidth=1.5,marker = '^',markevery = int(0.05*maxeval),alpha=0.5, label = "GPEI_NONS")

        plt.xlabel('Evaluations',fontsize=10)
        if type == 'covergence_curve':
            plt.ylabel('Mean of the function value',fontsize=10)
        elif type == 'mse_curve':
            plt.ylabel('Relative mean square error (global space)',fontsize=10)
        elif type == 'mae_curve':
            plt.ylabel('Relative mean square error (around evaluation points)',fontsize=10)
        elif type == 'r_curve':
            plt.ylabel('Relative mean square error (around current best)',fontsize=10)
        plt.legend(loc='best',fontsize=11)
        #plt.show()

        # Save the plot into eps format
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, file_dir)
        #sample_file_name = function_name(i+1)+'.eps'
        #sample_file_name = function_name(i+1)+'.pdf'

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)


        #plt.savefig(results_dir + sample_file_name,dpi=1000,bbox_inches='tight')
        plt.savefig(results_dir + sample_file_name,bbox_inches='tight')



def main(dim, sampling_index,run_index):

    pst=time()

    runs = 1

    method_type ='CRBF_Restart'

    maxeval = 50*dim

    kernel_sta = 1.0 * RBF(length_scale=1.0, length_scale_bounds="fixed")
    gp_sta = GaussianProcessRegressor(kernel=kernel_sta, n_restarts_optimizer=10)


    kernel_matern_1 = Matern(length_scale=1.0, length_scale_bounds="fixed", nu=1.5)
    gp_ma_1 = GaussianProcessRegressor(kernel=kernel_matern_1, n_restarts_optimizer=10)

    kernel_matern_2 = Matern(length_scale=1.0, length_scale_bounds="fixed", nu=2.5)
    gp_ma_2 = GaussianProcessRegressor(kernel=kernel_matern_2, n_restarts_optimizer=10)


    kernel_nons = C(1.0, (0.01, 100)) \
    * ManifoldKernel.construct(base_kernel=RBF(0.1), architecture=((1, 2),),
                                transfer_fct="tanh", max_nn_weight=1)
    #kernel_nons =  C(0.1, (0.01, 10.0))* (DotProduct(sigma_0=1.0))

    gp_nons = GaussianProcessRegressor(kernel=kernel_nons,alpha=1e-5, n_restarts_optimizer=10)



    for i in range(1):

        data = function_object(i+1,dim)
        print(data.info)

        if sampling_index == 0:
            sampling_method = mas.CandidateSpearmint(data=data, numcand=100 * data.dim)
            sampling_name = 'CandidateSpearmint'
        elif sampling_index == 1:
            sampling_method = mas.CandidateGradient(data=data, numcand=100 * data.dim)
            sampling_name = 'CandidateGradient'
        elif sampling_index == 2:
            sampling_method = mas.CandidateDYCORS(data=data, numcand=100 * data.dim)
            sampling_name = 'CandidateDYCORS'

        #calculate the function evaluation time
        t00=time()
        for kk in range(maxeval):
            x = np.random.uniform(data.xlow,data.xup,dim)

            fit = data.objfunction(x)
        t01=time()
        Teval = t01-t00
        print('function evaluation time = ',Teval)

        sample_size=250*dim
        P=Sample_points(sample_size,dim)
        ty = np.zeros([sample_size,1])
        tp=np.kron(np.ones((sample_size,1)), data.xlow)+np.multiply(((np.kron(np.ones((sample_size,1)), data.xup)-np.kron(np.ones((sample_size,1)), data.xlow))),P)


        for k in range(sample_size):
            tmp=np.zeros([data.dim])

            for h in range(data.dim):
                tmp[h]=tp[k,h]
            ty[k]=data.objfunction(tmp)


        if method_type == 'GRBF_Restart':
            [bestCost,total_time,phase_time,MAE,MSE,R]=DYCORS_main(runs,Multiquadric,maxeval,data,tp,ty,sample_size,sampling_method)
        elif method_type == 'CRBF_Restart':
            [bestCost,total_time,phase_time,MAE,MSE,R]=DYCORS_main(runs,CubicKernel,maxeval,data,tp,ty,sample_size,sampling_method)
        elif method_type == 'TRBF_Restart':
            [bestCost,total_time,phase_time,MAE,MSE,R]=DYCORS_main(runs,TPSKernel,maxeval,data,tp,ty,sample_size,sampling_method)

        averageCost = np.zeros(maxeval)
        # record average mae mse and r
        averageMAE = np.zeros(maxeval)
        averageMSE = np.zeros(maxeval)
        averageR = np.zeros(maxeval)

        for k in range(int(maxeval)):
            averageCost[k] = np.mean(bestCost[:,k])

        for k in range(int(maxeval/50)):
            averageMAE[k] = np.mean(MAE[:,k])
            averageMSE[k] = np.mean(MSE[:,k])
            averageR[k] = np.mean(R[:,k])

        # save the statistics best worst median mean std
        solution_1 = bestCost[:,maxeval-1]

        best_solution = np.min(solution_1)
        worst_solution = np.max(solution_1)
        median_solution = np.median(solution_1)
        mean_solution = np.mean(solution_1)
        std_solution = np.std(solution_1)





        filename1 = 'Result_Record/'+'F'+str(i+1)+'/'+method_type+'/'+sampling_name+'/'+str(dim)+'dim/'+str(maxeval)+'maxeval/'+str(run_index+1)+'/'
        sample_file_name1 = 'time_cost'+'.csv'
        script_dir1 = os.path.dirname(__file__)
        results_dir1 = os.path.join(script_dir1, filename1)
        if not os.path.isdir(results_dir1):
            os.makedirs(results_dir1)
        filehead1 = ['','',method_type]
        a = open(results_dir1 + sample_file_name1, "w")
        writer = csv.writer(a)
        writer.writerow(filehead1)
        a.close()
        s = 'F'+str(i+1)
        t=[s,'',str(total_time)]
        csvfile1 = open(results_dir1 + sample_file_name1, "a")
        writer = csv.writer(csvfile1)
        writer.writerow(t)
        csvfile1.close()





        filename4 = 'Result_Record/'+'F'+str(i+1)+'/'+method_type+'/'+sampling_name+'/'+str(dim)+'dim/'+str(maxeval)+'maxeval/'+str(run_index+1)+'/'
        sample_file_name4 = 'phase_computational_time'+'.csv'
        script_dir4 = os.path.dirname(__file__)
        results_dir4 = os.path.join(script_dir4, filename4)
        if not os.path.isdir(results_dir4):
            os.makedirs(results_dir4)
        filehead4 = ['','',method_type,'','','','','','','','','']
        a = open(results_dir4 + sample_file_name4, "w")
        writer = csv.writer(a)
        writer.writerow(filehead4)
        a.close()
        s = 'F'+str(i+1)
        csvfile4 = open(results_dir4 + sample_file_name4, "a")
        writer = csv.writer(csvfile4)
        writer.writerow([s])
        for k in range(runs):
            writer.writerow(['',str(k),str(phase_time[k][0]),str(phase_time[k][1]),str(phase_time[k][2]),str(phase_time[k][3]),str(phase_time[k][4]),str(phase_time[k][5]),str(phase_time[k][6]),str(phase_time[k][7]),str(phase_time[k][8]),str(phase_time[k][9])])
        csvfile4.close()

        filename5 = 'Result_Record/'+'F'+str(i+1)+'/'+method_type+'/'+sampling_name+'/'+str(dim)+'dim/'+str(maxeval)+'maxeval/'+str(run_index+1)+'/'
        sample_file_name5 = 'Average_convergence_history_'+'.csv'
        script_dir5 = os.path.dirname(__file__)
        results_dir5 = os.path.join(script_dir5, filename5)
        if not os.path.isdir(results_dir5):
            os.makedirs(results_dir5)
        filehead5 = ['','',method_type]
        a = open(results_dir5 + sample_file_name5, "w")
        writer = csv.writer(a)
        writer.writerow(filehead5)
        a.close()
        csvfile5 = open(results_dir5 + sample_file_name5, "a")
        writer = csv.writer(csvfile5)
        writer.writerow([s])
        for item in range(maxeval):
            writer.writerow(['',str(item),str(averageCost[item])])
        csvfile5.close()

        filename6 = 'Result_Record/'+'F'+str(i+1)+'/'+method_type+'/'+sampling_name+'/'+str(dim)+'dim/'+str(maxeval)+'maxeval/'+str(run_index+1)+'/'
        sample_file_name6 = 'Average_global_MSE_history_'+'.csv'
        script_dir6 = os.path.dirname(__file__)
        results_dir6 = os.path.join(script_dir6, filename6)
        if not os.path.isdir(results_dir6):
            os.makedirs(results_dir6)
        filehead6 = ['','',method_type]
        a = open(results_dir6 + sample_file_name6, "w")
        writer = csv.writer(a)
        writer.writerow(filehead6)
        a.close()
        csvfile6 = open(results_dir6 + sample_file_name6, "a")
        writer = csv.writer(csvfile6)
        writer.writerow([s])
        for item in range(int(maxeval/50)):
            writer.writerow(['',str(item),str(averageMAE[item])])
        csvfile6.close()

        filename7 = 'Result_Record/'+'F'+str(i+1)+'/'+method_type+'/'+sampling_name+'/'+str(dim)+'dim/'+str(maxeval)+'maxeval/'+str(run_index+1)+'/'
        sample_file_name7 = 'Average_local_MSE_history_'+'.csv'
        script_dir7 = os.path.dirname(__file__)
        results_dir7 = os.path.join(script_dir7, filename7)
        if not os.path.isdir(results_dir7):
            os.makedirs(results_dir7)
        filehead7 = ['','',method_type]
        a = open(results_dir7 + sample_file_name7, "w")
        writer = csv.writer(a)
        writer.writerow(filehead7)
        a.close()
        csvfile7 = open(results_dir7 + sample_file_name7, "a")
        writer = csv.writer(csvfile7)
        writer.writerow([s])
        for item in range(int(maxeval/50)):
            writer.writerow(['',str(item),str(averageMSE[item])])
        csvfile7.close()

        filename8 = 'Result_Record/'+'F'+str(i+1)+'/'+method_type+'/'+sampling_name+'/'+str(dim)+'dim/'+str(maxeval)+'maxeval/'+str(run_index+1)+'/'
        sample_file_name8 = 'Average_currentbest_MSE_history_'+'.csv'
        script_dir8 = os.path.dirname(__file__)
        results_dir8 = os.path.join(script_dir8, filename8)
        if not os.path.isdir(results_dir8):
            os.makedirs(results_dir8)
        filehead8 = ['','',method_type]
        a = open(results_dir8 + sample_file_name8, "w")
        writer = csv.writer(a)
        writer.writerow(filehead8)
        a.close()
        csvfile8 = open(results_dir8 + sample_file_name8, "a")
        writer = csv.writer(csvfile8)
        writer.writerow([s])
        for item in range(int(maxeval/50)):
            writer.writerow(['',str(item),str(averageR[item])])
        csvfile8.close()


        filename9 = 'Result_Record/'+'F'+str(i+1)+'/'+method_type+'/'+sampling_name+'/'+str(dim)+'dim/'+str(maxeval)+'maxeval/'+str(run_index+1)+'/'
        sample_file_name9 = 'Ave_FuncEvaTime'+'.csv'
        script_dir9 = os.path.dirname(__file__)
        results_dir9 = os.path.join(script_dir9, filename9)
        if not os.path.isdir(results_dir9):
            os.makedirs(results_dir9)
        filehead9 = ['','','Ave_FuncEvaTime']
        a = open(results_dir9 + sample_file_name9, "w")
        writer = csv.writer(a)
        writer.writerow(filehead9)
        a.close()
        s = 'F'+str(i+1)
        t=[s,'',str(Teval)]
        csvfile9 = open(results_dir9 + sample_file_name9, "a")
        writer = csv.writer(csvfile9)
        writer.writerow(t)
        csvfile9.close()

        print('program takes ',time()-pst,'seconds')


if __name__ == "__main__":
    # dim = int(sys.argv[1])
    # sampling_index = int(sys.argv[2])
    # run_index = int(sys.argv[3])
    # main(dim, sampling_index,run_index)
    main(1, 2, 0)
