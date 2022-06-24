from poap.controller import ThreadController, BasicWorkerThread
from GOArbf.MRSO2.MFO_main import *


def optimize(data, mfo_method_index=0, data_mining_index=0, runs=1, n_threads=1):

    method_type ='CRBF_Restart'
    #maxeval = 15*dim+350
    maxeval = 300
    R=10
    maxeval_l=np.int(maxeval/(R+1)*R)
    maxeval_h=maxeval-np.int(maxeval_l/R)

    sampling_method = CandidateDYCORS(data=data, numcand=100 * data.dim)

    if mfo_method_index==0:
        mfo_method='MFO_RBF_DYCORS_basic'
        data_mining_method='None'
        #Run on the LF model
        data.objfunction = data.objfunction_LF
        print("Start the run on LF model")
        x_y_pair=[]
        [result_l,hisval_l,time_cost_l,computation_cost_l,his_x_y_l]=runDYCORS(runs, n_threads,CubicKernel,maxeval_l,data,
                                                                                 sampling_method,x_y_pair)

        # move to step 2- select the best point in the LF run
        x_y_pair=[]
        x_y_pair.append({"point": result_l.params[0], "value": result_l.value})

        #Run on the HF model
        print("Start the run on HF model")
        data.objfunction = data.objfunction_HF
        [result_h,hisval_h,time_cost_h,computation_cost_h,his_x_y_h]=runDYCORS(runs, n_threads,CubicKernel,maxeval_h,data,
                                                                                 sampling_method,x_y_pair)
    elif mfo_method_index==1:
        mfo_method = 'MFO_RBF_DYCORS_improved'
        # Run on the LF model
        data.objfunction = data.objfunction_LF
        print("Start the run on LF model")
        x_y_pair = []
        [result_l, hisval_l, time_cost_l, computation_cost_l, his_x_y_l] = runDYCORS(runs,n_threads, CubicKernel, maxeval_l,
                                                                                       data, sampling_method,x_y_pair)

        # move to step 2- select the best N point in the LF run by using data mining method
        X=np.zeros([maxeval_l,data.dim])
        Y=np.zeros([maxeval_l])
        for k,item in enumerate(his_x_y_l):
            X[k,:]=item["point"]
            Y[k]=item["value"]

        #Begin the data mining stage!
        x_y_pair=data_mining(X,Y,data,data_mining_index)
        if data_mining_index==0:
            data_mining_method='KMeans'
        elif data_mining_index==1:
            data_mining_method='DBSCAN'
        elif data_mining_index==2:
            data_mining_method='Multi_Restart_Gradient'

        if data.dim==2:
            plt.figure(figsize=(12, 12))
            plt.subplot(221)
            plt.scatter(X[:, 0], X[:, 1], c=y_pred)
            plt.title("Incorrect Number of Blobs")

        x_y_pair.append({"point": result_l.params[0], "value": result_l.value})
        # Run on the HF model
        print("Start the run on HF model")
        data.objfunction = data.objfunction_HF
        [result_h, hisval_h, time_cost_h, computation_cost_h, his_x_y_h] = runDYCORS(runs, n_threads, CubicKernel, maxeval_h,
                                                                                       data, sampling_method,
                                                                                       x_y_pair)

def runDYCORS(n_eval,n_threads, model, maxeval, data,sampling_method,x_y_pair):

    start_time = time()
    # Define the essential parameters of PySOT
    nthreads = n_threads # number of parallel threads / processes to be initiated (1 for serial)
    #maxeval = 50 # maximum number of function evaluations
    nsamples = nthreads # number of simultaneous evaluations in each algorithm iteration (typically set equal to nthreads)

#    n_eval = 2
    hisval = np.zeros([n_eval, maxeval])
    computation_cost = np.zeros([n_eval, 10])
    runtime = np.zeros([n_eval, 1])


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
            SyncStrategyNoConstraints(
                worker_id=0, data=data,
                maxeval=maxeval, nsamples=nsamples,
                exp_design=LatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
                response_surface=RSUnitbox(RBFInterpolant(kernel=model, maxp=maxeval),data),
                #sampling_method=mas.CandidateGradient(data=data, numcand=100*data.dim),
                sampling_method=sampling_method,
                evaluated=x_y_pair
            )


    # Launch the threads and give them access to the objective function
        for _ in range(nthreads):
            worker = BasicWorkerThread(controller, data.objfunction)
            controller.launch_worker(worker)

    # Run the surrogate optimization strategy
        result = controller.run()
        t2 = time()
        runtime[i,0] = (t2 - t1)-controller.strategy.Tmac

        print('                 run time =',runtime[i,0])

        his_x_y=[]
        hisval[i,0] = controller.fevals[0].value
        his_x_y.append({"point": controller.fevals[0].params[0], "value": controller.fevals[0].value})
        for k in range(1, maxeval):
            his_x_y.append({"point": controller.fevals[k].params[0], "value": controller.fevals[k].value})
            if  controller.fevals[k].value < hisval[i,k-1]:
                hisval[i,k] = controller.fevals[k].value
            else:
                hisval[i,k] = hisval[i,k-1]


        for s in range(10):
            computation_cost[i,s] = controller.strategy.iteration_cost[s]


    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))
    end_time = time()
    time_cost = np.sum(runtime)/n_eval
    print('runtime = ',runtime)
    print('time_cost = ',time_cost)
    print('time of DYDCORS =',end_time - start_time)


    return result, hisval, time_cost, computation_cost, his_x_y

