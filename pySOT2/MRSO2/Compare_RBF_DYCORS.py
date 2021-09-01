import csv
import os
import numpy as np
import matplotlib as plt
from ..pySOT1.gp_regression import *
from sklearn.gaussian_process.kernels import Matern
from numpy.random import RandomState
import mat4py
import sys
from scipy import stats

from pylab import *
from matplotlib import style
import matplotlib.lines as  mlines
from decimal import Decimal


def drawConvergenceGraph(data,type,sample_file_name,file_dir, maxeval,methods):
    style.use('ggplot')
    x_axis = np.linspace(1, maxeval, maxeval)

    if type==1:
        for i in range(len(data)):
            data[i]=log10(np.clip(data[i],np.exp(-10),np.inf))


    marks=['*','p','D','o','s','^']
    colors=['k','r','b','g','m','b']
    for i in range(len(data)):
        plt.plot(x_axis, data[i], colors[i], linestyle='-', linewidth=1.5, marker=marks[i],
                 markevery=int(0.05 * maxeval), alpha=0.5, label=methods[i])


    if type == 0:
        text='Mean of the function value'
    else:
        text = 'Mean of the function value(log)'
    plt.xlabel('Evaluations', fontsize=15)
    plt.ylabel(text, fontsize=15)

    plt.legend(loc='upper right', fontsize=10)
    # plt.show()

    # Save the plot into eps format
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, file_dir)
    # sample_file_name = function_name(i+1)+'.eps'
    # sample_file_name = function_name(i+1)+'.pdf'

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # plt.savefig(results_dir + sample_file_name,dpi=1000,bbox_inches='tight')
    plt.savefig(results_dir + sample_file_name, bbox_inches='tight')



class Optimization_Method(object):

    def __init__(self, evalnum=None,trials=None,dim=None,fun_num=None,file_name=None):
        self.evalnum = evalnum
        self.trials = trials
        self.fun_num=fun_num
        self.dim = dim
        self.total_time=np.zeros([3,self.fun_num,self.trials])
        self.mhis=np.zeros([self.evalnum,self.fun_num])
        self.his=np.zeros([self.evalnum,self.trials,self.fun_num])
        self.mtotal_time=np.zeros([3,self.fun_num])
        self.statistic=np.zeros([self.fun_num,5])
        self.solutions = np.zeros([self.trials,self.fun_num])
        self.file_name=file_name


    def read(self):
        sample_file_name1 = 'Average_convergence_history'+'.csv'
        sample_file_name2 = 'history_x_y_h'+'.csv'
        sample_file_name3 = 'history_x_y_l'+'.csv'
        sample_file_name4 = 'time_cost'+'.csv'

        for t in range(self.fun_num):
            file_tail='F'+str(t+1)+'/'+str(dim)+'dim/'+str(maxeval)+'maxeval/'

            #check if the number of files matches self.trials
            file_num=0
            for fn in os.listdir(self.file_name+file_tail): #fn
                file_num=file_num+1
            if file_num < self.trials:
                    raise ValueError("Data missing! Maybe need more runs!")
            print('number of files in',(self.file_name+file_tail),'=',file_num)
            for fn in os.listdir(self.file_name+file_tail): #fn
                filename2 = self.file_name+file_tail+fn+'/'
                script_dir = os.path.dirname(__file__)
                results_dir = os.path.join(script_dir, filename2)


            for s in range(self.trials):
                with open(results_dir + sample_file_name1, "r") as csvfile:
                    reader = csv.reader(csvfile)
                    for k,rows in enumerate(reader):
                        if k>=2 and k<self.evalnum+2:
                            self.his[k-2,s,t]= float(rows[2])
                        if k==self.evalnum+1:
                            self.solutions[s,t]=float(rows[2])



                with open(results_dir + sample_file_name4, "r") as csvfile:
                    reader = csv.reader(csvfile)
                    for k,rows in enumerate(reader):
                        if k==1:
                            self.total_time[0,t,s]= float(rows[3])
                            self.total_time[1,t,s]= float(rows[4])
                            self.total_time[2,t,s]= float(rows[5])




    def calculate_mean(self):
        for s in range(self.fun_num):
            for k in range(self.evalnum):
                self.mhis[k,s]=np.mean(self.his[k,:,s])

            for y in range(3):
                self.mtotal_time[y,s]=np.mean(self.total_time[y,s,:])

            self.statistic[s,0]=np.min(self.solutions[:,s])
            self.statistic[s,1]=np.max(self.solutions[:,s])
            self.statistic[s,2]=np.median(self.solutions[:,s])
            self.statistic[s,3]=np.mean(self.solutions[:,s])
            self.statistic[s,4]=np.std(self.solutions[:,s])




def read_results(method,dim,fun_num,trial_num,maxeval):

    if method=='MRD_basic':
        file_name = 'Result_Record/'+'MFO/MFO_RBF_DYCORS_basic/None/'
    elif method=='MRD_DBSCAN':
        file_name = 'Result_Record/'+'MFO/MFO_RBF_DYCORS_improved/DBSCAN/'
    elif method=='MRD_KMeans':
        file_name = 'Result_Record/'+'MFO/MFO_RBF_DYCORS_improved/KMeans/'
    elif method=='MRD_Multi_Restart_Gradient':
        file_name = 'Result_Record/'+'MFO/MFO_RBF_DYCORS_improved/Multi_Restart_Gradient/'



    method_object = Optimization_Method(evalnum=maxeval, trials=trial_num, dim=dim, fun_num=fun_num,
                                        file_name=file_name)
    method_object.read()
    method_object.calculate_mean()

    return method_object

if __name__ == "__main__":

    methods=['MRD_basic','MRD_DBSCAN','MRD_KMeans','MRD_Multi_Restart_Gradient']
    D=[10, 30]
    trial_num=20
    fun_num=10



    for k,dim in enumerate(D):
        maxeval=15*dim+350
        competitors = []
        for h, method in enumerate(methods):
            competitors.append(read_results(method, dim, fun_num, trial_num, maxeval))
        for i in range(fun_num):

            tail = ['.pdf', '.eps', '_log10.pdf', '_log10.eps']

            for h, val2 in enumerate(tail):
                if val2 == '_log10.pdf' or val2 == '_log10.eps':
                    type = 1
                else:
                    type = 0
                file_dir0 = 'Graphs/MFO/convergence_curve/' + str(dim) + '_d_' + str(trial_num) + '_runs/'
                sample_file_name = 'F'+str(i + 1) +'_'+ str(dim) + 'd' + val2
                data=[competitors[0].mhis[:, i],competitors[1].mhis[:, i],competitors[2].mhis[:, i],competitors[3].mhis[:, i]]
                drawConvergenceGraph(data, type,sample_file_name, file_dir0, maxeval,methods)
                plt.figure()


    #start to print out the statistics
    # print('start to print out the statistics of ',str(dim),'_d problems')
    # for i in range(8):
    #     print('\multirow{5}{*}{F$_{', str(i + 1), '}$}','&', '$Best$','&',
    #           "{:.2E}".format(Decimal(str(competitors[0].statistic[0, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[1].statistic[0, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[2].statistic[0, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[3].statistic[0, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[4].statistic[0, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[5].statistic[0, i]))), '\\', '\\')
    #     print('&', '$Worst$','&',
    #           "{:.2E}".format(Decimal(str(competitors[0].statistic[1, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[1].statistic[1, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[2].statistic[1, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[3].statistic[1, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[4].statistic[1, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[5].statistic[1, i]))), '\\', '\\')
    #     print('&', '$Median$', '&',
    #           "{:.2E}".format(Decimal(str(competitors[0].statistic[2, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[1].statistic[2, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[2].statistic[2, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[3].statistic[2, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[4].statistic[2, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[5].statistic[2, i]))), '\\', '\\')
    #     print('&', '$Mean$', '&',
    #           "{:.2E}".format(Decimal(str(competitors[0].statistic[3, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[1].statistic[3, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[2].statistic[3, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[3].statistic[3, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[4].statistic[3, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[5].statistic[3, i]))), '\\', '\\')
    #     print('&', '$Std$', '&',
    #           "{:.2E}".format(Decimal(str(competitors[0].statistic[4, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[1].statistic[4, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[2].statistic[4, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[3].statistic[4, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[4].statistic[4, i]))), '&',
    #           "{:.2E}".format(Decimal(str(competitors[5].statistic[4, i]))), '\\', '\\')
    #     print('\cline{1 - 8}')


    # do the wilcoxon signed rank test
    # R=np.zeros([8,5])
    # p=np.zeros([8,5])
    # for k in range(5):
    #     for h in range(8):
    #         [R[h,k],p[h,k]]=stats.wilcoxon(competitors[5].solutions[:,h],competitors[k].solutions[:,h])
    #
    # for i in range(8):
    #     succ=['','','','','']
    #     for k in range(5):
    #         win = 0
    #         lose = 0
    #         for gg in range(len(competitors[5].solutions[:,i])):
    #             if competitors[5].solutions[gg,i]<competitors[k].solutions[gg,i]:
    #                 win = win + 1
    #             else:
    #                 lose = lose + 1
    #         if win <= lose:
    #             tmp=R[i,k]
    #             R[i,k]=210-tmp
    #         if (210-R[i,k])> R[i,k] and p[i,k]<0.05:
    #             succ[k]='+'
    #         elif ((210-R[i,k])< R[i,k] and p[i,k]<0.05):
    #             succ[k] = '-'
    #         else:
    #             succ[k] = '='
    #
    #
    #     print('F$_{', str(i + 1), '}$','&',
    #           str(210 - R[i, 0]), '&',
    #           str(R[i, 0]), '&',
    #           "{:.2E}".format(Decimal(str(p[i,0]))),'&$',succ[0], '$&',
    #           str(210 - R[i, 1]), '&',
    #           str(R[i, 1]), '&',
    #           "{:.2E}".format(Decimal(str(p[i,1]))),'&$',succ[1], '$&',
    #           str(210 - R[i, 2]), '&',
    #           str(R[i, 2]), '&',
    #           "{:.2E}".format(Decimal(str(p[i, 2]))), '&$', succ[2], '$\\','\\')
    #
    # print('***************************************************************')
    #
    # for i in range(8):
    #     succ=['','','','','']
    #     for k in range(5):
    #         if (210-R[i,k])> R[i,k] and p[i,k]<0.05:
    #             succ[k]='+'
    #         elif ((210-R[i,k])< R[i,k] and p[i,k]<0.05):
    #             succ[k] = '-'
    #         else:
    #             succ[k] = '='
    #
    #
    #     print('F$_{', str(i + 1), '}$','&',
    #           str(210 - R[i, 3]), '&',
    #           str(R[i, 3]), '&',
    #           "{:.2E}".format(Decimal(str(p[i, 3]))), '&$', succ[3], '$&',
    #           str(210 - R[i, 4]), '&',
    #           str(R[i, 4]), '&',
    #           "{:.2E}".format(Decimal(str(p[i, 4]))), '&$', succ[4], '$&',
    #           '&',
    #           '&',
    #           '&',
    #           '\\','\\')

    # for i in range(8):
    #     if i==0:
    #         print('\multirow{8}{*}{',str(dim),'-d}',)
    #     print('&','{F$_{', str(i + 1), '}$}','&',
    #   "{:.2f}".format(Decimal(str(competitors[0].mtotal_time[i,0]))), '&',
    #   "{:.2f}".format(Decimal(str(competitors[1].mtotal_time[i,0]))), '&',
    #   "{:.2f}".format(Decimal(str(competitors[2].mtotal_time[i,0]))), '&',
    #   "{:.2f}".format(Decimal(str(competitors[3].mtotal_time[i,0]))), '&',
    #   "{:.2f}".format(Decimal(str(competitors[4].mtotal_time[i,0]))), '&',
    #   "{:.2f}".format(Decimal(str(competitors[5].mtotal_time[i,0]))), '\\', '\\')
    #
    # print('start to calculate model accuracy')
    # print(competitors[0].mMSE)
    # for i in range(8):
    #     if i==0:
    #         print('\multirow{8}{*}{',str(dim),'-d}',)
    #     print('&','{F$_{', str(i + 1), '}$}','&',
    #   "{:.4f}".format(Decimal(str(competitors[0].mMSE[i]))), '&',
    #   "{:.4f}".format(Decimal(str(competitors[0].mMAE[i]))), '&',
    #   "{:.4f}".format(Decimal(str(competitors[0].mR[i]))), '&',
    #   "{:.4f}".format(Decimal(str(competitors[1].mMSE[i]))), '&',
    #   "{:.4f}".format(Decimal(str(competitors[1].mMAE[i]))), '&',
    #   "{:.4f}".format(Decimal(str(competitors[1].mR[i]))), '&',
    #   "{:.4f}".format(Decimal(str(competitors[2].mMSE[i]))), '&',
    #   "{:.4f}".format(Decimal(str(competitors[2].mMAE[i]))), '&',
    #   "{:.4f}".format(Decimal(str(competitors[2].mR[i]))),  '\\', '\\')
    #


