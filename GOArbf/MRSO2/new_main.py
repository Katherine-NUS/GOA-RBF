# -*- coding: utf-8 -*-
# Spearmint
#
# Academic and Non-Commercial Research Use Software License and Terms
# of Use
#
# Spearmint is a software package to perform Bayesian optimization
# according to specific algorithms (the “Software”).  The Software is
# designed to automatically run experiments (thus the code name
# 'spearmint') in a manner that iteratively adjusts a number of
# parameters so as to minimize some objective in as few runs as
# possible.
#
# The Software was developed by Ryan P. Adams, Michael Gelbart, and
# Jasper Snoek at Harvard University, Kevin Swersky at the
# University of Toronto (“Toronto”), and Hugo Larochelle at the
# Université de Sherbrooke (“Sherbrooke”), which assigned its rights
# in the Software to Socpra Sciences et Génie
# S.E.C. (“Socpra”). Pursuant to an inter-institutional agreement
# between the parties, it is distributed for free academic and
# non-commercial research use by the President and Fellows of Harvard
# College (“Harvard”).
#
# Using the Software indicates your agreement to be bound by the terms
# of this Software Use Agreement (“Agreement”). Absent your agreement
# to the terms below, you (the “End User”) have no rights to hold or
# use the Software whatsoever.
#
# Harvard agrees to grant hereunder the limited non-exclusive license
# to End User for the use of the Software in the performance of End
# User’s internal, non-commercial research and academic use at End
# User’s academic or not-for-profit research institution
# (“Institution”) on the following terms and conditions:
#
# 1.  NO REDISTRIBUTION. The Software remains the property Harvard,
# Toronto and Socpra, and except as set forth in Section 4, End User
# shall not publish, distribute, or otherwise transfer or make
# available the Software to any other party.
#
# 2.  NO COMMERCIAL USE. End User shall not use the Software for
# commercial purposes and any such use of the Software is expressly
# prohibited. This includes, but is not limited to, use of the
# Software in fee-for-service arrangements, core facilities or
# laboratories or to provide research services to (or in collaboration
# with) third parties for a fee, and in industry-sponsored
# collaborative research projects where any commercial rights are
# granted to the sponsor. If End User wishes to use the Software for
# commercial purposes or for any other restricted purpose, End User
# must execute a separate license agreement with Harvard.
#
# Requests for use of the Software for commercial purposes, please
# contact:
#
# Office of Technology Development
# Harvard University
# Smith Campus Center, Suite 727E
# 1350 Massachusetts Avenue
# Cambridge, MA 02138 USA
# Telephone: (617) 495-3067
# Facsimile: (617) 495-9568
# E-mail: otd@harvard.edu
#
# 3.  OWNERSHIP AND COPYRIGHT NOTICE. Harvard, Toronto and Socpra own
# all intellectual property in the Software. End User shall gain no
# ownership to the Software. End User shall not remove or delete and
# shall retain in the Software, in any modifications to Software and
# in any Derivative Works, the copyright, trademark, or other notices
# pertaining to Software as provided with the Software.
#
# 4.  DERIVATIVE WORKS. End User may create and use Derivative Works,
# as such term is defined under U.S. copyright laws, provided that any
# such Derivative Works shall be restricted to non-commercial,
# internal research and academic use at End User’s Institution. End
# User may distribute Derivative Works to other Institutions solely
# for the performance of non-commercial, internal research and
# academic use on terms substantially similar to this License and
# Terms of Use.
#
# 5.  FEEDBACK. In order to improve the Software, comments from End
# Users may be useful. End User agrees to provide Harvard with
# feedback on the End User’s use of the Software (e.g., any bugs in
# the Software, the user experience, etc.).  Harvard is permitted to
# use such information provided by End User in making changes and
# improvements to the Software without compensation or an accounting
# to End User.
#
# 6.  NON ASSERT. End User acknowledges that Harvard, Toronto and/or
# Sherbrooke or Socpra may develop modifications to the Software that
# may be based on the feedback provided by End User under Section 5
# above. Harvard, Toronto and Sherbrooke/Socpra shall not be
# restricted in any way by End User regarding their use of such
# information.  End User acknowledges the right of Harvard, Toronto
# and Sherbrooke/Socpra to prepare, publish, display, reproduce,
# transmit and or use modifications to the Software that may be
# substantially similar or functionally equivalent to End User’s
# modifications and/or improvements if any.  In the event that End
# User obtains patent protection for any modification or improvement
# to Software, End User agrees not to allege or enjoin infringement of
# End User’s patent against Harvard, Toronto or Sherbrooke or Socpra,
# or any of the researchers, medical or research staff, officers,
# directors and employees of those institutions.
#
# 7.  PUBLICATION & ATTRIBUTION. End User has the right to publish,
# present, or share results from the use of the Software.  In
# accordance with customary academic practice, End User will
# acknowledge Harvard, Toronto and Sherbrooke/Socpra as the providers
# of the Software and may cite the relevant reference(s) from the
# following list of publications:
#
# Practical Bayesian Optimization of Machine Learning Algorithms
# Jasper Snoek, Hugo Larochelle and Ryan Prescott Adams
# Neural Information Processing Systems, 2012
#
# Multi-Task Bayesian Optimization
# Kevin Swersky, Jasper Snoek and Ryan Prescott Adams
# Advances in Neural Information Processing Systems, 2013
#
# Input Warping for Bayesian Optimization of Non-stationary Functions
# Jasper Snoek, Kevin Swersky, Richard Zemel and Ryan Prescott Adams
# Preprint, arXiv:1402.0929, http://arxiv.org/abs/1402.0929, 2013
#
# Bayesian Optimization and Semiparametric Models with Applications to
# Assistive Technology Jasper Snoek, PhD Thesis, University of
# Toronto, 2013
#
# 8.  NO WARRANTIES. THE SOFTWARE IS PROVIDED "AS IS." TO THE FULLEST
# EXTENT PERMITTED BY LAW, HARVARD, TORONTO AND SHERBROOKE AND SOCPRA
# HEREBY DISCLAIM ALL WARRANTIES OF ANY KIND (EXPRESS, IMPLIED OR
# OTHERWISE) REGARDING THE SOFTWARE, INCLUDING BUT NOT LIMITED TO ANY
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OWNERSHIP, AND NON-INFRINGEMENT.  HARVARD, TORONTO AND
# SHERBROOKE AND SOCPRA MAKE NO WARRANTY ABOUT THE ACCURACY,
# RELIABILITY, COMPLETENESS, TIMELINESS, SUFFICIENCY OR QUALITY OF THE
# SOFTWARE.  HARVARD, TORONTO AND SHERBROOKE AND SOCPRA DO NOT WARRANT
# THAT THE SOFTWARE WILL OPERATE WITHOUT ERROR OR INTERRUPTION.
#
# 9.  LIMITATIONS OF LIABILITY AND REMEDIES. USE OF THE SOFTWARE IS AT
# END USER’S OWN RISK. IF END USER IS DISSATISFIED WITH THE SOFTWARE,
# ITS EXCLUSIVE REMEDY IS TO STOP USING IT.  IN NO EVENT SHALL
# HARVARD, TORONTO OR SHERBROOKE OR SOCPRA BE LIABLE TO END USER OR
# ITS INSTITUTION, IN CONTRACT, TORT OR OTHERWISE, FOR ANY DIRECT,
# INDIRECT, SPECIAL, INCIDENTAL, CONSEQUENTIAL, PUNITIVE OR OTHER
# DAMAGES OF ANY KIND WHATSOEVER ARISING OUT OF OR IN CONNECTION WITH
# THE SOFTWARE, EVEN IF HARVARD, TORONTO OR SHERBROOKE OR SOCPRA IS
# NEGLIGENT OR OTHERWISE AT FAULT, AND REGARDLESS OF WHETHER HARVARD,
# TORONTO OR SHERBROOKE OR SOCPRA IS ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGES.
#
# 10. INDEMNIFICATION. To the extent permitted by law, End User shall
# indemnify, defend and hold harmless Harvard, Toronto and Sherbrooke
# and Socpra, their corporate affiliates, current or future directors,
# trustees, officers, faculty, medical and professional staff,
# employees, students and agents and their respective successors,
# heirs and assigns (the "Indemnitees"), against any liability,
# damage, loss or expense (including reasonable attorney's fees and
# expenses of litigation) incurred by or imposed upon the Indemnitees
# or any one of them in connection with any claims, suits, actions,
# demands or judgments arising from End User’s breach of this
# Agreement or its Institution’s use of the Software except to the
# extent caused by the gross negligence or willful misconduct of
# Harvard, Toronto or Sherbrooke or Socpra. This indemnification
# provision shall survive expiration or termination of this Agreement.
#
# 11. GOVERNING LAW. This Agreement shall be construed and governed by
# the laws of the Commonwealth of Massachusetts regardless of
# otherwise applicable choice of law standards.
#
# 12. NON-USE OF NAME.  Nothing in this License and Terms of Use shall
# be construed as granting End Users or their Institutions any rights
# or licenses to use any trademarks, service marks or logos associated
# with the Software.  You may not use the terms “Harvard” or
# “University of Toronto” or “Université de Sherbrooke” or “Socpra
# Sciences et Génie S.E.C.” (or a substantially similar term) in any
# way that is inconsistent with the permitted uses described
# herein. You agree not to use any name or emblem of Harvard, Toronto
# or Sherbrooke, or any of their subdivisions for any purpose, or to
# falsely suggest any relationship between End User (or its
# Institution) and Harvard, Toronto and/or Sherbrooke, or in any
# manner that would infringe or violate any of their rights.
#
# 13. End User represents and warrants that it has the legal authority
# to enter into this License and Terms of Use on behalf of itself and
# its Institution.
from __future__ import print_function
from time import time
import csv
import sys
import optparse
import importlib
import time
import os

import glob
import numpy as np

try: import simplejson as json
except ImportError: import json

from collections import OrderedDict

# from spearmint.utils.database.mongodb import MongoDB
from spearmint.tasks.task_group       import TaskGroup
from modified_resource import *
from modified_resource import parse_resources_from_config
from modified_resource import print_resources_status

from spearmint.utils.parsing import parse_db_address

from MyDB import MyDB
import platform



def get_options(dir):
    parser = optparse.OptionParser(usage="usage: %prog [options] directory")

    parser.add_option("--config", dest="config_file",
                      help="Configuration file name.",
                      type="string", default="config.json")

    (commandline_kwargs, args) = parser.parse_args()

    # Read in the config file
    expt_dir = dir
    if not os.path.isdir(expt_dir):
        raise Exception("Cannot find directory %s" % expt_dir)
    expt_file = os.path.join(expt_dir, commandline_kwargs.config_file)

    try:
        with open(expt_file, 'r') as f:
            options = json.load(f, object_pairs_hook=OrderedDict)
    except:
        raise Exception("config.json did not load properly. Perhaps a spurious comma?")
    options["config"]  = commandline_kwargs.config_file


    # Set sensible defaults for options
    options['chooser']  = options.get('chooser', 'default_chooser')
    if 'tasks' not in options:
        options['tasks'] = {'main' : {'type' : 'OBJECTIVE', 'likelihood' : options.get('likelihood', 'GAUSSIAN')}}

    # Set DB address
    db_address = parse_db_address(options)
    if 'database' not in options:
        options['database'] = {'name': 'spearmint', 'address': db_address}
    else:
        options['database']['address'] = db_address

    if not os.path.exists(expt_dir):
        sys.stderr.write("Cannot find experiment directory '%s'. "
                         "Aborting.\n" % (expt_dir))
        sys.exit(-1)

    return options, expt_dir





def _gp_main(runs, maxeval, dir,fun_index,dim):



    options, expt_dir = get_options(dir)

    resources = parse_resources_from_config(options)
    # Add the maxeval


    resources["my-machine"].max_finished_jobs = maxeval
    # Load up the chooser.
    chooser_module = importlib.import_module('spearmint.choosers.' + options['chooser'])
    #chooser = chooser_module.init(options)
    experiment_name = options.get("experiment-name", 'unnamed-experiment')
    experiment_name = 'Experiment_Name/' + experiment_name + '/' + str(run_index + 1)

    script_dirx = os.path.dirname(__file__)
    results_dirx = os.path.join(script_dirx, experiment_name)
    if not os.path.isdir(results_dirx):
        os.makedirs(results_dirx)
    #define
    history_best = np.zeros([runs,maxeval])
    best_val = np.zeros([runs,1])
    previous_pos = np.zeros([runs,maxeval,dim])
    previous_val = np.zeros([runs,maxeval])
    t_s = time.time()
    for k in range(runs):
        #Initialize the chooser
        chooser = chooser_module.init(options)
        #Set database
        db = MyDB(db_name=experiment_name)
        db_address='localhost'

        for resource_name, resource in resources.iteritems():

            print(experiment_name)
            jobs = load_jobs(db, experiment_name)

            #remove the jobs in the previous run
            remove_jobs_in_last_run(db, jobs, experiment_name)

            # If the resource is currently accepting more jobs
            # TODO: here cost will eventually also be considered: even if the
            #       resource is not full, we might wait because of cost incurred
            # Note: I chose to fill up one resource and them move on to the next
            # You could also do it the other way, by changing "while" to "if" here
            FEs = 0

            while FEs < maxeval:

                # Load jobs from DB
                # (move out of one or both loops?) would need to pass into load_tasks
                jobs = load_jobs(db, experiment_name)

                # Remove any broken jobs from pending.
                remove_broken_jobs(db, jobs, experiment_name, resources)

                # Get a suggestion for the next job
                suggested_job, current_best,para_input= get_suggestion(chooser, resource.tasks, db, expt_dir, options, resource_name,fun_index,dim)
                sys.stderr.write('current best %s.\n' % current_best)
                if FEs < maxeval:
                    history_best[k,FEs]=current_best
                # Submit the job to the appropriate resource
                sucess, suggested_job= resource.attemptDispatch(experiment_name, suggested_job,db_address, expt_dir,para_input)
                # Set the status of the job appropriately (successfully submitted or not)
                if not sucess:
                    suggested_job['status'] = 'broken'
                else:
                    suggested_job['status'] = 'complete'
                _job_id = suggested_job['id']
                save_job(suggested_job, db, experiment_name,_job_id)

                jobs = load_jobs(db, experiment_name)

                # Print out the status of the resources
                # resource.printStatus(jobs)
                print_resources_status(resources.values(), jobs)
                FEs = FEs +1

            # If no resources are accepting jobs, sleep
            # (they might be accepting if suggest takes a while and so some jobs already finished by the time this point is reached)

            if tired(db, experiment_name, resources):
                time.sleep(options.get('polling-time', 0.05))

            print('*******************',experiment_name,'********************************')
            print('******************',str(k+1),'th run is finished!*********************')
            print('**********************************************************************')
            print('history best solutions are:', history_best)


        best_val[k,0] = current_best
        previous_pos[k,:,:] = np.array(chooser.task_group.inputs)
        previous_val[k,:] = np.array(chooser.task_group.values[chooser.objective['name']])
    t_e = time.time()
    time_cost = (t_e - t_s)/runs


    return best_val, time_cost, history_best, previous_pos, previous_val


def gp_main(runs, maxeval, dir,fun_index,dim,run_index):



    options, expt_dir = get_options(dir)

    resources = parse_resources_from_config(options)
    # Add the maxeval


    resources["my-machine"].max_finished_jobs = maxeval
    # Load up the chooser.
    chooser_module = importlib.import_module('spearmint.choosers.' + options['chooser'])
    #chooser = chooser_module.init(options)
    experiment_name = options.get("experiment-name", 'unnamed-experiment')
    experiment_name ='Experiment_Name/'+experiment_name +'/'+str(run_index+1)

    script_dirx = os.path.dirname(__file__)
    results_dirx = os.path.join(script_dirx, experiment_name)
    if not os.path.isdir(results_dirx):
        os.makedirs(results_dirx)
    #define
    history_best = np.zeros([runs,maxeval])
    best_val = np.zeros([runs,1])
    previous_pos = np.zeros([runs,maxeval,dim])
    previous_val = np.zeros([runs,maxeval])

    for k in range(runs):
        #Initialize the chooser
        chooser = chooser_module.init(options)
        #Set database
        db = MyDB(db_name=experiment_name)
        db_address='localhost'

        for resource_name, resource in resources.iteritems():

            jobs = load_jobs(db, experiment_name)
            #remove the jobs in the previous run
            remove_jobs_in_last_run(db, jobs, experiment_name,maxeval)
            comput_cost=load_comput_cost(db,experiment_name)


            FEs = len(jobs)
            if FEs >= maxeval:
                sys.exit()
            t_s = time.time()-comput_cost['cost']
            while FEs <= maxeval:

                # Load jobs from DB
                # (move out of one or both loops?) would need to pass into load_tasks
                job=dict()
                jobs = load_jobs(db, experiment_name)

                # Remove any broken jobs from pending.
                remove_broken_jobs(db, jobs, experiment_name, resources)

                for key,value in enumerate(jobs):
                    job[key+1]=value
                # Get a suggestion for the next job
                JOB,suggested_job, current_best,para_input= get_suggestion(chooser, resource.tasks, db, expt_dir, options, resource_name,fun_index,dim,job,jobs,FEs,experiment_name)
                sys.stderr.write('current best %s.\n' % current_best)
                if FEs < maxeval:
                    history_best[k,FEs]=current_best
                # Submit the job to the appropriate resource
                sucess, suggested_job= resource.attemptDispatch(experiment_name, suggested_job,db_address, expt_dir,para_input)
                # Set the status of the job appropriately (successfully submitted or not)
                if not sucess:
                    suggested_job['status'] = 'broken'
                else:
                    suggested_job['status'] = 'complete'
                _job_id = suggested_job['id']

                JOB[_job_id]=suggested_job

                save_job(JOB, db, experiment_name)
                comput_cost['cost']=time.time()-t_s
                save_comput_cost(comput_cost, db, experiment_name)

                #jobs = load_jobs(db, experiment_name)

                FEs = FEs +1
                print("current job id", str(FEs),"is finished!")


            print('*******************',experiment_name,'********************************')
            print('******************',str(k+1),'th run is finished!*********************')
            print('**********************************************************************')
            print('history best solutions are:', history_best)


        best_val[k,0] = current_best
        previous_pos[k,:,:] = np.array(chooser.task_group.inputs)
        previous_val[k,:] = np.array(chooser.task_group.values[chooser.objective['name']])
        Hb=np.inf
        for t in range(maxeval):
            if Hb > previous_val[k,t]:
                Hb=previous_val[k,t]
            history_best[k,t]=Hb




    return best_val, comput_cost['cost'], history_best, previous_pos, previous_val


def tired(db, experiment_name, resources):
    """
    return True if no resources are accepting jobs
    """
    jobs = load_jobs(db, experiment_name)
    for resource_name, resource in resources.iteritems():
        if resource.acceptingJobs(jobs):
            return False
    return True

def remove_jobs_in_last_run(db, jobs, experiment_name,maxeval):
    """
    In order to run multiple runs, for the jobs in the last run, their status will be set
    to 'broken'
    """
    #db.clean_all_db()
    # if jobs:
    #     for job in jobs:
    #         job['status'] = 'broken'
    #         s = 'jobs' + str(job['job_id'])
    #         save_job(job, db, experiment_name,s)
    #         db.remove(experiment_name,s)


def remove_broken_jobs(db, jobs, experiment_name, resources):
    """
    Look through jobs and for those that are pending but not alive, set
    their status to 'broken'
    """
    if jobs:
        for job in jobs:
            if job['status'] == 'pending':
                if not resources[job['resource']].isJobAlive(job):
                    sys.stderr.write('Broken job %s detected.\n' % job['id'])
                    job['status'] = 'broken'
                    db.remove(experiment_name, 'jobs')

# TODO: support decoupling i.e. task_names containing more than one task,
#       and the chooser must choose between them in addition to choosing X
def get_suggestion(chooser, task_names, db, expt_dir, options, resource_name,fun_index,dim,job,jobs,FEs,experiment_name):

    if len(task_names) == 0:
        raise Exception(" Error: trying to obtain suggestion for 0 tasks ")

    #experiment_name = options['experiment-name']+'_'+str(run_index+1)

    # We are only interested in the tasks in task_names
    task_options = { task: options["tasks"][task] for task in task_names }
    # For now we aren't doing any multi-task, so the below is simpler
    # task_options = options["tasks"]

    # Load the tasks from the database -- only those in task_names!
    task_group = load_task_group(db, options,experiment_name, task_names)

    # Load the model hypers from the database.
    hypers = load_hypers(db, experiment_name)

    # "Fit" the chooser - give the chooser data and let it fit the model.
    hypers = chooser.fit(task_group, hypers, task_options)

    # Save the hyperparameters to the database.
    save_hypers(hypers, db, experiment_name)

    # Ask the chooser to actually pick one.
    suggested_input = chooser.suggest()

    para_input = task_group.paramify(suggested_input)

    # TODO: implelent this
    suggested_task = task_names[0]  

    # Parse out the name of the main file (TODO: move this elsewhere)
    if "main-file" in task_options[suggested_task]:
        main_file = task_options[suggested_task]["main-file"]
    elif "main-file" in options:
        main_file = options['main-file']
    else:
        raise Exception("main-file not specified for task %s" % suggested_task)

    if "language" in task_options[suggested_task]:
        language = task_options[suggested_task]["language"]
    elif "language" in options:
        language = options['language']
    else:
        raise Exception("language not specified for task %s" % suggested_task)


    #jobs = load_jobs(db, experiment_name)

    job_id = FEs + 1

    #array to dict
    suggested_input_to_dct = {'shape': list(suggested_input.shape), 'dtype': str(suggested_input.dtype), 'lst_arr': suggested_input.flatten().tolist()}

    sub_job = {
        'id'          : job_id,
        'params'      : suggested_input_to_dct,
        'expt_dir'    : expt_dir,
        'tasks'       : task_names,
        'resource'    : resource_name,
        'main-file'   : main_file,
        'language'    : language,
        'status'      : 'new',
        'submit time' : time.time(),
        'start time'  : None,
        'end time'    : None,
        'fun_index'   : fun_index,
        'dim'         : dim


    }

    job[job_id]=sub_job

    save_job(job, db, experiment_name)

    return job,sub_job,chooser.current_best,para_input

def save_hypers(hypers, db, experiment_name):
    tmp = hypers.copy()

    if len(hypers) != 0:
        tmp['main']['hypers']['beta_alpha']=hypers['main']['hypers']['beta_alpha'].flatten().tolist()
        tmp['main']['hypers']['beta_beta'] = hypers['main']['hypers']['beta_beta'].flatten().tolist()
        tmp['main']['hypers']['ls'] = hypers['main']['hypers']['ls'].flatten().tolist()
    db.save(experiment_name, 'hypers',tmp)
    #db.print_all_db()
def load_hypers(db, experiment_name):

    hypers=db.load(experiment_name, 'hypers')
    if isinstance(hypers,dict) and len(hypers)!=0:
        tmp=hypers.copy()
        tmp['main']['hypers']['beta_alpha']=np.asarray(hypers['main']['hypers']['beta_alpha'])
        tmp['main']['hypers']['beta_beta'] = np.asarray(hypers['main']['hypers']['beta_beta'])
        tmp['main']['hypers']['ls'] = np.asarray(hypers['main']['hypers']['ls'])
    return hypers



def load_jobs(db, experiment_name):
    """load the jobs from the database
    jobs : list
        a list of jobs or an empty list
    """

    jobs = db.load(experiment_name, 'jobs')
    #db.print_all_db()
    if jobs is None:
        jobs = []
    if isinstance(jobs, dict):
        jobs = [jobs]

    return jobs

def save_job(job, db, experiment_name):
    """save a job to the database"""
    db.save(experiment_name, 'jobs',job)

def load_comput_cost(db, experiment_name):
    """load the jobs from the database
    jobs : list
        a list of jobs or an empty list
    """

    comput_cost = db.load(experiment_name, 'comput_cost')
    if comput_cost is None:
        comput_cost={'cost':0}


    return comput_cost

def save_comput_cost(comput_cost, db, experiment_name):
    """save a job to the database"""
    db.save(experiment_name, 'comput_cost',comput_cost)


def load_task_group(db, options,experiment_name, task_names=None):
    if task_names is None:
        task_names = options['tasks'].keys()
    task_options = { task: options["tasks"][task] for task in task_names }

    jobs = load_jobs(db, experiment_name)

    task_group = TaskGroup(task_options, options['variables'])

    if jobs:

        task_group.inputs  = np.array([job['params']['lst_arr']
                for job in jobs if job['status'] == 'complete'])
        # for k in enumerate(jobs):
        #     print(k)
        # print(task_group.inputs)

        task_group.pending = np.array([job['params']['lst_arr']
                for job in jobs if job['status'] == 'pending'])

        task_group.values  = {task : np.array([job['values'][task]
                for job in jobs if job['status'] == 'complete'])
                    for task in task_names}

        task_group.add_nan_task_if_nans()

        # TODO: record costs

    return task_group


# BROKEN
# def print_diagnostics(chooser):
#     sys.stderr.write("Optimizing over %d dimensions\n" % (expt_grid.vmap.cardinality))
#     best_val   = None
#     best_job   = None
#     best_input = None
#     if task.has_data():
#         best_input, best_val = chooser.get_best()
#         best_job = db.load(experiment_name, 'jobs', {'input' : best_input})
#
#         if best_job:
#             best_job = best_job[0]
#         else:
#             best_job
#             raise Warning('Job ID of best input/value pair not recorded.')
#
#     # Track the time series of optimization. This should eventually go into a diagnostics module.
#     trace_fh = open(os.path.join(expt_dir, 'trace.csv'), 'a')
#     trace_fh.write("%d,%f,%d,%d,%d,%d\n"
#                    % (time.time(), best_val, best_job,
#                       tasks.pending.shape[0], tasks.invalid.shape[0], tasks.data.shape[0]))
#     trace_fh.close()
#
#     # Print out the best job results
#     best_job_fh = open(os.path.join(expt_dir, 'best_job_and_result.txt'), 'w')
#     best_job_fh.write("Best result: %f\nJob-id: %d\nParameters: \n" %
#                       (best_val, best_job))
#
#     if best_input:
#         for name, params in task.get_params(best_input):
#             best_job_fh.write('%s: %s\n' % (name, params))
#
#     best_job_fh.close()

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
        13: 'Schoen-100k',
        14: 'BBOB_F15',
        15: 'BBOB_F16',
        16: 'BBOB_F17',
        17: 'BBOB_F18',
        18: 'BBOB_F19',
        19: 'BBOB_F20',
        20: 'BBOB_F21',
        21: 'BBOB_F22',
        22: 'BBOB_F23',
        23: 'BBOB_F24'

    }[function_index]






#def main(dim,fun_index,run_index):
def main(argv):

    if argv == None:
        raise Exception('Please input parameters!')
    else:
        dim = int(argv[1])
        fun_index = int(argv[2])
        run_index = int(argv[3])

    runs = 1

    maxeval = 50*dim

    # path=('C:\\Program Files\\MongoDB\\Server\\3.6\\bin')
    # os.chdir(path)
    # cmd = ('start/b mongod --logpath D:\\spearmint_win\\1.log --dbpath D:\\spearmint_win\\db\\')
    # os.system(cmd)
    # db = MyDB(db_name='rastrigin_5d')
    # db.print_all_db()

    for i in range(fun_index,fun_index+1):

        if platform.system()=="Windows":
            dir = "D:\\python_projects\\bbob_problems_from_taimoor\\spearmint_lmdb\\"+str(dim)+"_d_functions\\"+function_name(i + 1)+"_"+str(dim)+"d"
        else:
            dir = "/home/users/nus/iseyij/mywork/BBOB_from_taimoor/spearmint_lmdb/"+str(dim)+"_d_functions/"+function_name(i + 1)+"_"+str(dim)+"d"
        #call the gp_main function
        [best_val, average_time, bestCost, previous_pos, previous_val] = gp_main(runs, maxeval,dir,fun_index,dim,run_index)

        print(previous_pos)

        averageCost = np.zeros(maxeval)
        # record average mae mse and r


        for k in range(int(maxeval)):
            averageCost[k] = np.mean(bestCost[:, k])



        # save the statistics best worst median mean std
        solution_1 = bestCost[:, maxeval - 1]

        best_solution = np.min(solution_1)
        worst_solution = np.max(solution_1)
        median_solution = np.median(solution_1)
        mean_solution = np.mean(solution_1)
        std_solution = np.std(solution_1)

        filename1 = 'Result_Record/' + '/spearmint/' + str(dim) + '_dim/'+str(maxeval) + '_maxeval/'+'F'+str(i+1)+'/'+ str(run_index + 1)+'/'
        sample_file_name1 = 'time_cost' + '.csv'
        script_dir1 = os.path.dirname(__file__)
        results_dir1 = os.path.join(script_dir1, filename1)
        if not os.path.isdir(results_dir1):
            os.makedirs(results_dir1)
        filehead1 = ['', '', "spearmint"]
        a = open(results_dir1 + sample_file_name1, "wb")
        writer = csv.writer(a)
        writer.writerow(filehead1)
        a.close()
        s = 'F' + str(i + 1)
        t = [s, '', str(average_time)]
        with open(results_dir1 + sample_file_name1, "ab") as csvfile1:
            writer = csv.writer(csvfile1)
            writer.writerow(t)
            csvfile1.close()



        sample_file_name2 = 'statistics' + '.csv'
        script_dir2 = os.path.dirname(__file__)
        results_dir2 = os.path.join(script_dir2, filename1)
        if not os.path.isdir(results_dir2):
            os.makedirs(results_dir2)
        filehead2 = ['', '', "best","worst","median","mean","std"]
        a = open(results_dir2 + sample_file_name2, "wb")
        writer = csv.writer(a)
        writer.writerow(filehead2)
        a.close()
        s = 'F' + str(i + 1)
        t = [s, '', str(best_solution),str(worst_solution),str(median_solution),str(mean_solution),str(std_solution)]
        csvfile2 = open(results_dir2 + sample_file_name2, "ab")
        writer = csv.writer(csvfile2)
        writer.writerow(t)
        csvfile2.close()

        sample_file_name5 = 'Average_convergence_history_' + '.csv'
        script_dir5 = os.path.dirname(__file__)
        results_dir5 = os.path.join(script_dir5, filename1)
        if not os.path.isdir(results_dir5):
            os.makedirs(results_dir5)
        filehead5 = ['', '', "spearmint"]
        a = open(results_dir5 + sample_file_name5, "wb")
        writer = csv.writer(a)
        writer.writerow(filehead5)
        a.close()
        csvfile5 = open(results_dir5 + sample_file_name5, "ab")
        writer = csv.writer(csvfile5)
        writer.writerow([s])
        for item in range(maxeval):
            writer.writerow(['', str(item+1), str(averageCost[item])])
        csvfile5.close()


        sample_file_name6 = 'history_x_and_y' + '.csv'
        script_dir6 = os.path.dirname(__file__)
        results_dir6 = os.path.join(script_dir6, filename1)
        if not os.path.isdir(results_dir6):
            os.makedirs(results_dir6)
        filehead6 = ['', '', "spearmint"]
        a = open(results_dir6 + sample_file_name6, "wb")
        writer = csv.writer(a)
        writer.writerow(filehead6)
        a.close()
        csvfile6 = open(results_dir6 + sample_file_name6, "ab")
        writer = csv.writer(csvfile6)
        writer.writerow([s])
        for item in range(maxeval):
            tmp = [str(item+1),'']
            for h in range(dim):
                tmp.append(str(previous_pos[0,item,h]))
            tmp.append(str(previous_val[0,item]))
            writer.writerow(tmp)

        csvfile6.close()

def _main(dim,fun_index,run_index):


    runs = 1

    maxeval = 50*dim

    # path=('C:\\Program Files\\MongoDB\\Server\\3.6\\bin')
    # os.chdir(path)
    # cmd = ('start/b mongod --logpath D:\\spearmint_win\\1.log --dbpath D:\\spearmint_win\\db\\')
    # os.system(cmd)
    # db = MyDB(db_name='rastrigin_5d')
    # db.print_all_db()

    for i in range(fun_index, fun_index + 1):

        if platform.system() == "Windows":
            dir = "D:\\python_projects\\bbob_problems_from_taimoor\\spearmint_lmdb\\" + str(
                dim) + "_d_functions\\" + function_name(i + 1) + "_" + str(dim) + "d"
        else:
            dir = "/home/users/nus/iseyij/mywork/BBOB_from_taimoor/spearmint_lmdb/" + str(
                dim) + "_d_functions/" + function_name(i + 1) + "_" + str(dim) + "d"

        # call the gp_main function
        [best_val, average_time, bestCost, previous_pos, previous_val] = gp_main(runs, maxeval, dir, fun_index,
                                                                                 dim, run_index)

        print(previous_pos)

        averageCost = np.zeros(maxeval)
        # record average mae mse and r


        for k in range(int(maxeval)):
            averageCost[k] = np.mean(bestCost[:, k])

        # save the statistics best worst median mean std
        solution_1 = bestCost[:, maxeval - 1]

        best_solution = np.min(solution_1)
        worst_solution = np.max(solution_1)
        median_solution = np.median(solution_1)
        mean_solution = np.mean(solution_1)
        std_solution = np.std(solution_1)

        filename1 = 'Result_Record/' + '/spearmint/' + str(dim) + '_dim/' + str(
            maxeval) + '_maxeval/' + 'F' + str(i + 1) + '/' + str(run_index + 1) + '/'
        sample_file_name1 = 'time_cost' + '.csv'
        script_dir1 = os.path.dirname(__file__)
        results_dir1 = os.path.join(script_dir1, filename1)
        if not os.path.isdir(results_dir1):
            os.makedirs(results_dir1)
        filehead1 = ['', '', "spearmint"]
        a = open(results_dir1 + sample_file_name1, "wb")
        writer = csv.writer(a)
        writer.writerow(filehead1)
        a.close()
        s = 'F' + str(i + 1)
        t = [s, '', str(average_time)]
        with open(results_dir1 + sample_file_name1, "ab") as csvfile1:
            writer = csv.writer(csvfile1)
            writer.writerow(t)
            csvfile1.close()

        sample_file_name2 = 'statistics' + '.csv'
        script_dir2 = os.path.dirname(__file__)
        results_dir2 = os.path.join(script_dir2, filename1)
        if not os.path.isdir(results_dir2):
            os.makedirs(results_dir2)
        filehead2 = ['', '', "best", "worst", "median", "mean", "std"]
        a = open(results_dir2 + sample_file_name2, "wb")
        writer = csv.writer(a)
        writer.writerow(filehead2)
        a.close()
        s = 'F' + str(i + 1)
        t = [s, '', str(best_solution), str(worst_solution), str(median_solution), str(mean_solution),
             str(std_solution)]
        csvfile2 = open(results_dir2 + sample_file_name2, "ab")
        writer = csv.writer(csvfile2)
        writer.writerow(t)
        csvfile2.close()

        sample_file_name5 = 'Average_convergence_history' + '.csv'
        script_dir5 = os.path.dirname(__file__)
        results_dir5 = os.path.join(script_dir5, filename1)
        if not os.path.isdir(results_dir5):
            os.makedirs(results_dir5)
        filehead5 = ['', '', "spearmint"]
        a = open(results_dir5 + sample_file_name5, "wb")
        writer = csv.writer(a)
        writer.writerow(filehead5)
        a.close()
        csvfile5 = open(results_dir5 + sample_file_name5, "ab")
        writer = csv.writer(csvfile5)
        writer.writerow([s])
        for item in range(maxeval):
            writer.writerow(['', str(item + 1), str(averageCost[item])])
        csvfile5.close()

        sample_file_name6 = 'history_x_and_y' + '.csv'
        script_dir6 = os.path.dirname(__file__)
        results_dir6 = os.path.join(script_dir6, filename1)
        if not os.path.isdir(results_dir6):
            os.makedirs(results_dir6)
        filehead6 = ['', '', "spearmint"]
        a = open(results_dir6 + sample_file_name6, "wb")
        writer = csv.writer(a)
        writer.writerow(filehead6)
        a.close()
        csvfile6 = open(results_dir6 + sample_file_name6, "ab")
        writer = csv.writer(csvfile6)
        writer.writerow([s])
        for item in range(maxeval):
            tmp = [str(item + 1), '']
            for h in range(dim):
                tmp.append(str(previous_pos[0, item, h]))
            tmp.append(str(previous_val[0, item]))
            writer.writerow(tmp)

        csvfile6.close()

    # path=('C:\\Program Files\\MongoDB\\Server\\3.6\\bin')
    # os.chdir(path)
    # cmd = ('mongo admin --eval "db.shutdownServer()"')
    # os.system(cmd)


# if __name__ == '__main__':
#     dim =5
#     for fun_index in range(1):
#         for run_index in range(1):
#              main(dim,fun_index,run_index)
# if __name__ =='__main__':
#     main(sys.argv)
#     #main(dim=5,fun_index=2,run_index=0,acq_int=0)


if __name__ == '__main__':
    type='need_type_input'
    if type=='need_type_input':
        main(sys.argv)
    else:
        dim =5
        for fun_index in range(13,14):
            for run_index in range(1):
                 _main(dim,fun_index,run_index)
