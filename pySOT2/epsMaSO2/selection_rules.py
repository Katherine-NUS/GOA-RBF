"""
.. module:: selection_rules
   :synopsis: Acquisition functions / merit rules for selecting new points from candidates

.. moduleauthor:: Taimoor Akhtar <erita@nus.edu.sg>

"""

import scipy.stats as stats
import types
from .mo_utils import *
import random
from .hv import HyperVolume
import numpy as np
import time

INF = float('inf')

class MultiRuleSelection(object):
    """ This is a multi-rule selection methodology for cycling
        between different rules.
    """
    def __init__(self, data, epsilons):
        self.data = data
        self.epsilons = epsilons

        self.new_points = None
        self.rule_order = None
        self.nrules = 4
        self.prev_contribution = [0] * self.nrules
        self.total_contribution = [1] * self.nrules
        self.count = 0

        self.failcout = 0

        self.rules = []
        self.rules.append(EpsilonSelection(self.data, self.epsilons))
        self.rules.append(OspaceDistanceSelection(self.data))
        self.rules.append(DspaceDistanceSelection(self.data))
        self.rules.append(IntegratedSelection(self.data, self.epsilons))

    def select_points(self, xcand_nd, fhvals_nd, front, proposed_points, indices, npts):
        #'''
        self.rule_order = []

        if len(xcand_nd) >= 1:
            self.rule_order.append(0)
            indices = self.rules[0].select_points(xcand_nd = np.copy(xcand_nd),
                                                  fhvals_nd = np.copy(fhvals_nd),
                                                  front = np.copy(front), proposed_points = np.copy(proposed_points),
                                                  indices = indices, rate = 0.9)

        if len(xcand_nd) >= 2:
            self.rule_order.append(1)
            indices = self.rules[1].select_points(xcand_nd = np.copy(xcand_nd),
                                                  fhvals_nd = np.copy(fhvals_nd),
                                                  front = np.copy(front), proposed_points = np.copy(proposed_points),
                                                  indices = indices)

        if len(xcand_nd) >= 3:
            self.rule_order.append(2)
            indices = self.rules[2].select_points(xcand_nd = np.copy(xcand_nd),
                                                  fhvals_nd = np.copy(fhvals_nd),
                                                  front = np.copy(front), proposed_points = np.copy(proposed_points),
                                                  indices = indices)

        self.rule_order.append(3)

        #'''

        '''
        self.rule_order = []
        self.rule_order.append(0)
        self.rule_order.append(1)
        self.rule_order.append(2)
        self.rule_order.append(3)
        self.rule_order.append(4)
        indices = self.rules[3].select_points(xcand_nd = np.copy(xcand_nd),
                                              fhvals_nd = np.copy(fhvals_nd),
                                              front = np.copy(front), proposed_points = np.copy(proposed_points),
                                              indices = indices, rate = 0.6, npts = 5)
        '''

        return indices



class HyperVolumeSelection(object):
    """ This is the rule for hypervolume based selection of new points
    """
    def __init__(self, data):
        """
        :param data:
        :param npts:
        """
        self.data = data

    def select_points(self, front, xcand_nd, fhvals_nd, indices = None, npts = 1):

        # Use hypervolume contribution to select the next best
        # Step 1 - Normalize Objectives
        (M, l) = xcand_nd.shape
        temp_all = np.vstack((fhvals_nd, front))
        minpt = np.zeros(self.data.nobj)
        maxpt = np.zeros(self.data.nobj)
        for i in range(self.data.nobj):
            minpt[i] = np.min(temp_all[:,i])
            maxpt[i] = np.max(temp_all[:,i])
        normalized_front = np.asarray(normalize_objectives(front, minpt, maxpt))
        (N, temp) = normalized_front.shape
        normalized_cand_fh = np.asarray(normalize_objectives(fhvals_nd.tolist(), minpt, maxpt))

        # Step 2 - Make sure points already selected are not included in new points list
        if indices is not None:
            nd = range(N)
            dominated = []
            fvals = []
            for index in indices:
                fvals = np.vstack((normalized_front, normalized_cand_fh[index,:]))
                (nd, dominated) = ND_Add(np.transpose(fvals), dominated, nd)
            normalized_front = fvals[nd,:]
            N = len(nd)

        # Step 3 - Compute Hypervolume Contribution
        hv = HyperVolume(1.1*np.ones(self.data.nobj))
        xnew = np.zeros((npts, l))
        if indices is None:
            indices = []
        hv_vals = -1*np.ones(M)
        hv_vals[indices] = -2
        for j in range(npts):
            # 3.1 - Find point with best HV improvement
            base_hv = hv.compute(normalized_front)
            for i in range(M):
                if hv_vals[i] != 0 and hv_vals[i] != -2:
                    nd = range(N)
                    dominated = []
                    fvals = np.vstack((normalized_front, normalized_cand_fh[i,:]))
                    (nd, dominated) = ND_Add(np.transpose(fvals), dominated, nd)
                    if dominated and dominated[0] == N: # Record is dominated
                        hv_vals[i] = 0
                    else:
                        new_hv = hv.compute(fvals[nd,:])
                        hv_vals[i] = new_hv - base_hv
            # vals = np.zeros((M,2))
            # vals[:,0] = xcand_nd[:,0]
            # vals[:,1] = hv_vals
            # print(vals)
            # 3.2 - Update selected candidate list
            index = np.argmax(hv_vals)
            xnew[j,:] = xcand_nd[index,:]
            indices.append(index)
            # 3.3 - Bar point from future selection and update non-dominated set
            hv_vals[index] = -2
            nd = range(N)
            dominated = []
            fvals = np.vstack((normalized_front, normalized_cand_fh[index,:]))
            (nd, dominated) = ND_Add(np.transpose(fvals), dominated, nd)
            normalized_front = fvals[nd,:]
            N = len(nd)
        return indices


class DspaceDistanceSelection(object):
    """
    Implementation of the Decision-Space Selection
    Rule in GOMORS that chooses new points based
    on max-min decision space distance from
    evaluated points
    """
    def __init__(self, data):
        """
        :param data:
        :param npts:
        """
        self.data = data

    def select_points(self, xcand_nd, fhvals_nd, front, proposed_points, indices = None, npts = 1, rate = 0.75):
        if indices is not None:
            selected_points = np.vstack((proposed_points, xcand_nd[indices,:]))
        else:
            selected_points = np.copy(proposed_points)
        xnew = np.zeros((npts, self.data.dim))
        for i in range(npts):
            dists = scp.distance.cdist(xcand_nd, selected_points)
            dmerit = np.amin(np.asmatrix(dists), axis=1)
            if indices is not None:
                dmerit[indices] = -1
            index = np.argmax(dmerit)
            if indices is None:
                indices = []
            indices.append(index)
            xnew[i,:] = xcand_nd[index,:]
            selected_points = np.vstack((selected_points, xnew[i,:]))
        return indices


class OspaceDistanceSelection(object):
    """
    Implementation of the Objective-Space Selection
    Rule in GOMORS that chooses new points based
    on max-min approximate obj space distance from
    evaluated points
    """
    def __init__(self, data):
        """
        :param data:
        :param npts:
        """
        self.data = data

    def select_points(self, xcand_nd, fhvals_nd, front, proposed_points = None, indices = None, npts = 1, rate = 0.75):

        # Step 1 - Normalize Objectives

        temp_all = np.vstack((fhvals_nd, front))
        minpt = np.zeros(self.data.nobj)
        maxpt = np.zeros(self.data.nobj)
        for i in range(self.data.nobj):
            minpt[i] = np.min(temp_all[:,i])
            maxpt[i] = np.max(temp_all[:,i])
        #normalized_fvals = np.asarray(normalize_objectives(front, minpt, maxpt))
        normalized_fvals = np.copy(front)
        #normalized_cand_fh = np.asarray(normalize_objectives(fhvals_nd.tolist(), minpt, maxpt))
        normalized_cand_fh = np.copy(fhvals_nd)

        # Step 2 - Make sure points already selected are not included in new points list
        if indices is not None:
            selected_fvals = np.vstack((normalized_fvals, normalized_cand_fh[indices,:]))
        else:
            selected_fvals = np.copy(normalized_fvals)

        # Step 3 - Find point(s) with max-min distance in objective space
        dists = scp.distance.cdist(normalized_cand_fh, selected_fvals)
        dmerit = np.amin(np.asmatrix(dists), axis=1)
        xnew = np.zeros((npts, self.data.dim))
        for i in range(npts):
            if indices is not None:
                dmerit[indices] = -1
            index = np.argmax(dmerit)
            if indices is None:
                indices = []
            indices.append(index)
            xnew[i,:] = xcand_nd[index,:]
            selected_fvals = np.vstack((selected_fvals, normalized_cand_fh[index,:]))
        return indices

class EpsilonSelection(object):
    """ This is the rule for epsilon-progress based selection of new points
    """
    def __init__(self, data, epsilon):
        """
        :param data:
        :param npts:
        """
        self.data = data
        self.epsilon = epsilon

    def select_points(self, xcand_nd, fhvals_nd, front, proposed_points = None, indices = None, npts = 1, rate = 0.75):
        # Randomly select a point from points with epsilon progress
        (M, l) = xcand_nd.shape
        (N, l) = front.shape
        # Step 1 - Add older points already selected to the eps_front
        if indices is not None:
            ndf_index = range(N)
            df_index = []
            box_index = []
            fvals = None
            for index in indices:
                fvals = np.vstack((front, fhvals_nd[index,:]))
                (ndf_index, df_index, box_index, F_box) = epsilon_ND_Add(np.transpose(fvals), df_index, ndf_index, box_index, self.epsilon)
            front = fvals[ndf_index,:]
            N = len(ndf_index)

        # Step 2 - Check if there is Epsilon Progress and add those points to a list
        max_step = 0

        if indices is None:
            indices = []

        steps = [-1] * M

        print('Number of Cand = {}, Number of eND Points = {}'.format(M, N))

        for i in range(M):
            if i not in indices:
                point = np.asmatrix(np.copy(fhvals_nd[i, :]))
                boxA = np.copy(np.floor(point / self.epsilon))

                delta = INF
                for j in range(N):
                    boxB = np.copy(np.floor(front[j, :] / self.epsilon))
                    temp_delta = np.amax(np.subtract(boxB, boxA))
                    if delta > temp_delta:
                        delta = temp_delta
                steps[i] = delta

                if steps[i] > max_step:
                    max_step = steps[i]

        ep_indices = []
        for i in range(M):
            if steps[i] >= rate * max_step:
                ep_indices.append(i)

        for j in range(npts):
            if ep_indices:
                index = random.randint(0, len(ep_indices)-1)
                print('max step = {}, threshold = {}, promising = {}'.format(max_step, rate * max_step, len(ep_indices)))
                indices.append(ep_indices[index])
                ep_indices.remove(ep_indices[index])
            else:
                index = random.randint(0, M-1)
                while index in indices:
                    index = random.randint(0, M-1)
                indices.append(index)
        return indices



class IntegratedSelection(object):
    """ This is the rule for epsilon-progress based selection of new points
    """
    def __init__(self, data, epsilon):
        """
        :param data:
        :param npts:
        """
        self.data = data
        self.epsilon = epsilon

    def select_points(self, xcand_nd, fhvals_nd, front, proposed_points, indices = None, npts = 1, rate = 0.75):
        # Randomly select a point from points with epsilon progress
        (M, l) = xcand_nd.shape
        (N, l) = front.shape
        # Step 1 - Add older points already selected to the eps_front
        if indices is not None:
            ndf_index = range(N)
            df_index = []
            box_index = []
            fvals = None
            for index in indices:
                fvals = np.vstack((front, fhvals_nd[index,:]))
                (ndf_index, df_index, box_index, F_box) = epsilon_ND_Add(np.transpose(fvals), df_index, ndf_index, box_index, self.epsilon)
            front = fvals[ndf_index,:]
            N = len(ndf_index)

        # Step 2 - Check if there is Epsilon Progress and add those points to a list
        max_step = 0

        if indices is None:
            indices = []

        steps = [-1] * M

        print('Number of Cand = {}, Number of eND Points = {}'.format(M, N))

        for i in range(M):
            if i not in indices:
                point = np.asmatrix(np.copy(fhvals_nd[i, :]))
                boxA = np.copy(np.floor(point / self.epsilon))

                delta = INF
                for j in range(N):
                    boxB = np.copy(np.floor(front[j, :] / self.epsilon))
                    temp_delta = np.amax(np.subtract(boxB, boxA))
                    if delta > temp_delta:
                        delta = temp_delta
                steps[i] = delta

                if steps[i] > max_step:
                    max_step = steps[i]

        temp_all = np.vstack((fhvals_nd, front))
        minpt = np.zeros(self.data.nobj)
        maxpt = np.zeros(self.data.nobj)
        for i in range(self.data.nobj):
            minpt[i] = np.min(temp_all[:, i])
            maxpt[i] = np.max(temp_all[:, i])
        # normalized_fvals = np.asarray(normalize_objectives(front, minpt, maxpt))
        normalized_fvals = np.copy(front)
        # normalized_cand_fh = np.asarray(normalize_objectives(fhvals_nd.tolist(), minpt, maxpt))
        normalized_cand_fh = np.copy(fhvals_nd)

        rnpts = min(M, npts)
        weights = [0.5] * rnpts
        '''
        weights = []
        for i in range(rnpts - 1):
            weights.append(float(i) / float(rnpts - 1))
        weights.append(1.0)
        '''

        npt = 0
        while npt < rnpts:
            if indices:
                selected_fvals = normalized_cand_fh[indices, :]
                dists = scp.distance.cdist(normalized_cand_fh, selected_fvals)
                dmerit = np.amin(np.asmatrix(dists), axis = 1)
                dmerit[indices] = -1
                max_distance = np.amax(dmerit)
            else:
                dmerit = [0] * M
                max_distance = 1


            integrated_value = [0] * M
            for i in range(M):
                integrated_value[i] = weights[npt] * float(steps[i]) / float(max_step) + (1 - weights[npt]) * float(dmerit[i]) / float(max_distance)

            index = np.argmax(integrated_value)
            indices.append(index)
            npt += 1

        return indices
