"""
.. module:: merit_functions
   :synopsis: Merit functions for the adaptive sampling

.. moduleauthor:: David Eriksson <dme65@cornell.edu>,
                David Bindel <bindel@cornell.edu>

:Module: merit_functions
:Author: David Eriksson <dme65@cornell.edu>,
        David Bindel <bindel@cornell.edu>

"""

from ..pySOT1.utils import *
import scipy.spatial as scp
from scipy.stats import norm

def candidate_merit_weighted_distance(cand, npts=1):
    """Weighted distance merit function for the candidate points based methods

    :param cand: Candidate point object
    :type cand: Object
    :param npts: Number of points selected for evaluation
    :type npts: int

    :return: Points selected for evaluation, of size npts x dim
    :rtype: numpy.array
    """

    new_points = np.ones((npts,  cand.data.dim))

    for i in range(npts):
        ii = cand.next_weight
        weight = cand.weights[(ii+len(cand.weights)) % len(cand.weights)]
        merit = weight*cand.fhvals + \
            (1-weight)*(1.0 - unit_rescale(cand.dmerit))

        merit[cand.dmerit < cand.dtol] = np.inf
        jj = np.argmin(merit)
        cand.fhvals[jj] = np.inf
        new_points[i, :] = cand.xcand[jj, :]

        # Update distances and weights
        ds = scp.distance.cdist(cand.xcand, np.atleast_2d(new_points[i, :]))
        cand.dmerit = np.minimum(cand.dmerit, ds)
        cand.next_weight += 1

    return new_points

def candidate_top_points(cand, npts=1):
    """Weighted distance merit function for the candidate points based methods

    :param cand: Candidate point object
    :type cand: Object
    :param npts: Number of points selected for evaluation
    :type npts: int

    :return: Points selected for evaluation, of size npts x dim
    :rtype: numpy.array
    """

    new_points = np.ones([20,cand.data.dim])

    for i in range(npts):
        ii = cand.next_weight
        weight = cand.weights[(ii+len(cand.weights)) % len(cand.weights)]
        merit = weight*cand.fhvals + \
            (1-weight)*(1.0 - unit_rescale(cand.dmerit))

        merit[cand.dmerit < cand.dtol] = np.inf
        ss=np.argsort(merit.T)

        for k in range(20):
            new_points[k,:] = cand.xcand[ss[0,k], :]

        # Update distances and weights


    return new_points




def candidate_spearmint(cand, npts=1,top_num=20):
    """Weighted distance merit function for the candidate points based methods

    :param cand: Candidate point object
    :type cand: Object
    :param npts: Number of points selected for evaluation
    :type npts: int

    :return: Points selected for evaluation, of size npts x dim
    :rtype: numpy.array
    """
    top = top_num
    new_points = np.ones((top,  cand.data.dim))
    #select the top 20 points first
    for i in range(top):
        ii = cand.next_weight
        weight = cand.weights[(ii+len(cand.weights)) % len(cand.weights)]
        merit = weight*cand.fhvals + \
            (1-weight)*(1.0 - unit_rescale(cand.dmerit))

        merit[cand.dmerit < cand.dtol] = np.inf
        jj = np.argmin(merit)
        cand.fhvals[jj] = np.inf
        new_points[i, :] = cand.xcand[jj, :]

        # Update distances and weights
        ds = scp.distance.cdist(cand.xcand, np.atleast_2d(new_points[i, :]))
        cand.dmerit = np.minimum(cand.dmerit, ds)
        cand.next_weight += 1

    return new_points

def candidate_EI(cand, npts=1):
    """EI merit function for the candidate points based methods
    :param cand: Candidate point object
    :param npts: Number of points selected for evaluation

    :return: Points selected for evaluation
    """

    new_points = np.ones((npts,  cand.data.dim))
    y_min = cand.fbest

    for i in range(npts):
        y_n = (y_min-cand.fhvals)/np.sqrt(cand.shvals)  #normalization
        ei = (y_min-cand.fhvals)*norm.cdf(y_n)+np.sqrt(cand.shvals)*norm.pdf(y_n)

        jj = np.argmax(ei)
        
        cand.fhvals[jj] = -np.inf
        new_points[i, :] = cand.xcand[jj, :]

        # Update distances and weights

    return new_points


