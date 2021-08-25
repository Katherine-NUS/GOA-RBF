import numpy as np
import scipy.spatial as scp
import math

POSITIVE_INFINITY = float("inf")

def nd_sorting(F,nmax):
    (M, l) = F.shape
    nd_ranks = np.ones((l,), dtype=np.int)*POSITIVE_INFINITY
    P = np.ones((l,), dtype=np.int)
    for i in range(0, l):
        P[i] = i
    i=1
    count = 0
    while count < nmax and len(P) > 0:
        (ndf_index, df_index) = ND_Front(F[:,P])
        for j in range(0,len(ndf_index)):
            nd_ranks[P[ndf_index[j]]] = i
        count = count + len(ndf_index)
        P_new = np.ones((len(df_index),), dtype=np.int)
        for j in range(0,len(df_index)):
            P_new[j] = P[df_index[j]]
        P = P_new
        i = i+1
    return nd_ranks

def nd_sorting_best_percentage(F,nmax,best_percent):
    """
    rank only for the best half in ordered of the value
    :param F:
    :param nmax:
    :return:
    """
    indx_good = index_best_percentage(F[0], best_percent)
    F_new = np.zeros((F.shape[0], len(indx_good)))
    for i, indx in enumerate(indx_good):
        F_new[:, i] = F[:, indx]

    nd_ranks_good = nd_sorting(F_new, nmax)

    (M, l) = F.shape
    nd_ranks = np.ones((l,), dtype=np.int)*POSITIVE_INFINITY
    for i, indx in enumerate(indx_good):
        nd_ranks[indx] = nd_ranks_good[i]
    return nd_ranks


def ND_Front(F):
    (M, l) = F.shape
    df_index = []
    ndf_index = [int(0)]
    for i in range(1, l):
        (ndf_index, df_index) = ND_Add(F[:,0:i+1], df_index, ndf_index)
    return (ndf_index, df_index)

def ND_Add(F, df_index, ndf_index):
    (M, l) = F.shape
    l = int(l - 1)
    ndf_count = len(ndf_index)
    ndf_index.append(l)
    ndf_count += 1
    j = 1
    while j < ndf_count:
        if domination(F[:,l],F[:,ndf_index[j-1]],M):
            df_index.append(ndf_index[j-1])
            ndf_index.remove(ndf_index[j-1])
            ndf_count -= 1
        elif domination(F[:,ndf_index[j-1]],F[:,l],M):
            df_index.append(l)
            ndf_index.remove(l)
            ndf_count -= 1
            break
        else:
            j += 1
    return (ndf_index, df_index)

def epsilon_ND_front(F, e, lB):

    M, l = F.shape

    if lB is None:
        lB = F.min(1)

    F_box = np.transpose(compute_epsilon_precision(np.transpose(F), e, lB))

    df_index = []
    box_index = []
    ndf_index = [int(0)]
    ndf_count = 1
    for i in range(1, l):
        ndf_index.append(i)
        ndf_count = ndf_count + 1
        j = 1
        while(j < ndf_count):
            if domination(F_box[:, i], F_box[:, ndf_index[j - 1]], M):
                df_index.append(ndf_index[j - 1])
                ndf_index.remove(ndf_index[j-1])
                ndf_count = ndf_count - 1
            elif domination(F_box[:,ndf_index[j - 1]], F_box[:, i], M):
                df_index.append(i)
                ndf_index.remove(i)
                ndf_count = ndf_count - 1
                break
            elif np.array_equal(F_box[:, i], F_box[:, ndf_index[j - 1]]):
                d1 = np.linalg.norm((F[:, i] - F_box[:, i]) / e)
                d2 = np.linalg.norm((F[:, ndf_index[j - 1]] - F_box[:, i]) / e)
                if(d1 < d2):
                    box_index.append(ndf_index[j - 1])
                    ndf_index.remove(ndf_index[j - 1])
                    ndf_count = ndf_count - 1
                else:
                    box_index.append(i)
                    ndf_index.remove(i)
                    ndf_count = ndf_count - 1
                    break
            else:
                j = j + 1

    return ndf_index, df_index, box_index

def compute_epsilon_precision(F, e, lB):
# This function comnputes epsilon precise values of all elements in F
    M, l = F.shape
    F_box = lB * np.ones(l) + np.multiply(np.floor((F - lB * np.ones(l)) / (e * np.ones(l))), (e * np.ones(l)))
    return F_box

def domination(fA, fB, M):
    d = False
    for i in range(0,M):
        if fA[i] > fB[i]:
            d = False
            break
        elif fA[i] < fB[i]:
            d = True
    return d

def weakly_dominates(fA, fB, M):
    d = False
    for i in range(0,M):
        if fA[i] > fB[i]:
            d = False
            break
        elif fA[i] <= fB[i]:
            d = True
    return d

def taboo_region(X, X_c, sigma, dim, nc):
    flag = 1
    for i in range(nc):
        if X_c[i, dim+4] == 0:
            sigma_new = sigma
        else:
            sigma_new = np.power(1/2, X_c[i, dim+4])*sigma
        d = scp.distance.euclidean(X, X_c[i, 0:dim])
        if d < sigma_new:
            flag = 0
            break
    return flag

def taboo_region_new(X, X_c, sigma, dim, nc):
    flag = 1
    for i in range(nc):
        if X_c[i, dim+4] == 0:
            sigma_new = sigma
        else:
            sigma_new = np.power(1/2, X_c[i, dim+4])*sigma
        d = scp.distance.euclidean(X, X_c[i, 0:dim])
        if d < sigma_new*np.sqrt(dim):
            flag = 0
            break
    return flag


def dynamic_taboo_region(X, X_c, sigma, dim, nc, d_thresh):
    flag = 1
    for i in range(nc):
        if X_c[i, dim+4] == 0:
            sigma_new = sigma
        else:
            sigma_new = np.power(1/2, X_c[i, dim+4])*sigma
        d = scp.distance.euclidean(X, X_c[i, 0:dim])
        if d < sigma_new*d_thresh/np.sqrt(len(X)):
            flag = 0
            break
    return flag

def index_mean_percentage(data, best_percent):
    """
    select the index based on the the mean percentage related to best solution found
    :param data:
    :return:
    """
    index = np.argsort(data)
    select_data = []
    select_index = []
    for i in index:
        select_data.append(data[i])
        select_index.append(i)
        mean_ = np.mean(select_data)
        if data[index[0]] > 0:
            cond = mean_ > (1+best_percent)*data[index[0]]
        else:
            cond = mean_ > (1-best_percent)*data[index[0]]
        if cond:
            select_data.pop(-1)
            select_index.pop(-1)
    select_index = np.asarray(select_index)
    return select_index

def index_best_half(data):
    """
    select the index the best half ordered in value
    :param data:
    :return:
    """
    indx = np.argsort(data)
    half_length = int(math.ceil(len(indx) / 2))
    select_index = indx[:half_length]
    return select_index

def index_best_percentage(data, best_percent):
    """
    select the index the best half ordered in value
    :param data:
    :return:
    """
    indx = np.argsort(data)
    p_length = int(math.ceil(len(indx)*best_percent))
    select_index = indx[:p_length]
    return select_index

