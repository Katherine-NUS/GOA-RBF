import numpy as np


def push_to_boundary(SM,x,size):
    bp = np.matlib.repmat(x,size, 1)
    for i in range(size):
        num = np.random.random_integers(1,SM.data.dim)
        ids=np.random.permutation(SM.data.dim)

        for s in range(num):
            tmp=np.random.random()
            if tmp<0.33:
              bp[i,ids[s]] = SM.data.xlow[ids[s]]
            elif tmp>=0.33 and tmp<0.66:
              bp[i, ids[s]] = 0.00
            else:
              bp[i,ids[s]] = SM.data.xup[ids[s]]
    return bp