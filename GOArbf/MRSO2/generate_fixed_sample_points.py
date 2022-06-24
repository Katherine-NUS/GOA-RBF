import numpy as np

def Sample_points(sample_size,dim):
    P=np.random.RandomState(0).uniform(0,1,[sample_size,dim])
    return P