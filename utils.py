import os
import numpy as np
from sklearn.preprocessing import normalize
   
def log_domain(log_matirx):
    #np.seterr(divide='ignore')
    #return np.where(log_matirx==0, -1E10, np.log(log_matirx) )
    return np.log(log_matirx)
def exp_domain(log_matirx):
    #np.seterr(divide='ignore')
    #return np.where(log_matirx==-1E10, 0, np.exp(log_matirx) )
    return np.exp(log_matirx)
def init_norm(matrix):
    return normalize(matrix, axis=1, norm='l1')
def init_norm_1D(matrix):
    return matrix/np.sum(matrix)