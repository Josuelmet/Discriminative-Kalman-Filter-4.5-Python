import numpy as np
from scipy.spatial.distance import cdist

def nw_est(ztest, xtrain, ztrain, sz2, return_S_est=False):
    '''
    NW_EST = Nadaraya-Watson kernel regression.
    '''

    dz = ztest.shape[1]
    P_i = (2*np.pi*sz2)**(dz/2)


    # cdist('sqeuclidean') calculates sum((ztrain[i] - ztest) ** 2) for all rows of ztrain (i = 0, 1, ... len(ztrain)).
    exp = np.exp(-cdist(ztrain, ztest, 'sqeuclidean')/(2*sz2))
    
    # exp is a 2D column vector. Just in case, take the mean of each (1-element) row of exp, while keeping it a 2D column vector.
    P_i = (2*np.pi*sz2)**(dz/2) * np.mean(exp, axis=1, keepdims=True) 
    
    # Take the sum of each column of P_i. P_i should just be one column, so sP should just be one integer.
    sP = P_i.sum(axis=0)
    
    # Multiply each column of xtrain by P_i, entrywise. Then, sum each column, yielding a 1D row.
    xP = np.sum(xtrain * P_i, axis=0)

    # sP oftentimes equals 0. Luckily, numpy handles this by returning NaN.
    f_est = xP / sP
    
    #if sP == 0:
    #    return np.inf


    # If S_est is not needed, do not calculate it.
    if not return_S_est:
        return f_est
    

    nt, dx = xtrain.shape
    S_est = np.zeros((dx, dx))

    for i in range(nt):
        erri = xtrain[i, :] - f_est
        S_est = S_est + P_i[i] / sP * np.outer(erri, erri)
    
    

    return f_est, S_est