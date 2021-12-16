import numpy as np
from numpy.linalg import inv, pinv, eig, lstsq


def DKF_filtering(means, _vars, A, G, VZ):
    '''
    DKF_Filtering filters under the DKF model:
        hidden(t)|observed(t) ~ N(means, _vars)
        hidden(t)|hidden(t-1) ~ N(A*hidden(t-1), G)
        hidden(t) ~ N(0, VZ)

    This function returns the estimates:
        hidden(t)|observed(1:t-1) ~ N(means_filtered, vars_filtered)
        means are [n,T] dimensional
        _vars are [n,n,T] dimensional
            _vars has the underscore to prevent confusion with the Python "vars" keyword.
    '''
     
    n, T = means.shape
    means_filtered = np.zeros((n,T))
    
    S = _vars[:, :, 0]
    nu = means[:, 0]
    means_filtered[:, 0] = nu
    
    
    for t in np.arange(1, T):
        
        aa = means[:, t]
        bb = _vars[:, :, t]
        
        if not np.all(eig( inv(bb) - inv(VZ) )[0] > 0):
            bb = inv(inv(bb) + inv(VZ))
            
        Mi = pinv(G + A.dot(S).dot(A.T))
        S = pinv(pinv(bb) + Mi - pinv(VZ))
        
        # "A \ y" in MATLAB is "linalg.lstsq(A, y, rcond=None)[0] in Python."
        bb_aa = lstsq(bb, aa, rcond=None)[0] # bb \ aa
        nu = S.dot(bb_aa + Mi.dot(A.dot(nu)))
        
        means_filtered[:,t] = nu
        
    
    return means_filtered