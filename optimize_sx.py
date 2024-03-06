import numpy as np
from scipy.optimize import minimize_scalar
from nw_est import nw_est

import warnings

def optimize_sx(ztest, xtrain, ztrain):
    '''
    Run NW kernel regression, minimizing leave-one-out MSE on the training set, so as to find the optimal bandwidth.
    
    Returns: The optimal bandwidth to use.
    '''
    
    
    def mse(sx):
        '''
        Helper function to calculate the error between NW regression's f_hat and xtrain, when using kernel bandwidth sx.
        '''

        # Ensure that sx is neither negative nor 0.
        sx = np.abs(sx)
        if sx < 1e-5:
            sx = 1e-5

        preds = np.zeros_like(xtrain) # Predictions

        # For each column of preds:
        for i in range(preds.shape[1]):
            '''
            Store the i-th result of nw_est as the i-th column in preds.
            Run NW kernel regression with bandwidth chosen by minimizing leave-one-out MSE on the training set.
            
            ztest  = the i-th column of zTest, transposed into one 2D row.
            xtrain = xtrain without its i-th column, then transposed.
            ztrain = ztrain without its i-th column, then transposed.
            '''
            preds[:, i] = nw_est(ztest  = ztest[:, i:i+1].T,
                                 xtrain = np.delete(xtrain, i, axis=1).T,
                                 ztrain = np.delete(ztrain, i, axis=1).T,
                                 sz2 = sx)
    
        # Return MSE of prediction made using bandwidth sx.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean((xtrain - preds) ** 2)


    '''
    Find argmax of x on function mse.
    Sometimes this will throw RuntimeWarnings because of a division by zero in nw_est.py.
    In that case, nw_est returns an array of NaNs, which the optimizer correctly handles as a very high number.
    '''
    sx_opt = minimize_scalar(mse)
    
    return sx_opt.x # Just return the argmax, don't keep any of the metadata about how optimization went.
