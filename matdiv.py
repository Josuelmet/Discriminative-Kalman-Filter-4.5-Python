import numpy as np

def matdiv(a, b):
    '''
    Recreates MATLAB's matrix operation a / b.
    '''
    return a.dot(np.linalg.pinv(b))