import numpy as np

# Helper function:
from matdiv import matdiv


def KalmanOneStep(x, mu, V, A, C, S, G):
    
    P = A.dot(V).dot(A.T) + G
    
    PC = P.dot(C.T)
    K0 = matdiv(PC, C.dot(PC) + S)
    Amu = A.dot(mu)
    mu = Amu + K0.dot(x - C.dot(Amu))
    
    V = P - K0.dot(C).dot(P)
    
    return mu, V


def KalmanPredict(x, A, C, S, G, mu0, V0):
    
    dz = mu0.size
    n = x.shape[1]
    z = np.zeros((dz, n))
    
    # step 1
    K = matdiv(V0.dot(C.T), C.dot(V0).dot(C.T) + S)
    z[:,0] = mu0 + K.dot(x[:,0] - C.dot(mu0))
    V = (np.eye(dz) - K.dot(C)).dot(V0)
    vars = np.zeros((dz, dz, n))
    vars[:, :, 0] = V
    
    # remaining steps
    for k in np.arange(1, n):
        mu, V = KalmanOneStep(x[:,k], z[:,k-1], V, A, C, S, G)
        z[:,k] = mu
        vars[:,:,k] = V
        
    return z, vars