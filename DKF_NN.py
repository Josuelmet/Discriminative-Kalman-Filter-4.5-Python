import numpy as np
import time

import torch
from torch import nn

from scipy.optimize import minimize_scalar


# Helper functions:
from nw_est import nw_est
from optimize_sx import optimize_sx




def train_nn(model, learning_rate, weight_decay, iterations, x_train, y_train):
    '''
    Perform gradient descent on the model.
    
    Returns: nothing.
    '''
    
    loss_fn = torch.nn.MSELoss(reduction='sum')

    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for t in range(iterations):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x_train)

        # Compute and print loss.
        loss = loss_fn(y_pred, y_train)
        
        #if t % 500 == 499:
        #    print(t, loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        
        
        
        
        
        
def predict_nn(model, x0_test, x0_test_tensor, z0_test, x1, x1_tensor):
    '''
    model = neural network.
    
    x0_test and z0_test are held-out training data to run NW regression with before running predictions on x1.
    x0_test = test data inputs.
    x0_test_tensor = test data inputs in Tensor form.
    z0_test = test data outputs.
    
    x1 = data to predict.
    x1_tensor = data to predict, in Tensor form.
    
    Returns:
        - predicted values and covariances of x1, after calibrating using data from x0_test.
        - the starting time of NN prediction.
    '''
    
    
    # Estimate error covariance on heldout data (x0_test and z0_test):
    
    # dz = dimensionality of test data outputs
    # n0_test = number of test data points.
    dz, n0_test = z0_test.shape

    # Get prediction for x0_test.
    znn0_test = model(x0_test_tensor).detach().numpy().T # znn0_test = NN prediction of x0_test.
    z0resnn = z0_test - znn0_test


    z0resOuternn = np.zeros((dz**2, n0_test))

    for t in range(n0_test):
        z0resOuternn[:,t] = np.outer(z0resnn[:,t], z0resnn[:,t]).flatten()
    
    
    
    
    '''
    # Run NW kernel regression, minimizing leave-one-out MSE on the training set, so as to find the optimal bandwidth.
    
    # Find argmax of x on function mseres.
    # Sometimes this will throw RuntimeWarnings because of a division by zero in nw_est.py.
    # In that case, nw_est returns an array of NaNs, which the optimizer correctly handles as a very high number.
    '''
    
    
    sxresoptnn = optimize_sx(ztest = x0_test,
                             xtrain = z0resOuternn,
                             ztrain = x0_test)
    
    
    '''
    Predict NN on x1 ---------------------------------------------------------------------------------------------
    '''
    n1 = x1.shape[1] # Get the number of data points to predict.

    starting_time = time.time()

    u_means_nn = np.zeros((dz, n1))
    u_vars_nn = np.zeros((dz, dz, n1))

    for t in range(n1):    
        # Get the output from neural network for t-th observation (row) of x1_tensor, since x1_tensor is tall (?x10 Tensor).
        u_means_nn[:, t] = model(x1_tensor[t]).detach().numpy()

        u_vars_nn[:,:,t] = nw_est(x1[:,t:t+1].T, z0resOuternn.T, x0_test.T, sxresoptnn, return_S_est=False).reshape((dz, dz))
        
        
        
    return u_means_nn, u_vars_nn, starting_time