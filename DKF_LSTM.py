import numpy as np
import torch
from torch import nn

# Helper function
from DKF_NN import predict_nn



class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True) #lstm
        
        self.fc = nn.Linear(hidden_size * num_layers, num_classes) #fully connected last layer
        
        
    
    def forward(self,x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        #hn = hn.view(-1, self.hidden_size * self.num_layers) #reshaping the data for Dense layer next
        
        hn = hn.transpose(0,1)
        hn = hn.reshape(-1, self.hidden_size * self.num_layers)
        
        out = self.fc(hn) #Final Output
        
        return out
    
    
    

#def add_dim1(tensor):
#    return tensor.reshape((tensor.shape[0], 1, tensor.shape[1]))




def run_lstm(lstm, n_steps, dx, num_epochs, learning_rate, weight_decay, x0, z0, boundary, x1, x1_tensor):
    '''
    As in the normal code, x0, z0, and x1 are all 2x? or 10x? ndarrays;
    boundary is the number of data points in x0_train;
    x1_tensor is a ?x10 tensor.
    
    Returns: means and variances of prediction.
    '''
 
    # Convert to tensor, reshape to add a timestamp at dim = 1.
    # dimensions = [rows, timestamps, features]

    # Access x0 directly to get different training data.
    # This data is chronologically continuous, unlike x0_train and x0_test, which are chronologically dscontinuous and random.
    X_train = x0[:, :boundary]
    X_test  = x0[:, boundary:]

    y_train = z0[:, :boundary]
    y_test  = z0[:, boundary:]

    # OLD CODE (n_steps = 1):
    #X_train_tensor = add_dim1(torch.from_numpy(X_train.T).float())
    #X_test_tensor  = add_dim1(torch.from_numpy(X_test.T).float())

    # Initialize the LSTM input tensors.
    X_train_tensor = torch.zeros((X_train.shape[1] - n_steps + 1, n_steps, dx))
    X_test_tensor = torch.zeros(( X_test.shape[1]  - n_steps + 1, n_steps, dx))

    # For each index i of an LSTM input tensor,
    # put in n_steps elements of X_train / X_test.
    for i in range(len(X_train_tensor)):
        X_train_tensor[i] = torch.from_numpy(X_train.T).float()[i:i+n_steps]

    for i in range(len(X_test_tensor)):
        X_test_tensor[i] = torch.from_numpy(X_test.T).float()[i:i+n_steps]
    
    # Form the LSTM output target tensors.
    y_train_tensor = torch.from_numpy(y_train.T).float()[n_steps-1:]
    y_test_tensor  = torch.from_numpy(y_test.T).float()[n_steps-1:]




    # Train model.

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, weight_decay=weight_decay) 

    losses = []
    losses_val = []

    for epoch in range(num_epochs):
        outputs = lstm.forward(X_train_tensor) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0

        # obtain the loss function
        loss = criterion(outputs, y_train_tensor)
        losses.append(float(loss))

        loss.backward() #calculates the loss of the loss function

        optimizer.step() #improve from loss, i.e backprop
        #if epoch % 500 == 499:
        #    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 
                                
        with torch.no_grad():
            losses_val.append(float(criterion(lstm.forward(X_test_tensor), y_test_tensor)))
            
            
            
    # Calibrate prediction method using held-out data, then get predicted data from LSTM.
    # Make x1_tensor a 4D array so that calling x1_tensor_lstm[i] will return a 3D array for any i.
    x1_tensor_lstm = torch.zeros((len(x1_tensor) - n_steps + 1, 1, n_steps, dx))
    for i in range(len(x1_tensor_lstm)):
        x1_tensor_lstm[i] = x1_tensor[i:i+n_steps]
        
    u_means_lstm, u_vars_lstm, tic = predict_nn(model = lstm,
                                                
                                                x0_test = X_test[:,n_steps-1:],
                                                
                                                x0_test_tensor = X_test_tensor,
                                                
                                                z0_test = y_test[:,n_steps-1:],
                                                
                                                x1 = x1[:,n_steps-1:],
                                                
                                                x1_tensor = x1_tensor_lstm)
    
    
    return u_means_lstm, u_vars_lstm, tic, (losses, losses_val)