# Autoregressive model tutorial

# Import the necessary libraries
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
from src.utils import plot_losses
from timeit import default_timer as timer

# Create an AR model for prediction
class Linear_AR(nn.Module):

    def __init__(self, num_taps, lossfn_type, lr, num_epochs, init_net=True, device='cpu'):
        super(Linear_AR, self).__init__()
        
        self.num_taps = num_taps
        if lossfn_type.lower() == "mse":
            self.lossfn = nn.MSELoss()
        self.lr = lr
        self.init_net = init_net
        self.num_epochs = num_epochs
        self.device = device
        if self.init_net == True:
            self.init_linearAR()
        else:
            pass
            
    def init_linearAR(self):
        self.net = nn.Sequential(
            nn.Linear(self.num_taps, 1))
        #print(self.net)
    
    def forward(self, inputs):
        return self.net(inputs)

def train_armodel(model, nepochs, inputs, targets, tr_split=0.8, tr_verbose=False):
    
    # Train -  Validation split
    tr_inputs, tr_targets, val_inputs, val_targets = train_validation_split(
        inputs, targets, tr_split=tr_split
    )

    # Initialise optimization parameters
    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    criterion = model.lossfn
    model.net.train()
    model.net = push_model(model.net, device=model.device)

    losses = []
    val_losses = []

    total_time = 0.0

    for epoch in range(nepochs):
        
        starttime = timer()
        optimizer.zero_grad()
        X = Variable(tr_inputs, requires_grad=False).type(torch.FloatTensor)
        tr_predictions = model(X)
        tr_loss = criterion(tr_predictions, tr_targets)
        tr_loss.backward(retain_graph=True)
        optimizer.step()
        losses.append(tr_loss.item())

        with torch.no_grad():
            '''
            _, P = val_inputs.shape
            val_predictions = predict_armodel(model=model, eval_input=tr_inputs[-1, :], n_predict=len(val_inputs))
            val_loss = criterion(torch.FloatTensor(val_predictions).reshape((-1, 1)), val_targets)
            val_losses.append(val_loss.item())
            '''
            X_val = Variable(val_inputs, requires_grad=False).type(torch.FloatTensor)
            val_predictions = model(X_val)
            val_loss = criterion(val_predictions, val_targets)
            val_losses.append(val_loss.item())
            
        
        endtime = timer()
        # Measure wallclock time
        time_per_epoch = endtime - starttime
        total_time += time_per_epoch

        if tr_verbose == True and (((epoch + 1) % 50) == 0):
        #if (((epoch + 1) % 100) == 0 or epoch == 0):
            print("Epoch: {}/{}, Training MSE Loss:{:.8f}, Val. MSE Loss:{:.8f}, Time elapsed:{} secs ".format(
                epoch+1, nepochs, tr_loss, val_loss, time_per_epoch))

    # Measure wallclock time for total training
    print("Time elapsed measured in seconds:{}".format(total_time))

    #if tr_verbose == True:
    #    plot_losses(losses, val_losses)

    return losses, val_losses, model

'''
def predict_armodel(model, eval_input, n_predict):
    p = eval_input.shape[0]
    out=np.zeros(p+n_predict)
    out[:p] = eval_input
    model.net.eval()
    with torch.no_grad():
        for i in range(n_predict):
            X_eval = Variable(torch.Tensor(out[i:p+i]), requires_grad=False).type(torch.FloatTensor)
            val_prediction = model(X_eval)
            out[p+i] = val_prediction.numpy()
    return out[p:]
'''

def predict_armodel(model, eval_input, n_predict):

    eval_predictions = []
    eval_input = torch.Tensor(eval_input)
    model.eval()
    with torch.no_grad():

        for _ in range(n_predict // model.output_size + 1):
            
            X_eval = Variable(eval_input, requires_grad=False).type(torch.FloatTensor)
            val_prediction = model.forward(X_eval)
            eval_predictions.append(val_prediction)
            #eval_input = torch.roll(eval_input, -1)
            eval_input = torch.roll(eval_input, -model.output_size)
            if eval_input.shape[1] is not None:
                #eval_input[:, -1] = val_prediction
                eval_input[:, -model.output_size:] = val_prediction.reshape((1, -1, 1))
            else:
                eval_input[-model.output_size:] = val_prediction.reshape((1, -1, 1))
                #eval_input[-1] = val_prediction
        
    #eval_predictions = np.row_stack(eval_predictions)
    eval_predictions = torch.stack(eval_predictions).numpy().reshape((-1, 1))
    eval_predictions = eval_predictions.flatten()[:n_predict]
    return eval_predictions

def plot_timeseries(ts_data, tr_data=None, predicted_data=None, eval=False):

    if not eval:
        plt.figure()
        plt.plot(ts_data, 'k--', linewidth=3)
        plt.legend(['Original timeseries'])
    else:
        plt.figure()
        plt.plot(np.arange(0, len(tr_data)), tr_data, 'g+-', linewidth=3)
        plt.plot(np.arange(len(tr_data), len(tr_data) + len(predicted_data)), predicted_data, 'r+-', linewidth=2)
        plt.plot(ts_data, 'k--', linewidth=3)
        plt.legend(['Training timeseries', 'Predicted timeseries', 'Original timeseries'])
    
    plt.xlabel("N / time")
    plt.ylabel("Signal Amplitude")
    plt.title("Plot of timeseries")
    plt.show()

    return None

# Create the necessary time series data
def generate_sine(N, w):
    """Generates a sine wave to be used as time-series

    Args:
        N ([int]): No. of data points
        w ([int]): Angular frequency of the sine wave
    
    Returns:
        sine wave of desired number of points and 
        frequency
    """
    ts_data = np.sin(w * np.arange(N))
    ts_data = ts_data.reshape((-1, 1))
    return ts_data

def create_dataset(ts_data, alen, ulen=1):
    """ Creates the suitable dataset for time-series
    forecasting
    Args:
        ts_data ([np.array]): Time-series data (t x 1)
        alen ([type]): Analysis window (in samples)
        ulen (int, optional): Update window (in samples). 
        Defaults to 1.

    Returns:
        inputs [np.array]: Inputs in particular dataset
        targets [np.array]: Targets in particular dataset
    """
    num_windows = (len(ts_data) - alen) // ulen
    assert num_windows > 0
    print("Number of windows for modeling the data:{}".format(num_windows))
    inputs = np.zeros((num_windows, alen))
    targets = np.zeros((num_windows, 1))
    
    for num in range(num_windows):
        
        l = num * ulen
        u = alen + (num * ulen)
        inputs[num, :] = ts_data[l:u].reshape(-1,)
        targets[num] = ts_data[u].reshape(-1,)

    #inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], ts_data.shape[1]))
    return inputs, targets

def train_validation_split(inputs, targets, tr_split=0.5):

    num_train_samples = int(tr_split * len(inputs))
    num_val_samples = len(inputs) - num_train_samples
    tr_inputs = torch.Tensor(inputs[:num_train_samples])
    tr_targets = torch.Tensor(targets[:num_train_samples])
    val_inputs = torch.Tensor(inputs[num_train_samples:])
    val_targets = torch.Tensor(targets[num_train_samples:])

    return tr_inputs, tr_targets, val_inputs, val_targets

def push_model(nets, device='cpu'):
    nets = nets.to(device=device)
    return nets

def main():

    # Set input parameters
    w = 0.1 # Angular frequency of the sine wave
    N = 200 # No. of points
    num_taps = 20
    ulen = 1
    tr_to_val_split = 0.75

    # Generate and plot data
    ts_data = generate_sine(N, w) 
    plot_timeseries(ts_data, tr_data=None, predicted_data=None, eval=False)

    inputs, targets = create_dataset(ts_data, alen=num_taps, ulen=ulen)
    tr_inputs, tr_targets, val_inputs, val_targets = train_validation_split(
        inputs, targets, tr_split=tr_to_val_split
    )

    model = Linear_AR(num_taps=num_taps, lossfn_type="mse", lr=0.1, num_epochs=80, 
                      init_net=True, device='cpu')

    #tr_losses, val_losses, model = train_armodel(model, nepochs=model.num_epochs, tr_inputs=tr_inputs,
    #                                    tr_targets=tr_targets, val_inputs=val_inputs, 
    #                                    val_targets=val_targets)
    tr_losses, val_losses, model = train_armodel(model=model,
                                            nepochs=model.num_epochs,
                                            inputs=inputs,
                                            targets=targets,
                                            tr_split=tr_to_val_split
                                            )

    plot_losses(tr_losses, val_losses)

    test_input = val_inputs[0].reshape((1, -1))
    predicted_tsdata = predict_armodel(model=model, eval_input=test_input, n_predict=len(val_inputs))
    plot_timeseries(ts_data, tr_data=ts_data[:len(tr_inputs) + num_taps], predicted_data=predicted_tsdata, eval=True)

    return None

if __name__ == "__main__":
    main()
