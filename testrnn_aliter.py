# This is a simple implementation to do a sanity check on the RNN used
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import sys
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from timeit import default_timer as timer

# Create the necessary time series data
def generate_sine(k, n, add_noise=False):
    
    series_x = np.linspace(0, np.pi * 2 * k, n, dtype=np.float32)
    series_x2 = np.linspace(0, np.pi * 0.85 * 2 * k, n, dtype=np.float32)
    
    if not add_noise:
        series_y = np.sin(series_x) #+ np.sin(series_x2)
    else:
        series_y = np.sin(series_x) + np.random.normal(loc=0.0, scale=1e-1, size=(len(series_x),)) #+ np.sin(series_x2)
    ts_data_xy = np.concatenate((series_x.reshape((-1, 1)), series_y.reshape((-1, 1))), axis=1)
    return ts_data_xy

# Plot the data
def plot_timeseries(ts_data, tr_data=None, predicted_data=None, eval=False):

    if not eval:
        plt.figure()
        plt.plot(ts_data, linewidth=2)
        plt.legend(['Original timeseries'])
    else:
        plt.figure()
        plt.plot(np.arange(0, len(tr_data)), tr_data, 'g+-', linewidth=2)
        plt.plot(np.arange(len(tr_data), len(tr_data) + len(predicted_data)), predicted_data, 'r+-', linewidth=3)
        plt.plot(ts_data, 'k--', linewidth=2)
        plt.legend(['Training timeseries', 'Predicted timeseries', 'Original timeseries'])
    
    plt.xlabel("N / time")
    plt.ylabel("Signal Amplitude")
    plt.title("Plot of timeseries")
    plt.show()
    return None

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

    # The data shape should be N x T x 1 (for 1-dimensional feature data)
    inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], ts_data.shape[1]))

    return inputs, targets

def train_validation_split(inputs, targets, tr_split=0.5):

    num_train_samples = int(tr_split * len(inputs))
    num_val_samples = len(inputs) - num_train_samples
    tr_inputs = torch.Tensor(inputs[:num_train_samples, :, :])
    tr_targets = torch.Tensor(targets[:num_train_samples])
    val_inputs = torch.Tensor(inputs[num_train_samples:, :, :])
    val_targets = torch.Tensor(targets[num_train_samples:])

    return tr_inputs, tr_targets, val_inputs, val_targets

# Create an AR model for prediction

class RNN_model(nn.Module):
    
    def __init__(self, input_size, output_size, n_hidden, n_layers, 
        num_directions, model_type, batch_first, lr, device,
        num_epochs):
        super(RNN_model, self).__init__()

        # Defining some parameters
        self.hidden_dim = n_hidden
        self.num_layers = n_layers
        self.input_size = input_size
        self.output_size = output_size
        self.num_directions = num_directions
        self.model_type = model_type
        self.batch_first = batch_first
        self.lr = lr
        if device is None:
            self.device = self.get_device()
        else:
            self.device = device

        self.num_epochs = num_epochs

        # Defining the layers

        # RNN Layer
        if model_type.lower() == "rnn":
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, batch_first=self.batch_first)   
        elif model_type.lower() == "lstm":
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, batch_first=self.batch_first, bidirectional=False)   
        elif model_type.lower() == "gru":
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_dim, 
                num_layers=self.num_layers, batch_first=self.batch_first, bidirectional=False)   
        else:
            print("Model type cannot be recognized!!") 
            sys.exit() 
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim * self.num_directions, self.output_size)
    
    def init_h0(self, batch_size):
        
        # This method generates the first hidden state of zeros (h0) which is used in the forward pass
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return h0

    def get_device(self):
    
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
        return device

    def forward(self, x):

        batch_size = x.shape[0]
        r_out, hn_all = self.rnn(x)
        r_out = r_out.contiguous().view(batch_size, -1, self.num_directions, self.hidden_dim)[:,-1,:,:]
        r_out_last_step = r_out.reshape((-1, self.hidden_dim))
        y = self.fc(r_out_last_step)
        return y

#NOTE: Introduced the function here, but not incorporated into the code so far
def push_model(nets, device='cpu'):
    nets = nets.to(device=device)
    return nets

def train_rnn(model, nepochs, tr_inputs, tr_targets, val_inputs, val_targets, tr_verbose=True):
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    
    #scheduler = ExponentialLR(optimizer, gamma=0.99)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.9)
    criterion = nn.MSELoss()
    losses = []
    val_losses = []

    total_time = 0.0
    
    # Start time
    starttime = timer()

    for epoch in range(nepochs):
        
        optimizer.zero_grad()
        X = Variable(tr_inputs, requires_grad=False).type(torch.FloatTensor)
        tr_predictions = model.forward(X)
        tr_loss = criterion(tr_predictions, tr_targets)
        tr_loss.backward(retain_graph=True)
        optimizer.step()

        losses.append(tr_loss.item())

        with torch.no_grad():
            
            _, P, _ = val_inputs.shape
            val_predictions = predict_rnn(model=model, eval_input=tr_inputs[-1, :, :].reshape((1, P, -1)), n_predict=len(val_inputs))
            val_loss = criterion(torch.FloatTensor(val_predictions).reshape((-1, 1)), val_targets)
            val_losses.append(val_loss.item())
            '''
            X_val = Variable(val_inputs, requires_grad=False).type(torch.FloatTensor)
            val_predictions = model.forward(X_val)
            val_loss = criterion(val_predictions, val_targets)
            val_losses.append(val_loss.item())
            '''

        endtime = timer()
        # Measure wallclock time
        time_elapsed = endtime - starttime
        
        #if tr_verbose == True and (((epoch + 1) % 50) == 0 or epoch == 0):
        if (((epoch + 1) % 100) == 0 or epoch == 0):
            print("Epoch: {}/{}, Training MSE Loss:{:.9f}, Val. MSE Loss:{:.9f}, Time elapsed:{} secs ".format(epoch+1, 
            model.num_epochs, tr_loss, val_loss, time_elapsed))

    # Measure wallclock time for total training
    print("Time elapsed measured in seconds:{}".format(timer() - starttime))

    return losses, val_losses, model

def predict_rnn(model, eval_input, n_predict):

    eval_predictions = []
    eval_input = torch.Tensor(eval_input)
    model.eval()
    with torch.no_grad():

        for _ in range(n_predict):
            
            X_eval = Variable(eval_input, requires_grad=False).type(torch.FloatTensor)
            val_prediction = model.forward(X_eval)
            eval_predictions.append(val_prediction)
            eval_input = torch.roll(eval_input, -1)
            if eval_input.shape[1] is not None:
                eval_input[:, -1] = val_prediction
            else:
                eval_input[-1] = val_prediction
            '''
            X_eval = Variable(torch.Tensor(eval_input), requires_grad=False).type(torch.FloatTensor)
            val_prediction = model.forward(X_eval)
            eval_predictions.append(val_prediction.numpy())
            eval_input = np.roll(eval_input, shift=-1)
            if eval_input.shape[1] is not None:
                eval_input[:, -1] = val_prediction.numpy()
            else:
                eval_input[-1] = val_prediction.numpy()
            '''
    #eval_predictions = np.row_stack(eval_predictions)
    eval_predictions = torch.stack(eval_predictions).numpy().reshape((-1, 1))
    return eval_predictions


def plot_losses(tr_losses, val_losses):
    plt.figure()
    plt.plot(tr_losses, 'r+-', linewidth=2)
    plt.plot(val_losses, 'b*-', linewidth=1.5)
    plt.xlabel("No. of training iterations")
    plt.ylabel("MSE Loss")
    plt.legend(['Training Set', 'Validation Set'])
    plt.title("MSE loss vs. no. of training iterations")
    #plt.savefig('./models/loss_vs_iterations.pdf')
    plt.show()


def main():

    # Set input parameters
    k = 20 # frequency
    N = 600 # Number of points
    p = 10 # No. of taps (lags)
    ulen = 1
    tr_to_val_split = 0.8

    # Generate and plot data
    ts_data_xy = generate_sine(k=k, n=N, add_noise=True) 
    ts_data = ts_data_xy[:, 1].reshape((-1, 1))
    plot_timeseries(ts_data, tr_data=None, predicted_data=None, eval=False)

    inputs, targets = create_dataset(ts_data, alen=p, ulen=ulen)
    tr_inputs, tr_targets, val_inputs, val_targets = train_validation_split(
        inputs, targets, tr_split=tr_to_val_split
    )

    model = RNN_model(input_size=tr_inputs.shape[-1], output_size=tr_targets.shape[-1], 
        n_hidden=40, n_layers=2, num_directions=1, model_type="LSTM", batch_first=True,
        lr=1e-2, device='cpu', num_epochs=600)

    tr_losses, val_losses, model = train_rnn(model=model, nepochs=model.num_epochs, tr_inputs=tr_inputs,
        tr_targets=tr_targets, val_inputs=val_inputs, val_targets=val_targets
    )

    plot_losses(tr_losses, val_losses)
    test_input = val_inputs[0].reshape((1, -1, 1))
    predicted_tsdata = predict_rnn(model=model, eval_input=test_input, n_predict=len(val_inputs))
    plot_timeseries(ts_data, tr_data=ts_data[:len(tr_inputs) + p], 
        predicted_data=predicted_tsdata, eval=True)

    return None

if __name__ == "__main__":
    main()
