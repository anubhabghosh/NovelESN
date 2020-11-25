"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
numpy
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from src.utils import get_cycle
import json
import pickle as pkl
import os,sys

class RNN(nn.Module):
    def __init__(self,input_size=1,hidden_size=32,num_layers=1,type="LSTM"):
        super(RNN, self).__init__()
        if type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,  # rnn hidden unit
                num_layers=num_layers,  # number of rnn layer
                batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
        elif type =="GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,  # rnn hidden unit
                num_layers=num_layers,  # number of rnn layer
                batch_first=True,
                # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
        else:
            print("Error: unknown type {}".format(type),file=sys.stderr)
            sys.exit(1)

        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        outs = self.out(r_out)

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # outs = outs.view(-1, TIME_STEP, 1)
        # return outs, h_state
        # or even simpler, since nn.Linear can accept inputs of any dimension
        # and returns outputs with same dimension except for the last

        return outs, h_state


if __name__ == "__main__":
    # torch.manual_seed(1)    # reproducible

    # Hyper Parameters
    TIME_STEP = 10  # rnn time step
    INPUT_SIZE = 1  # rnn input size
    LR = 0.02  # learning rate

    # show data
    steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)  # float32 for converting torch FloatTensor
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    plt.plot(steps, y_np, 'r-', label='target (cos)')
    plt.plot(steps, x_np, 'b-', label='input (sin)')
    plt.legend(loc='best')
    plt.show()

    rnn = RNN()
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()

    h_state = None  # for initial hidden state

    plt.figure(1, figsize=(12, 5))
    plt.ion()  # continuously plot

    for step in range(100):
        start, end = step * np.pi, (step + 1) * np.pi  # time range
        # use sin predicts cos
        steps = np.linspace(start, end, TIME_STEP, dtype=np.float32,
                            endpoint=False)  # float32 for converting torch FloatTensor
        x_np = np.sin(steps)
        y_np = np.cos(steps)

        x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
        y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

        prediction, h_state = rnn(x, h_state)  # rnn output

        # !! next step is important !!
        h_state = tuple(h.data for h in h_state)  # repack the hidden state, break the connection from last iteration

        loss = loss_func(prediction, y)  # calculate loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        # Plotting
        plt.plot(steps, y_np.flatten(), 'r-')
        plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
        plt.draw()
        plt.pause(0.05)

    plt.ioff()
    plt.show()

def make_sample(value):
    return torch.from_numpy(np.array([value], dtype=np.float32)[np.newaxis, :, np.newaxis])

def predict_rnn(model, n_future, h_state, x_init, ytrue=None,enplot=True,enliveplot=False):
    if enliveplot:
        plt.figure()
        plt.ion()

    x_hat = np.zeros(n_future)
    previous = make_sample(x_init)
    for i in range(n_future):
        prediction, h_state = model(previous, h_state)  # rnn output
        if isinstance(h_state, tuple):
            h_state = tuple(h.data for h in h_state)
        else:
            h_state = h_state.data  # repack the hidden state, break the connection from last iteration

        x_hat[i] = prediction[0, 0, 0].detach().numpy()
        previous = prediction.data
        if enliveplot:
            if not (ytrue is None):
                plt.plot(i, ytrue[i], "k.", label="True")
            plt.plot(i, x_hat[i], "r.", label="Predicted")
            plt.draw()
            plt.pause(0.001)

    if enliveplot:
        plt.ioff()

    if enplot:
        plt.figure()
        plt.plot(ytrue, "k.", label="True")
        plt.plot(x_hat, "r.", label="predicted")
        plt.legend()
        plt.show()

    err=None
    if not (ytrue is None):
        err = ((ytrue-x_hat)**2).mean()
    return x_hat, h_state, err


def train_rnn(X, Y, verbose=False, enplot=True, enliveplot=False, val_cycle=20, n_epochs=1, lr=1e-3, hidden_size=8, num_layers=1,type="LSTM"):
    model = RNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers, type=type)
    model.train()
    xtrain, ytrain, yval = get_cycle(X, Y, val_cycle)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimize all cnn parameters
    loss_func = nn.MSELoss()
    T = xtrain.shape[0]

    err = []

    #if enplot:
    #    xhat = np.zeros(xtrain.shape[0])

    if enliveplot:
        plt.figure(1, figsize=(12, 5))
        plt.ion()  # continuously plot

    #snr_db = 30
    #pnoise = xtrain[:, 1].var()/(10**(snr_db/10))
    xdata = xtrain[:, 1]
            #+ np.random.randn(xtrain.shape[0]) * np.sqrt(pnoise)

    x_in = torch.from_numpy(xdata.astype(np.float32)[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
    BATCH_SIZE = 32
    batches = [(x_in[:, i:min([xdata.shape[0]-1, i+BATCH_SIZE]), :], x_in[:, i+1:min([xdata.shape[0],i+1+BATCH_SIZE]), :]) for i in range(0, T-1,BATCH_SIZE)]

    # batches = [(xx.permute(1,0,2),yy.permute(1,0,2)) for xx,yy in batches]
    with open("training_log_{}_val_cycle_{}.log".format(type, val_cycle), "a") as tr_log:
        print("Model Configuration:\n", file=tr_log)
        print("Val_cycle: {}, Hidden_size: {}, Num_layers: {}, Num_epochs: {}, lr: {}, rnntype: {}\n".format(
            val_cycle, hidden_size, num_layers, n_epochs, lr, type), file=tr_log)
        for epoch in range(n_epochs):
            h_state = None  # for initial hidden state

            epoch_err = []
            for x_in, y_in in batches:
                # x_in = torch.from_numpy(np.array([xdata[t]], dtype=np.float32)[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
                # y_in = torch.from_numpy(np.array([xdata[t + 1]], dtype=np.float32)[np.newaxis, :, np.newaxis])

                prediction, h_state = model(x_in, h_state)  # rnn output

                # !! next step is important !!
                if isinstance(h_state, tuple):
                    h_state = tuple(h.data for h in h_state)
                else:
                    h_state = h_state.data  # repack the hidden state, break the connection from last iteration

                loss = loss_func(prediction, y_in)  # calculate loss
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward(retain_graph=True)  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                epoch_err.append(loss.item())
                # if enplot:
                #    xhat.append(prediction[:, 0, 0].squeeze().detach().numpy())

                #if enliveplot:
                #    plt.plot(t + 1, y_in[0, 0, 0], "k.")
                #    plt.plot(t + 1, prediction[0, 0, 0].detach().numpy(), "r.")
                #    plt.draw()
                #    plt.pause(0.001)

            err.append(np.mean(epoch_err))
            if (verbose == True) and (epoch % 5 == 4 or epoch == 0):
                print("Epoch:{}, Training MSE:{}".format(epoch+1, err[-1]), file=tr_log)

    if enliveplot:
        plt.ioff()

    if enplot:
       # plt.figure()
       # plt.plot(xtrain[:, 1], "k.")
        #plt.plot(xhat, "r.")
        #plt.show()

        plt.figure()
        plt.plot(err)
        plt.show()

    model.eval()
    yval_hat, h_state, eval_err = predict_rnn(model, yval.shape[0],
                                              h_state,
                                              xtrain[-1, 1],
                                              ytrue=yval[:, 1],
                                              enliveplot=enliveplot,
                                              enplot=enplot)

    return model, h_state, err[-1], eval_err, yval_hat


import itertools


def reshape_h(h_state, num_layers):
    if isinstance(h_state, tuple):
        return tuple(h[:, -1, :].reshape(num_layers, 1, -1) for h in h_state)
    else:
        return h_state[:, -1, :].reshape(num_layers, 1, -1)


def field_crossprod(f):
    """return list of dictionnary with all possible combinations of the items inside the list values."""
    keys = list(f.keys())
    l = list(itertools.product(*[f[k] for k in keys]))
    out = [{keys[i]:ll[i] for i in range(len(keys))} for ll in l]
    return out


def train_and_predict_RNN(X, Y, enplot=False, n_future=120, val_cycles=None, dataset="dynamo"):
    if val_cycles is None:
        val_cycles = [20]

 #   k = 20
#    n = 30 * k
    if dataset == "dynamo":
    # For Dynamo 
        params = {"val_cycle": [72, 73, 74],
              "hidden_size": [8, 16, 32],
              "num_layers": [1,2],
              "n_epochs": [50, 100],
              "lr": [1e-2, 1e-3],
              "type": ["GRU", "LSTM"]}

    elif dataset=="solar":
        params = {"val_cycle": [20, 21, 22],
              "hidden_size": [8, 16, 32],
              "num_layers": [1,2],
              "n_epochs": [50, 100],
              "lr": [1e-2, 1e-3],
              "type": ["GRU", "LSTM"]}
    else:
        print("(error) > Dataset {} unknown. Exit.".format(dataset))
        sys.exit(1)

    #if os.path.isfile("rnn_crossval_dynamo.pkl"):
    #    with open("rnn_crossval_dynamo.pkl", "rb") as fp:
    #        errors = pkl.load(fp)
    crossval_file="crossval_files/rnn_crossval_{}.pkl".format(dataset)
    if os.path.isfile(crossval_file):
        with open(crossval_file, "rb") as fp:
            errors = pkl.load(fp)

    else:

        d_l = field_crossprod(params)

        # timeline = np.linspace(0, np.pi * 2 * k, n, dtype=np.float32)  # float32 for converting torch FloatTensor
        # xx = np.sin(timeline)
        # xtrain = np.concatenate([timeline[:, None], xx[:, None]], axis=1).astype(np.float32)

        tr_err = np.zeros(tuple(len(params[k]) for k in params.keys()))
        val_err = np.zeros(tuple(len(params[k]) for k in params.keys()))

        # tr_err = np.zeros((len(d_l), 1))
        # val_err = np.zeros((len(d_l), 1))

        params_list = []
        for i, opts in enumerate(d_l):
            idx = tuple(params[k].index(opts[k]) for k in opts.keys())
            print(i+1, "/", len(d_l), ":", json.dumps(opts))
            params_list.append(json.dumps(opts))
            _, _, train_err, validation_err, _ = train_rnn(X, Y, verbose=True, enplot=False, enliveplot=False, **opts)
            tr_err[idx] = train_err
            val_err[idx] = validation_err
            # tr_err[i] = train_err
            # val_err[i] = validation_err
            # break

        errors = {"params": params, "params_list_dl": params_list, "tr_err": tr_err, "val_err": val_err}
        # errors = {"params": params, "tr_err": tr_err, "val_err": val_err}
        # with open("rnn_crossval_dynamo.pkl", "wb") as fp:
        #     pkl.dump(errors,fp)

        with open(crossval_file, "wb") as fp:
            pkl.dump(errors, fp)
    
    best_idx = np.unravel_index(np.argmin(errors["val_err"].mean(0), axis=None), errors["val_err"].shape)
    # best_idx = np.argmin(errors["val_err"].mean(0))

    best_opts = {k: errors["params"][k][ii] for k, ii in zip(errors["params"].keys(),best_idx)} #{k:params[i] for }
    # best_opts = errors["params_list_dl"][best_idx]

    print("Best options are:\n{}".format(best_opts))

    '''
    best_opts["val_cycle"] = len(Y)-1

    model, h_state_out, train_err, test_err, test_predictions = train_rnn(X, Y, verbose=False, enplot=enplot, enliveplot=False,
                                                              **best_opts)

    # predict_rnn(model, yval.shape[0], h_state, xtrain[-1, 1], ytrue=yval, enliveplot=False,enplot=True)
    xtrain_hat, h_state, yfuture_hat = predict_rnn(model, n_future, h_state_out, Y[-1][-1, 1], ytrue=None, enliveplot=True)

    # Test
    # h_state = None
    #xtrain_hat, h_state = predict_rnn(model, xtrain.shape[0], h_state, xtrain[0, 1], ytrue=xtrain[:, 1], enplot=True)
#    y_future, _ = predict_rnn(model, 11*12, h_state, ytest[-1], ytrue=None, enliveplot=False, enplot=True)
    '''


