"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
numpy
"""
from scipy import signal
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR

from src.utils import get_cycle, concat_data
import json
import pickle as pkl
import os, sys


class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, type="LSTM"):
        super(RNN, self).__init__()
        if type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,  # rnn hidden unit
                num_layers=num_layers,  # number of rnn layer
                batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )
        elif type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,  # rnn hidden unit
                num_layers=num_layers,  # number of rnn layer
                batch_first=True,
                # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )

        elif type == "RNN":
            self.rnn = nn.RNN(
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
        # Initialize hidden state with zeros
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

    # previous = make_sample(x_init)
    previous = x_init.reshape(1, -1, 1)
    p = previous.shape[1]
    x_hat[:p] = previous.detach().cpu().numpy()[0, 0, 0]

    for i in range(p, n_future):
        prediction, h_state = model(previous, h_state)  # rnn output
        x_hat[i] = prediction.detach().cpu().numpy()[0, 0, 0]
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

    err = None
    if not (ytrue is None):
        err = ((ytrue-x_hat)**2).mean()
    return x_hat, h_state, err


def train_rnn(X, Y, verbose=False, enplot=True, enliveplot=False, val_cycle=20, n_epochs=1, lr=1e-3, hidden_size=8, num_layers=1,type="LSTM"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers, type=type).to(device)
    model.train()

    if val_cycle != -1:
        xtrain_data, ytrain_data, yval_data = get_cycle(X, Y, val_cycle)

    else:
        # All cycles used for training
        xtrain_data = sum(Y, [])
        # xtrain, ytrain, yval = get_cycle(X, Y, val_cycle)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimize all cnn parameters
    #lmbda = lambda epoch: 0.95
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    loss_func = nn.MSELoss()

    err = []
    batch_size = 32
    #if enplot:
    #enliveplot = True
    #snr_db = 30
    #pnoise = xtrain[:, 1].var()/(10**(snr_db/10))
    #xtrain_data = concat_data(xtrain_tmp)
    #ytrain_data = concat_data(ytrain_tmp)
    # yval_data = concat_data(yval_tmp)
    T = xtrain_data.shape[0]

    x_train = torch.from_numpy(xtrain_data[:, 1].reshape(1, -1, 1)).type(torch.Tensor).to(device)
    x_val = torch.from_numpy(yval_data[:, 1].reshape(1, -1, 1)).type(torch.Tensor).to(device)

    # y_train = torch.from_numpy(ytrain_data).type(torch.Tensor)
    xhat = np.zeros(T)
    xhat[0] = xtrain_data[0, 1]
    # train = torch.utils.data.TensorDataset(xtrain_data, ytrain_data)
    # test = torch.utils.data.TensorDataset(x_test, y_test_data)

    if enliveplot:
        plt.figure()
        plt.ion()

    with open("training_log_{}_val_cycle_{}.log".format(type, val_cycle), "a") as tr_log:
        print("Model Configuration:\n", file=tr_log)
        print("Val_cycle: {}, Hidden_size: {}, Num_layers: {}, Num_epochs: {}, lr: {}, rnntype: {}\n".format(
            val_cycle, hidden_size, num_layers, n_epochs, lr, type), file=tr_log)

        previous = x_train[:, 0, :].reshape(1, -1, 1)

        n_teaching_seq = 10
        x_wave = np.linspace(0, 1, T - 1, endpoint=False)
        Pteach=[]
        #plt.close()
        #plt.figure()
        sequences = [0 for _ in range(n_teaching_seq)]
        for i in range(n_teaching_seq):
            teach_schedule = signal.square(2 * np.pi * 5 * x_wave + 30*np.random.rand(1)[0], 1-i/n_teaching_seq )/2+1/2
          #   plt.subplot(211)
          #  plt.plot(teach_schedule+i)
            sequences[i] = teach_schedule.reshape(1, -1).astype(bool)
        # teach = np.concatenate(sequences, axis=0)
        # plt.subplot(212)
        # plt.plot(teach.sum(0)/teach.shape[0])
        iseq = 0
        for epoch in range(2000):
            epoch_err = []
            h_state = None
            loss = 0
            if iseq > n_teaching_seq-1:
                break
            for t in range(T-1):
                prediction, h_state = model(previous, h_state)  # rnn output
                h_state = reshape_h(h_state, num_layers)
                loss += loss_func(prediction, x_train[:, t + 1, :].reshape(1, 1, 1))  # calculate loss
                xhat[t+1] = prediction.detach().cpu().numpy()[0, 0, 0]

                if sequences[iseq][0, t]:
                    previous = x_train[:, t + 1, :].reshape(1, 1, 1)
                else:
                    previous = prediction.data

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward(retain_graph=True)  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            scheduler.step()
            epoch_err.append(loss.item()/(T-1))
            err.append(epoch_err[-1])
            Pteach.append(sequences[iseq].sum()/(T-1))
            log_str = "Epoch:{}, Training MSE:{}, p_teach: {}".format(epoch + 1, err[-1], Pteach[-1])
            if (epoch % 2 == 0):
                if verbose:
                    print(log_str, file=tr_log)
                if enliveplot:
                    plt.clf()
                    plt.subplot(411)
                    plt.plot(xtrain_data[:, 1], "k.")
                    plt.plot(xhat, "r.")
                    plt.title(log_str)
                    plt.subplot(412)
                    plt.plot(sequences[iseq][0, :])
                    plt.subplot(413)
                    plt.plot(err)
                    plt.subplot(414)
                    plt.plot(Pteach)

                    plt.draw()
                    plt.pause(0.001)
            if epoch > 10 and all([abs(err[-1]-err[-pback-1]) <= 1e-3 for pback in range(1, 3)]):
                iseq = iseq+1

    if enliveplot:
        plt.ioff()
        plt.savefig("err_liveplot.pdf")
        plt.close()

    if enplot:
       plt.figure()
       plt.plot(xtrain_data[:, 1], "k.")
       plt.plot(xhat, "r.")
       plt.show()

    plt.figure()
    plt.plot(err)
    plt.savefig("training_error.pdf")

    enliveplot=False

    model.eval()

    xtrain_hat, h_state, train_err = predict_rnn(model, xtrain_data.shape[0], None,
                                                 x_train[:, 0, :].reshape(1, 1, 1), ytrue=xtrain_data[:, 1],
                                              enliveplot=False,
                                              enplot=enplot)

    yval_hat, h_state, eval_err = predict_rnn(model, x_val.shape[0], h_state, x_train[:, -1, :],
                                              ytrue=yval_data[:, 1],
                                              enliveplot=enliveplot, enplot=enplot)
    print(err[-1], train_err, eval_err)

    return model, h_state, train_err, eval_err, yval_hat


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
    #   n = 30 * k

    if dataset == "dynamo":
    # For Dynamo 
        params = {"val_cycle": [72, 73, 74],
              "hidden_size": [128, 32, 16, 8],
              "num_layers": [1, 2, 4, 6],
              "n_epochs": [200],
              "lr": [1e-2, 1e-3],
              "type": ["LSTM", "GRU"]}

    elif dataset == "solar":
        params = {"val_cycle": [20, 21, 22],
              "hidden_size": [8, 16, 32],
              "num_layers": [1, 2],
              "n_epochs": [50, 100],
              "lr": [1e-2, 1e-3],
              "type": ["GRU", "LSTM"]}
    else:
        print("(error) > Dataset {} unknown. Exit.".format(dataset))
        sys.exit(1)
    d_l = field_crossprod(params)

    # if os.path.isfile("rnn_crossval_dynamo.pkl"):
    #    with open("rnn_crossval_dynamo.pkl", "rb") as fp:
    #        errors = pkl.load(fp)
    crossval_file = "crossval_files/rnn_crossval_{}.pkl".format(dataset)
    if os.path.isfile(crossval_file):
        with open(crossval_file, "rb") as fp:
            errors = pkl.load(fp)

    else:
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

    best_opts = d_l[0]#{k: errors["params"][k][ii] for k, ii in zip(errors["params"].keys(), best_idx)} #{k:params[i] for }
    # best_opts = errors["params_list_dl"][best_idx]

    print("Best options are:\n{}".format(best_opts))

    # Test cycle
    best_opts["val_cycle"] = len(Y)-1

    model, h_state_out, train_err, test_err, test_predictions = train_rnn(X, Y, verbose=False, enplot=enplot, enliveplot=False,
                                                              **best_opts)

    best_opts["val_cycle"] = - 1
    model, h_state_out, train_err, _, _ = train_rnn(X, Y, verbose=False, enplot=enplot, enliveplot=False,
                                                              **best_opts)

    #predict_rnn(model, yval.shape[0], h_state_out, xtrain[-1, 1], ytrue=yval, enliveplot=False,enplot=True)

    xtrain_hat, h_state, yfuture_hat = predict_rnn(model, n_future, h_state_out, Y[-1][-1, 1], ytrue=None, enliveplot=True)

    # Test
    # h_state = None
    #xtrain_hat, h_state = predict_rnn(model, xtrain.shape[0], h_state, xtrain[0, 1], ytrue=xtrain[:, 1], enplot=True)
#    y_future, _ = predict_rnn(model, 11*12, h_state, ytest[-1], ytrue=None, enliveplot=False, enplot=True)



