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

class RNN(nn.Module):
    def __init__(self,input_size=1,hidden_size=32,num_layers=1):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,  # rnn hidden unit
            num_layers=num_layers,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
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


def predict_rnn(model, n_future, h_state, x_init, ytrue=None,enplot=True,enliveplot=False):
    if enliveplot:
        plt.figure()
        plt.ion()

    x_hat = np.zeros(n_future)
    previous = torch.from_numpy(np.array([x_init], dtype=np.float32)[np.newaxis, :, np.newaxis])
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
        plt.plot(ytrue,"k.",label="True")
        plt.plot(x_hat, "r.", label="predicted")
        plt.show()

    return x_hat, h_state


def train_rnn(model, xtrain, enplot=True,n_epochs=1,enliveplot=False,lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimize all cnn parameters
    loss_func = nn.MSELoss()
    T = xtrain.shape[0]
    err = []
    if enplot:
        xhat = np.zeros(xtrain.shape[0])

    if enliveplot:
        plt.figure(1, figsize=(12, 5))
        plt.ion()  # continuously plot
    for epoch in range(n_epochs):

        h_state = None  # for initial hidden state
        for t in range(T-1):
            x_in = torch.from_numpy(np.array([xtrain[t, 1]], dtype=np.float32)[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
            y_in = torch.from_numpy(np.array([xtrain[t + 1, 1]], dtype=np.float32)[np.newaxis, :, np.newaxis])

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

            err.append(loss.item())
            if enplot:
                xhat[t] = prediction[0, 0, 0].detach().numpy()

            if enliveplot:
                plt.plot(t + 1, y_in[0, 0, 0], "k.")
                plt.plot(t + 1, prediction[0, 0, 0].detach().numpy(), "r.")
                plt.draw()
                plt.pause(0.001)
        print(epoch, np.mean(err[-T:]))
    if enliveplot:
        plt.ioff()

    if enplot:
        plt.figure()
        plt.plot(xtrain[:, 1], "k.")
        plt.plot(xhat,"r.")
        plt.show()

    return model, h_state

def train_and_predict_RNN(model, xtrain, ytrain, ytest, enplot=False, n_future=120):
    k = 20
    n = 30 * k

    timeline = np.linspace(0, np.pi * 2 * k, n, dtype=np.float32)  # float32 for converting torch FloatTensor
    #xtrain = np.concatenate([timeline[:, None], np.sin(timeline)[:,None]],axis=1).astype(np.float32)

    model.train()
    model, h_state_out = train_rnn(model, xtrain, enplot=True, enliveplot=False, n_epochs=10, lr=1e-4)
    model.eval()
    xtrain_hat, h_state = predict_rnn(model, xtrain.shape[0], None, xtrain[0, 1], ytrue=xtrain[:, 1], enliveplot=True)

    # test
    # h_state = None
    #xtrain_hat, h_state = predict_rnn(model, xtrain.shape[0], h_state, xtrain[0, 1], ytrue=xtrain[:, 1], enplot=True)
    ytest_hat, h_state = predict_rnn(model, ytest.shape[0], h_state_out, xtrain[-1, 1], ytrue=ytest, enliveplot=False,enplot=True)
    y_future, _ = predict_rnn(model, 11*12, h_state, ytest[-1], ytrue=None, enliveplot=False,enplot=True)


