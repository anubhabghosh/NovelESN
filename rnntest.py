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
        h_state = h_state.data  # repack the hidden state, break the connection from last iteration

        loss = loss_func(prediction, y)  # calculate loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        # Plotting
        plt.plot(steps, y_np.flatten(), 'r-')
        plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
        plt.draw();
        plt.pause(0.05)

    plt.ioff()
    plt.show()

def train_and_predict_RNN(model, xtrain, ytrain, ytest,tau=1):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # optimize all cnn parameters
    loss_func = nn.MSELoss()

    h_state = None  # for initial hidden state
    n_predict=ytest.shape[0]

    plt.close()
    T = xtrain.shape[0]
    err=[]
    for epoch in range(10):
        loss=0
        for t in range(1, T-1):

            x = torch.from_numpy(xtrain[:t,1].reshape(1,-1,1).astype(np.float32))  # shape (batch, time_step, input_size)
            y = torch.from_numpy(xtrain[t:t+tau,1].reshape(1,-1,1).astype(np.float32))

            prediction, h_state = model(x, h_state)  # rnn output

            # !! next step is important !!
            h_state = tuple(h.data for h in h_state)  # repack the hidden state, break the connection from last iteration

            loss += loss_func(prediction[:, -1, :], y)  # calculate loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        err.append(loss.item())

        print(epoch, np.mean(err[-T:]))

    model.eval()
    out=torch.from_numpy(np.zeros((T+n_predict),dtype=np.float32))
    out[:T]=torch.from_numpy(xtrain[:,1].astype(np.float32))
    for ipredict in range(n_predict):
        prediction, h_state = model(out[ipredict:T+ipredict].reshape(1,-1,1), h_state)
        out[T+ipredict]=prediction[0,-1,0]
    prediction = out[-n_predict:].detach().numpy()

    plt.figure();plt.subplot(311)
    plt.plot(ytest);
    plt.plot(prediction)
    plt.title("Training data size: {}".format(xtrain.shape[0]))
    plt.subplot(312)
    plt.plot(err)
    plt.subplot(313)
    plt.plot(xtrain[:,1])
    plt.savefig("Training_data_size{}.png".format(xtrain.shape[0]))
    return prediction