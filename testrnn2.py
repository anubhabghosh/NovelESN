import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from rnntest import RNN

if __name__ == "__main__":
    k = 20
    n = 30 * k

    timeline = np.linspace(0, np.pi * 2 * k, n, dtype=np.float32)  # float32 for converting torch FloatTensor
    x = np.sin(timeline)
    y = np.cos(timeline)

    plt.figure()
    plt.plot(timeline, x)
    plt.show()
    n_batches = 100

    TIME_STEP = n // n_batches

    model = RNN(1, 10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # optimize all cnn parameters
    loss_func = nn.MSELoss()


    err = []
    plt.figure()
    plt.ion()
    for epoch in range(1):
        h_state = None  # for initial hidden state
        for i, t in enumerate(timeline[1:]):
            x_in = torch.from_numpy(np.array([x[i]])[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
            y_in = torch.from_numpy(np.array([x[i+1]])[np.newaxis, :, np.newaxis])

            prediction, h_state = model(x_in, h_state)  # rnn output

            # !! next step is important !!
            if isinstance(h_state,tuple):
                h_state = tuple(h.data for h in h_state)
            else:
                h_state = h_state.data  # repack the hidden state, break the connection from last iteration

            loss = loss_func(prediction, y_in)  # calculate loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward(retain_graph=True)  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            err.append(loss.item())
            plt.plot(t + 1, y_in[0, 0, 0], "k.")
            plt.plot(t + 1, prediction[0, 0, 0].detach().numpy(), "r.")
            plt.draw()
            plt.pause(0.001)


    print(err[-1])
    plt.ioff()

    # test
    h_state = None
    previous = torch.from_numpy(np.array([0], dtype=np.float32)[np.newaxis, :, np.newaxis])
    model.eval()
    plt.figure()
    plt.ion()
    for i, t in enumerate(timeline[1:]):
        prediction, h_state = model(previous, h_state)  # rnn output
        if isinstance(h_state, tuple):
            h_state = tuple(h.data for h in h_state)
        else:
            h_state = h_state.data  # repack the hidden state, break the connection from last iteration

        #plt.plot(t + 1, y_in[0, 0, 0], "k.")
        plt.plot(t + 1, prediction[0, 0, 0].detach().numpy(), "r.")
        plt.draw()
        plt.pause(0.001)
        previous = prediction.data
    plt.ioff()

