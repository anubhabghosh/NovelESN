
import torch.nn as nn


def get_lstm(input_size=10, hidden_size=20, num_layers=1, **kwargs):
    return nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)


def get_gru(input_size=10, hidden_size=20, num_layers=1, **kwargs):
    return nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)