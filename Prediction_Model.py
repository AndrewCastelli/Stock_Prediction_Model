import torch
from torch import nn


class NN(nn.Module):
    """
    [Long Short Term Memory] PyTorch LSTM class to instantiate model for Stock Predictions.
    LSTM Model is essentially a Recurrent Neural Network with the Cell State above Hidden State.
    :param input_dim: Number of features to be input (1 price per day).
    :param hidden_dim: Number of hidden layers to run batches through.
    :param num_layers: Number of lstm and/or linear layers.
    :param output_dim: Number of features to be output (1 price per day).
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(NN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM will handle value weights for other gates
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Start by populating cell and hidden states with 0's
        cell_s = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        hidden_s = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        input_fc, (_h, _c) = self.lstm(x, (hidden_s.detach(), cell_s.detach()))
        # Set output_fc final step of hidden_s (the one that matters)
        output_fc = self.fc(input_fc[:, -1, :])

        return output_fc
