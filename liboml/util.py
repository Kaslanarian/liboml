import numpy as np
from torch import nn


def mlp(*layer_args, activation=nn.ReLU):
    if len(layer_args) < 2:
        raise ValueError("MLP must have at least 2 layers!")
    layer_list = [nn.Linear(layer_args[0], layer_args[1])]
    for i in range(2, len(layer_args)):
        layer_list.append(activation())
        layer_list.append(nn.Linear(layer_args[i - 1], layer_args[i]))
    return nn.Sequential(*layer_list)


def sigmoid(x):
    y = np.empty_like(x)
    y[x >= 0] = 1 / (1 + np.exp(-x[x >= 0]))
    y[x < 0] = 1 - 1 / (1 + np.exp(x[x < 0]))
    return y
