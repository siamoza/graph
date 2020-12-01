import numpy as np
import torch as tr


def sigmoid(x):
    # activation f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def feedforward(self, input_):
        return sigmoid(np.dot(self.weight, input_) + self.bias)


print(tr.cuda.is_available())


