import numpy as np


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))


r = sigmoid_derivative(0.7)

print(r)