from typing import List

import numpy as np


def random_start_weights(list_of_layer_dimensions):
    """
    creates random weights for each edge in computational graph of size 'list_of_layer_dimensions'

    :param list_of_layer_dimensions: array of dimensions of particular layer starting with input layer,
    through hidden layers, out with output layer.
    :return: list of ndarrays - list of 2n ndarrays - weights from neurons in layer a to a + 1, where a starts with 0
    (input layer) and goes through hidden layers until output layer a = L
    """
    weights = []
    for idx in range(len(list_of_layer_dimensions) - 1):
        first_layer_dim = list_of_layer_dimensions[idx]
        second_layer_dim = list_of_layer_dimensions[idx + 1]
        weights.append(np.random.rand(second_layer_dim, first_layer_dim))
    return weights


def random_start_biases(list_of_layer_dimensions):
    """
    creates random start biases for given layers of neural network

    :param list_of_layer_dimensions: array of dimensions of particular layer starting with input layer,
    through hidden layers, out with output layer.
    :return: list of ndarrays -  biases for each dim except in input layer
    """
    biases = []
    for dim in list_of_layer_dimensions[1:]:
        biases.append(np.array(np.random.rand(dim), dtype=np.float))
    return biases


def sigmoid(x):
    """
    activation function - simple and classical sigmoid
    """
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    """
    Derivative of sigmoid function
    """
    return sigmoid(x) * (1 - sigmoid(x))


class FeedForwardNN:
    """
    Feed forward neural network with:
    - cost function c(y*, y) = 0.5 * ||y* - y||^2
    - constant minibatch size of 1
    - default learning rate of 0.01
    """

    def __init__(self,
                 layer_dimensions: List[int],
                 activation_function=sigmoid,
                 activation_derivative=dsigmoid,
                 learning_rate=0.01) -> None:
        super().__init__()
        self.layer_dimensions = layer_dimensions
        self.w = random_start_weights(layer_dimensions)
        self.b = random_start_biases(layer_dimensions)
        self.activations = []
        self.z = []
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.eta = learning_rate

    def forward(self, input_vector: np.array) -> np.array:
        """
    Executes a forward step.

    Given input vector, computes activation values of all neurons and memoizes
    them, next returns activation vector of neurons in the last layer

    :param input_vector: input vector of shape(0, layer_dimensions[0])
    :return: returns activation vector of output layer
    """
        for idx, layer in enumerate(self.layer_dimensions):
            if idx is 0:
                self.activations.append(input_vector)
            else:
                previous_a = self.activations[-1]
                zeta = np.dot(self.w[idx - 1], previous_a) + self.b[idx - 1]
                self.z.append(zeta)
                a = self.activation_function(zeta)
                self.activations.append(a)
        return self.activations[-1]

    def fit(self, xs, ys):
        """
        Fits the model against the learning set
        :param xs: data
        :param ys: labels
        :return: nothing, the function changes parameters (biases and weights)
        to fit the network against train data
        """
        for x, y in zip(xs, ys):
            self._fit_single(x, y)

    def _fit_single(self, x, y):
        predicted = self.forward(x)
        gradient_weights, gradient_biases = self.backward(y, predicted)
        self.update_weights_and_biases(gradient_biases, gradient_weights)

    def update_weights_and_biases(self, grad_biases, grad_weights):
        for i in range(len(self.w)):
            self.w[i] = self.w[i] - (self.eta * grad_weights[i])
        for i in range(len(self.b)):
            self.b[i] = self.b[i] - (self.eta * grad_biases[i])

    def backward(self, real, predicted):
        """
        Executes backpropagation algorithm fitting weights and biases against predicted value
        """
        gradient_weights = [np.zeros(weight.shape) for weight in self.w]
        gradient_biases = [np.zeros(bias.shape) for bias in self.b]

        # weights and biases from layer L-1 to layer L
        partial_cost_last_a = predicted - real
        delta = self.activation_derivative(self.z[-1])
        last_layer_neuron_factor = partial_cost_last_a * delta
        last_layer_weights_gradient = np.dot(last_layer_neuron_factor.reshape((-1, 1)),
                                             self.activations[-2].reshape((1, -1)))
        gradient_weights[-1] = last_layer_weights_gradient
        gradient_biases[-1] = last_layer_neuron_factor

        # weights for other hidden layers
        for i in range(len(self.activations) - 2, 0, -1):
            z_and_old_delta = self.z[i] * delta
            delta = np.dot(self.w[i].transpose(), z_and_old_delta)
            squashed_derivative = self.activation_derivative(self.z[i - 1])
            neuron_error = squashed_derivative * delta
            gradient_weights[i - 1] = np.dot(neuron_error.reshape((-1, 1)), self.activations[i - 1].reshape((1, -1)))
            gradient_biases[i - 1] = neuron_error
        return gradient_weights, gradient_biases
