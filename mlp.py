"""
neural_network.py
~~~~~~~~~~

Module is for building a classic dense neural network

Weights and biases are initialized randomly according to a normal distribution
Training of the network is done using back propagation
"""

import numpy as np
from numba import jit
from random import random


class MLP:
    """Neural Network class"""

    def __init__(self, shape=None):
        """
        Constructor for the MLP. Takes the shape of the network
        :param shape: list of integers, describes how many layers and neurons by layer the network has
        """
        self.shape = shape
        self.biases = []
        self.weights = []
        self.activations = []
        self.derivatives = []
        if shape:
            for y in shape[1:]:  # biases random initialization
                self.biases.append(np.random.randn(y, 1))
            for x, y in zip(shape[1:], shape[:-1]):  # weights random initialization
                self.weights.append(np.random.randn(x, y))
            for y in shape[:]:  # activations zeros initialization
                self.activations.append(np.zeros((y, 1)))
            for x, y in zip(shape[1:], shape[:-1]):  # derivatives zeros initialization
                self.derivatives.append(np.zeros((x, y)))

    def feed_forward(self, a):
        """
        Main function, takes an input vector and calculate the output by propagation through the network

        :param a: column of integers, inputs for the network
        :return: column of integers, output neurons activation
        """
        self.activations[0] = a
        for idx, (b, w) in enumerate(zip(self.biases, self.weights)):
            a = sigmoid(np.dot(w, a) + b)
            self.activations[idx + 1] = a
        return a

    def back_prop(self, error, verbose=False):
        """
        Main function, takes an input vector and calculate the error by back propagation through the network

        :param error: column of integers, errors from the target
        :param verbose: boolean, to print or not print the current derivatives
        :return: column of integers, output error
        """
        for idx in reversed(range(len(self.derivatives))):

            next_activations = self.activations[idx + 1]
            delta = error * sigmoid_derivative(next_activations)
            current_activations = self.activations[idx].T
            self.derivatives[idx] = np.dot(delta, current_activations)
            error = np.dot(self.weights[idx].T, delta)

            if verbose:
                print(f'Derivatives for {idx} is {self.derivatives[idx]} and error is {error}')
        return error

    def gradient_descent(self, learning_rate):
        """
        Gradient Descent function, takes a learning rate and performs the gradient descent algorithm\
         on the weights with the calculated derivatives
        :param learning_rate: integer, the step size at each iteration
        :return: column of integers, output error
        """
        for idx in range(len(self.weights)):
            weights = self.weights[idx]
            derivatives = self.derivatives[idx]
            weights += derivatives * learning_rate

    def train(self, input_list, target_list, epochs, learning_rate):
        """
        Train function, takes in learning rate,input_list, target_list and the no of epochs and \
         trains the network with the given inputs and labelled data
        :param input_list: list of integers, the input data
        :param target_list: list of integers, the labelled op data
        :param epochs: integer, the no of epochs to be performed
        :param learning_rate: integer, the step size for learning
        """

        for i in range(epochs):
            sum_errors = 0

            for j, (inp, target) in enumerate(zip(input_list, target_list)):

                net_inp = np.array([[inp[0]], [inp[1]]])

                output = self.feed_forward(net_inp)

                error = target - output

                self.back_prop(error)

                self.gradient_descent(learning_rate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)

                # Epoch complete, report the training error
                print("Error: {} at epoch {}".format(sum_errors / len(input_list), i + 1))

        print("Training complete!")
        print("=====")

    @staticmethod
    def _mse(tar, out):
        """
        The mse function, calculates the mean squared error
        :param tar: int, actual target
        :param out: int, predicted output
        :return: int, mean squared error
        """
        return np.average((tar - out) ** 2)


@jit(nopython=True)
def sigmoid(z):
    """
    The sigmoid function, classic neural net activation function
    @jit is used to speed up computation
    """
    return 1.0 / (1.0 + np.exp(-z))


@jit(nopython=True)
def sigmoid_derivative(x):
    """
    The sigmoid derivative function, used for back prop
    @jit is used to speed up computation
    """
    return x * (1.0 - x)


if __name__ == '__main__':
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])  # inputs for the network \
    # (consist's of a list of two random variables * 1000)

    targets = np.array([[[i[0] + i[1]]] for i in inputs])  # corresponding sum of the two variables

    network_shape = [2, 5, 1]  # shape of the network(can be changed to the user's wish)

    mlp = MLP(network_shape)  # initialise the network class

    mlp.train(inputs, targets, epochs=50, learning_rate=0.1)  # train the network using the train method

    #  create dummy data
    test_input = np.array([[0.55], [0.25]])
    test_target = np.array([[0.80]])

    # get a prediction
    predicted_output = mlp.feed_forward(test_input)

    print()
    print(f'Network believes that {test_input[0][0]} + {test_input[1][0]} is equal to\
 {predicted_output[0][0]} and the actual output is {test_target[0][0]}')
