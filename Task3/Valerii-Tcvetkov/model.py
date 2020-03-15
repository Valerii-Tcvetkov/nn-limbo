import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        # raise Exception("Not implemented!")

        self.out_classes = n_output_classes
        image_width, image_height, in_channels = input_shape

        self.layers = [
            ConvolutionalLayer(in_channels, conv1_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            Flattener(),
            FullyConnectedLayer(4 * conv2_channels, n_output_classes)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        # raise Exception("Not implemented!")

        for param in self.params().values():
            param.grad = np.zeros_like(param.value)

        for layer in self.layers:
            X = layer.forward(X)

        loss, grad = softmax_with_cross_entropy(X, y)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        # raise Exception("Not implemented!")

        for layer in self.layers:
            X = layer.forward(X)

        # raise Exception("Not implemented!")
        # return pred

        pred = np.argmax(X, axis=1)

        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        # raise Exception("Not implemented!")

        result = {
            'W0': self.layers[0].params()['W'],
            'B0': self.layers[0].params()['B'],
            'W3': self.layers[3].params()['W'],
            'B3': self.layers[3].params()['B'],
            'W7': self.layers[7].params()['W'],
            'B7': self.layers[7].params()['B']
        }

        return result
