# -*- coding: utf-8 -*-
"""
network.py

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A module to implement the SGD learning algorithm for a feedforward neural 
network. Gradients are caculated using backpropagation.
"""
#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):
    def __init__(self, sizes):
        """The list 'sizes' contains the number of neurons in the respective
        layers of the network."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x)
                        for x,y in zip(sizes[:-1],sizes[1:])]
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmod(np.dot(w, a)+b)
        return a
    def SGD(self, training_data, epochs, mini_batch_sizes, eta,
            test_data=None):
            """ Train the neural network using mini-batch stochastic
            gradient descent. The 'training_data' is a list of tuples
            '(x,y)' represeting the training inputs and the desired outputs
            If 'test_data' is provided then the network will be evaluted against
            the test data after each epoch, and partial progress printed
            out. 
            """
            if test_data: n_test = len(test_data)
            n = len(training_data)
            for j in xrange(epochs):
                random.shuffle(training_data)
                mini_batches = [
                    training_data[k:k+mini_batch_sizes]
                    for k in xrange(0, n, mini_batch_sizes)]
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta)
                if test_data:
                    print "Epoch {0}: {1}/{2}".format(
                    j, self.evaluate(test_data), n_test)
                else:
                    print "Epoch {0} complete".format(j)
                
    def update_mini_batch(self, mini_batch, eta):
        """update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch
        The ''mini_batch'' is a list of tuples ''(x,y)'', and ''eta''
        is the learning rate.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]
                    