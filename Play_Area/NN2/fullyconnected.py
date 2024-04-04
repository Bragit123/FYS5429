
import matplotlib.pyplot as plt
from jax import random, vmap
import numpy as np
from funcs import RELU, sigmoid, derivate
from scheduler import *
from typing import Callable


class FullyConnected:
    """
    Fully connected layer. Defines a number of output nodes, and computes a
    weighted sum of the input for each output, and runs the resulting value
    through an activation function.

    ## Attributes:
        - input_length (int): Number of inputs. This corresponds to the
          number of nodes in the previous layer of the neural network.
        - output_length (int): Number of outputs to produce from the layer.
        - weights_size (tuple): Shape of weights array.
        - bias_length (int): Number of biases.
        - weights (ndarray): Two-dimensional array of shape weights_size. With
          one row for each input value, and one column for each output value of
          the layer.
        - bias (ndarray): One-dimensional array containing the biases, with one
          bias for each output value of the layer.
    """
    def __init__(self, input_length: int, output_length: int, act_func: Callable[[np.ndarray],np.ndarray], seed: int = 100):
        """
        Constructor

        ## Parameters:
        - input_length (int): Number of inputs. This corresponds to the
          number of nodes in the previous layer of the neural network.
        - output_length (int): Number of outputs to produce from the layer.
        - seed (int): Seed for generating random initial weights and biases in
          the layer.
        """
        self.input_length = input_length
        self.output_length = output_length
        self.weights_size = (self.input_length, self.output_length)
        self.bias_length = self.output_length
        self.act_func = act_func
        # self.scheduler_weights = AdamMomentum(0.01, 0.9, 0.999, 0.01) #temporary
        # self.scheduler_bias = AdamMomentum(0.01, 0.9, 0.999, 0.01) #temporary
        # self.scheduler_weights = Adam(0.01, 0.9, 0.999) #temporary
        # self.scheduler_bias = Adam(0.01, 0.9, 0.999) #temporary
        self.scheduler_weights = Adagrad(0.01) #temporary
        self.scheduler_bias = Adagrad(0.01) #temporary

        self.seed = seed

        ## Initialize random weights and biases.
        self.reset_weights()

    def reset_weights(self):
        rand_key = random.PRNGKey(self.seed)
        self.weights = random.normal(key=rand_key, shape=self.weights_size)
        self.bias = random.normal(key=rand_key, shape=(self.bias_length,))*0.1
    
    def reset_schedulers(self):
        self.scheduler_weights.reset()
        self.scheduler_bias.reset()

    def feed_forward(self, input: np.ndarray):
        """
        Feeds input forward through the neural network.

        ## Parameters:
            - input (ndarray): Two-dimensional input array to be fed forward through
            the neural network. The first axis corresponds to the different
            inputs, and the second axis corresponds to the input values.

        ## Returns:
            ndarray: Two-dimensional array containing the output from the layer.
            The axes correspond to the same axes as the input.
        """
        self.input = input # Save input for use in backpropagate().
        #num_inputs = np.shape(input)[0]


        self.z = input @ self.weights + self.bias

        # calculate a, add bias
        output = self.act_func(self.z)

        return output

    def backpropagate(self, dC_doutput: np.ndarray, lmbd: float = 1e-4):
        """
        Backpropagates through the layer to find the partial derivatives of the
        cost function with respect to each weight, bias and input value. The
        derivatives with respect to weights and biases are used to update the
        weights and biases using gradient descent, while the derivatives with
        respect to the input is returned for use in backpropagation through
        previous layers.

        ## Parameters
            - dC_doutput (ndarray): Two-dimensional array containing the
              partial derivatives of the cost function with respect to every
              output value from this layer. The first axis is the different
              inputs, and the second axis corresponds to every partial
              derivative.
            - lmbd (float): WILL BE CHANGED TO SCHEDULER.

        ## Returns
            ndarray: Partial derivatives of the cost function with respect to
            every input value to this layer.
        """
        input = self.input
        grad_act = vmap(vmap(derivate(self.act_func)))
        input_size = np.shape(input)

        ## Initialize weights and biases.
        #grad_weights = np.zeros(self.weights_size)
        #grad_biases = np.zeros(np.shape(self.bias))
        #grad_input = np.zeros(input_size)

        #dC_da = dC_doutput * grad_act(self.z)
        delta_matrix = dC_doutput * grad_act(self.z)
        grad_weights = input.T @ delta_matrix/input_size[0]
        grad_biases = np.sum(delta_matrix, axis=0).reshape(1,np.shape(delta_matrix)[1])/input_size[0]
        grad_input = delta_matrix@self.weights.T

        # print(f"Before : {grad_weights}")
        self.grad_weights = grad_weights + self.weights * lmbd
        # print(f"After : {grad_weights}")

        #for i in range(self.input_length):
            #for j in range(self.output_length):
                ## Compute the gradients.
            #    grad_weights = grad_weights.at[i,j].set(np.sum(dC_doutput[:,j] * grad_act(self.z[:,j]) * input[:,i])/input_size[0])
            #    grad_biases = grad_biases.at[j].set(np.sum(1*grad_act(self.z[:,j])*dC_doutput[:,j])/input_size[0])

            #    zero_matrix = np.zeros(input_size)
            #    grad_input += zero_matrix.at[:,i].set(dC_doutput[:,j] * grad_act(self.z[:,j]) * self.weights[i,j])


        # scheduler_weights = Adam(0.001, 0.9, 0.999)
        
        # print("Weights")
        update_weights = self.scheduler_weights.update_change(self.grad_weights)
        print(update_weights)
        self.weights -= update_weights
        # scheduler_bias = Adam(0.001, 0.9, 0.999)
        # print("Bias")
        update_bias = self.scheduler_bias.update_change(grad_biases)
        # print(update_bias)
        self.bias -= update_bias
        # print()

        return grad_input
