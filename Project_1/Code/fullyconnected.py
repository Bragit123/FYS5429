
import matplotlib.pyplot as plt
from jax import random, vmap
import jax.numpy as jnp
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
    def __init__(self, input_length: int, output_length: int, act_func: Callable[[jnp.ndarray],jnp.ndarray], seed: int = 100):
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
        self.scheduler_weights = AdamMomentum(0.01, 0.9, 0.999, 0.001) #temporary
        self.scheduler_bias = AdamMomentum(0.01, 0.9, 0.999, 0.001) #temporary

        ## Initialize random weights and biases.
        self.reset_weights(seed)

    def reset_weights(self, seed):
        rand_key = random.PRNGKey(seed)
        self.weights = random.normal(key=rand_key, shape=self.weights_size)
        self.bias = random.normal(key=rand_key, shape=(self.bias_length,))*0.1

    def feed_forward(self, input: jnp.ndarray):
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
        #num_inputs = jnp.shape(input)[0]


        self.z = input @ self.weights + self.bias

        # calculate a, add bias
        output = self.act_func(self.z)

        return output

    def backpropagate(self, dC_doutput: jnp.ndarray, lmbd: float = 0.001):
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
        #self.grad_weights = 0
        input = self.input
        grad_act = vmap(vmap(derivate(self.act_func)))
        input_size = jnp.shape(input)

        ## Initialize weights and biases.
        #grad_weights = jnp.zeros(self.weights_size)
        #grad_biases = jnp.zeros(jnp.shape(self.bias))
        #grad_input = jnp.zeros(input_size)

        #dC_da = dC_doutput * grad_act(self.z)
        delta_matrix = dC_doutput * grad_act(self.z)
        grad_weights = input.T @ delta_matrix/input_size[0]
        grad_biases = jnp.sum(delta_matrix, axis=0).reshape(1,jnp.shape(delta_matrix)[1])/input_size[0]
        grad_input = delta_matrix@self.weights.T
        #grad_input = self.weights.T@delta_matrix

        # print(f"Before : {grad_weights}")
        grad_weights = grad_weights + self.weights * lmbd
        # print(f"After : {grad_weights}")

        #for i in range(self.input_length):
            #for j in range(self.output_length):
                ## Compute the gradients.
            #    grad_weights = grad_weights.at[i,j].set(jnp.sum(dC_doutput[:,j] * grad_act(self.z[:,j]) * input[:,i])/input_size[0])
            #    grad_biases = grad_biases.at[j].set(jnp.sum(1*grad_act(self.z[:,j])*dC_doutput[:,j])/input_size[0])

            #    zero_matrix = jnp.zeros(input_size)
            #    grad_input += zero_matrix.at[:,i].set(dC_doutput[:,j] * grad_act(self.z[:,j]) * self.weights[i,j])


        # scheduler_weights = Adam(0.001, 0.9, 0.999)
        self.scheduler_bias.reset()
        self.scheduler_weights.reset()
        self.weights -= self.scheduler_weights.update_change(grad_weights)
        # scheduler_bias = Adam(0.001, 0.9, 0.999)
        self.bias -= self.scheduler_bias.update_change(grad_biases)

        return grad_input
