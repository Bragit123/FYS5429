
import matplotlib.pyplot as plt
from jax import random, vmap
import jax.numpy as jnp
from funcs import RELU, derivate
from scheduler import AdamMomentum


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
    def __init__(self, input_length: int, output_length: int, seed: int = 100):
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

        ## Initialize random weights and biases.
        self.reset_weights(seed)

    def reset_weights(self, seed):
        rand_key = random.PRNGKey(seed)
        self.weights = random.normal(key=rand_key, shape=self.weights_size)
        self.bias = random.normal(key=rand_key, shape=(self.bias_length,1)) * 0.01

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
        num_inputs = jnp.shape(input)[0]

        ## Initialize output arrays.
        output = jnp.zeros((num_inputs, self.output_length))
        z = jnp.zeros((num_inputs, self.output_length))

        for i in range(num_inputs):
            for j in range(self.output_length):
                # Weighted sum
                z = z.at[i,j].set(jnp.sum(self.weights[:,j]*input[i,:])+self.bias[j,0])

        self.z = z
        output = RELU(self.z) # Run z through activation function.

        return output


    def backpropagate(self, dC_doutput: jnp.ndarray, lmbd: float = 0.01):
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
        grad_act = derivate(RELU)
        input_size = jnp.shape(input)

        ## Initialize weights and biases.
        grad_weights = jnp.zeros(self.weights_size)
        grad_biases = jnp.zeros(self.bias_length)
        grad_input = jnp.zeros(input_size)

        for i in range(self.input_length):
            for j in range(self.output_length):
                ## Compute the gradients.
                grad_weights = grad_weights.at[i,j].set(jnp.sum(dC_doutput[:,j] * grad_act(self.z[:,j]) * input[:,i]))
                grad_biases = grad_biases.at[j].set(jnp.sum(1*grad_act(self.z[:,j])*dC_doutput[:,j]))
                grad_input += grad_input.at[:,i].set(dC_doutput[:,j] * grad_act(self.z[:,j] * self.weights[i,j]))

        ## Update weights and biases using gradient descent.
        # self.weights -= grad_weights*lmbd # Need to implement scheduler
        # self.bias -= grad_biases*lmbd # Need to implement scheduler
        
        scheduler = AdamMomentum(0.01, 0.9, 0.999, 0.001)
        print("0weights = ", self.weights)
        print("0bias = ", self.bias)
        self.weights -= scheduler.update_change(grad_weights)
        self.bias -= scheduler.update_change(grad_biases)
        print("1weights = ", self.weights)
        print("1bias = ", self.bias)

        return grad_input
