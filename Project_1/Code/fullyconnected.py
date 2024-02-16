
import matplotlib.pyplot as plt
from jax import random, vmap
import jax.numpy as jnp
from funcs import RELU, derivate


class FullyConnected:
    def __init__(self, input_length, output_length, seed=100):
        self.input_length = input_length
        self.output_length = output_length

        self.weights_size = (self.input_length, self.output_length)

        rand_key = random.PRNGKey(seed)
        self.weights = random.normal(key=rand_key, shape=self.weights_size)
        self.bias = random.normal(key=rand_key, shape=(self.output_length,)) * 0.01

        self.bias_length = self.bias.shape

        # self.reset_weights()

    def feed_forward(self, input):
        num_inputs = jnp.shape(input)[0]
        output = jnp.zeros((num_inputs, self.output_length))
        self.z = jnp.zeros((num_inputs, self.output_length))
        for i in range(num_inputs):
            for j in range(self.output_length):
                self.z = self.z.at[i,j].set(jnp.sum(self.weights[:,j]*input[i,:])+self.bias[j])
                output = output.at[i,j].set(self.z[i,j])

        output = RELU(output)

        return output


    def backpropagate(self, input, dC_doutput, lmbd):
        grad_act = derivate(RELU)
        input_size = jnp.shape(input)

        grad_weights = jnp.zeros(self.weights_size)
        grad_biases = jnp.zeros(self.bias_length)
        grad_input = jnp.zeros(input_size)

        for i in range(self.input_length):
            for j in range(self.output_length):
                grad_weights = grad_weights.at[i,j].set(jnp.sum(dC_doutput[:,j] * grad_act(self.z[:,j]) * input[:,i]))
                grad_biases = grad_biases.at[j].set(jnp.sum(1*grad_act(self.z[:,j])*dC_doutput[:,j]))
                grad_input += grad_input.at[:,i].set(dC_doutput[:,j] * grad_act(self.z[:,j] * self.weights[i,j]))

        self.weights -= grad_weights*lmbd # Need to implement scheduler
        self.bias -= grad_biases*lmbd

        return grad_input
