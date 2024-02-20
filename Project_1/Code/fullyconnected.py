
import matplotlib.pyplot as plt
from jax import random, vmap
import jax.numpy as jnp
from funcs import CostCrossEntropy, derivate


class FullyConnected:
    def __init__(self, input_size, output_size, seed=100):
        self.input_size = input_size
        self.output_size = output_size

        self.num_inputs, self.input_length = self.input_size
        self.num_outputs, self.output_length = self.output_size

        self.weights_size = (self.input_length, self.output_length)

        rand_key = random.PRNGKey(seed)
        self.weights = random.normal(key=rand_key, shape=self.weights_size)
        self.bias = random.normal(key=rand_key, shape=(self.output_length,1)) * 0.01

        self.bias_size = self.bias.shape

        self.loss_func = CostCrossEntropy #we should probably send in loss functions and make them the elements of a dictionary

        # self.reset_weights()

    def feed_forward(self, input):
        output = jnp.zeros(self.output_size)
        self.z = jnp.zeros(self.output_length)
        for i in range(self.num_outputs):
            for j in range(self.output_length):
                self.z = self.z.at[j].set(jnp.sum(self.weights[:,j]*input[i,:])+self.bias[j][0])
                output = output.at[i,j].set(self.z[j])

        output = self.RELU(output)

        return output

    def RELU(self, X):
        return jnp.where(X > jnp.zeros(X.shape), X, jnp.zeros(X.shape))


    def backpropagate(self, output, target, lmbd):
        cost = self.loss_func(target)
        grad_cost = vmap(derivate(cost))
        grad_act = derivate(self.RELU)

        grad_weights = jnp.zeros(self.weights_size)
        grad_biases = jnp.zeros(self.bias_size)

        for i in range(self.input_length):
            for j in range(self.output_length):
                grad_weights = grad_weights.at[i,j].set(jnp.sum(output[:,j]*grad_act(self.z[j])*grad_cost(output[:,j])))
                grad_biases = grad_biases.at[j,0].set(jnp.sum(1*grad_act(self.z[j])*grad_cost(output[:,j])))

        self.weights -= grad_weights*lmbd #Neew to implement scheduler
        self.bias -= grad_biases*lmbd
        return 0
