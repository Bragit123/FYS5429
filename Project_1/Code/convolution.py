from scipy.signal import correlate2d, correlate
import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp

class Convolution:
    def __init__(self, input_size, kernel_size, seed=100):
        self.input_size = input_size
        self.kernel_size = kernel_size
        
        self.num_inputs, self.input_depth, self.input_height, self.input_width = self.input_size
        self.num_kernels = self.kernel_size[0]
        self.kernel_height = self.kernel_size[2]
        self.kernel_width = self.kernel_size[3]
        
        self.output_size = (self.num_inputs, self.num_kernels, self.input_height - self.kernel_height + 1, self.input_width - self.kernel_width + 1)
        self.bias_size = self.output_size[1:]

        rand_key = random.PRNGKey(seed)
        self.kernels = random.normal(key=rand_key, shape=self.kernel_size)
        self.bias = random.normal(key=rand_key, shape=self.bias_size) * 0.01
        self.output = jnp.zeros(self.output_size)

        # self.reset_weights()
    
    def feed_forward(self, input):
        output = jnp.zeros(self.output_size)
        for n in range(self.num_inputs):
            for i in range(self.num_kernels):
                for c in range(self.input_depth):
                    corr = correlate2d(input[n,c,:,:], self.kernels[i,c,:,:], "valid") + self.bias[i,:,:]
                    output = output.at[n,i,:,:].set(jnp.sum(corr, axis=1))

        output = self.RELU(output)

        return output
    
    def RELU(self, X):
        return jnp.where(X > jnp.zeros(X.shape), X, jnp.zeros(X.shape))

    
    def backpropagate(self):
        return 0
