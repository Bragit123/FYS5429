from scipy.signal import correlate2d, convolve2d
import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp
from funcs import RELU

class Convolution:
    def __init__(self, input_size, kernel_size, seed=100):
        self.input_size = input_size
        self.kernel_size = kernel_size
        
        self.input_depth, self.input_height, self.input_width = self.input_size
        self.num_kernels = self.kernel_size[0]
        self.kernel_height = self.kernel_size[2]
        self.kernel_width = self.kernel_size[3]
        
        self.bias_size = (self.num_kernels, self.input_height - self.kernel_height + 1, self.input_width - self.kernel_width + 1) # Equal to output_size

        rand_key = random.PRNGKey(seed)
        self.kernels = random.normal(key=rand_key, shape=self.kernel_size)
        self.bias = random.normal(key=rand_key, shape=self.bias_size) * 0.01

    
    def feed_forward(self, input):
        self.input = input
        num_inputs = jnp.shape(input)[0]
        output_size = (num_inputs,) + self.bias_size
        output = jnp.zeros(output_size)
        for n in range(num_inputs):
            for i in range(self.num_kernels):
                for c in range(self.input_depth):
                    corr = correlate2d(input[n,c,:,:], self.kernels[i,c,:,:], "valid") + self.bias[i,:,:]
                    output = output.at[n,i,:,:].set(jnp.sum(corr, axis=1))

        output = RELU(output)

        return output

    
    def backpropagate(self, dC_doutput, lmbd):
        input = self.input
        input_shape = jnp.shape(input)
        
        grad_kernel = jnp.zeros(self.kernel_size)
        grad_biases = jnp.zeros(self.bias_size)
        grad_input = jnp.zeros(input_shape)

        kernel_zeros = jnp.zeros(jnp.shape(grad_kernel))
        input_zeros = jnp.zeros(jnp.shape(grad_input))

        for n in range(input_shape[0]):
            for i in range(self.num_kernels):
                for d in range(self.input_depth):
                    grad_kernel += kernel_zeros.at[i,d,:,:].set(correlate2d(input[n,d,:,:], dC_doutput[n,i,:,:], "valid"))
                    grad_input += input_zeros.at[n,d,:,:].set(convolve2d(dC_doutput[n,i,:,:], self.kernels[d,i,:,:], "full"))
                
        grad_biases = jnp.sum(dC_doutput, axis=0)

        self.kernels -= grad_kernel * lmbd
        self.bias -= grad_biases * lmbd

        return grad_input