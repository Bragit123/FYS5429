from scipy.signal import correlate2d, convolve2d
import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp
from funcs import RELU

class Convolution:
    """
    Convolution layer. Uses convolution with one or more kernels to detect
    patterns in the input data.

    ## Attributes:
        - input_size (tuple): Shape of input array containing four values, one
        for each dimension of input. The four tuple values are
            0: Input depth.
            1: Number of rows.
            2: Number of columns.
        - kernel_size (tuple): Shape of kernels array containing four values, one
        for each dimension of the kernel. The four tuple values are
            0: Number of kernels.
            1: Kernel depth (same as input depth).
            2: Number of kernel rows.
            3: Number of kernel columns.
        - input_depth (int): Depth of input.
        - input_height (int): Number of input rows.
        - input_width (int): Number of input columns.
        - num_kernels (int): Number of kernels.
        - kernel_height (int): Number of kernel rows.
        - kernel_width (int): Number of kernel columns.
        - bias_size (tuple): Shape of bias array containing three values. The
          three tuple values are
            0: Number of bias "matrices" (one for each kernel).
            1: Number of bias rows.
            2: Number of bias columns.
        - kernels (ndarray): Array of shape kernel_size containing all the
          kernels.
        - bias (ndarray): Array of shape bias_size containing all the biases.

    """
    def __init__(self, input_size: tuple, kernel_size: tuple, seed: int = 100):
        """
        Constructor

        ## Parameters:
        - input_size (tuple): Shape of input array containing four values, one
        for each dimension of input. The four tuple values are
            0: Number of inputs.
            1: Input depth.
            2: Number of rows.
            3: Number of columns.
        - kernel_size (tuple): Shape of kernels array containing four values, one
        for each dimension of the kernel. The four tuple values are
            0: Number of kernels.
            1: Kernel depth (same as input depth).
            2: Number of kernel rows.
            3: Number of kernel columns.
        - seed (int): Seed for generating random initial weights and biases in
          the layer.
        """
        self.input_size = input_size
        self.kernel_size = kernel_size
        
        self.input_depth, self.input_height, self.input_width = self.input_size
        self.num_kernels = self.kernel_size[0]
        self.kernel_height = self.kernel_size[2]
        self.kernel_width = self.kernel_size[3]
        
        ## Compute bias_size. This is equal to the output size.
        self.bias_size = (self.num_kernels, self.input_height - self.kernel_height + 1, self.input_width - self.kernel_width + 1)

        ## Initialize kernels and biases.
        self.reset_weights(seed)

    def reset_weights(self, seed):
        rand_key = random.PRNGKey(seed)
        self.kernels = random.normal(key=rand_key, shape=self.kernel_size)
        self.bias = random.normal(key=rand_key, shape=self.bias_size) * 0.01
    
    def feed_forward(self, input: jnp.ndarray):
        """
        Feeds input forward through the neural network.

        ## Parameters:
            - input (ndarray): Four-dimensional input array to be fed forward through
            the neural network. The four axes are:
                0: Different inputs.
                1: Input depth.
                2: Rows.
                3: Columns.
        
        ## Returns:
            ndarray: Four-dimensional array containing the pooled output. This
            array has the same dimensions as the input, but the number of rows
            and columns should have decreased.
        """
        self.input = input
        num_inputs = jnp.shape(input)[0]
        output_size = (num_inputs,) + self.bias_size

        ## Initialize output array.
        output = jnp.zeros(output_size)

        for n in range(num_inputs):
            for i in range(self.num_kernels):
                for c in range(self.input_depth):
                    ## Correlate input with the kernels.
                    corr = correlate2d(input[n,c,:,:], self.kernels[i,c,:,:], "valid") + self.bias[i,:,:]
                    output = output.at[n,i,:,:].set(jnp.sum(corr, axis=1))

        ## Compute output using activation function.
        output = RELU(output)

        return output

    
    def backpropagate(self, dC_doutput: jnp.ndarray, lmbd: float = 0.01):
        """
        Backpropagates through the layer to find the partial derivatives of the
        cost function with respect to each weight (kernel element), bias and
        input value. The derivatives with respect to weights and biases are used
        to update the weights and biases using gradient descent, while the
        derivatives with respect to the input is returned for use in
        backpropagation through previous layers.

        ## Parameters
            - dC_doutput (ndarray): Four-dimensional array containing the
              partial derivatives of the cost function with respect to every
              output value from this layer. The four axes are:
                0: Different inputs.
                1: Input depth.
                2: Rows.
                3: Columns.
            - lmbd (float): WILL BE CHANGED TO SCHEDULER.
        
        ## Returns
            ndarray: Partial derivatives of the cost function with respect to
            every input value to this layer. This array has the same dimensions
            as dC_doutput.
        """
        input = self.input
        input_shape = jnp.shape(input)
        
        ## Initialize gradients.
        grad_kernel = jnp.zeros(self.kernel_size)
        grad_biases = jnp.zeros(self.bias_size)
        grad_input = jnp.zeros(input_shape)

        kernel_zeros = jnp.zeros(jnp.shape(grad_kernel))
        input_zeros = jnp.zeros(jnp.shape(grad_input))

        for n in range(input_shape[0]):
            for i in range(self.num_kernels):
                for d in range(self.input_depth):
                    ## Compute gradients with respect to kernels and input.
                    grad_kernel += kernel_zeros.at[i,d,:,:].set(correlate2d(input[n,d,:,:], dC_doutput[n,i,:,:], "valid"))
                    grad_input += input_zeros.at[n,d,:,:].set(convolve2d(dC_doutput[n,i,:,:], self.kernels[d,i,:,:], "full"))
        
        ## Compute the gradient with respect to biases.
        grad_biases = jnp.sum(dC_doutput, axis=0)

        ## Update the kernels and biases using gradient descent.
        self.kernels -= grad_kernel * lmbd
        self.bias -= grad_biases * lmbd

        return grad_input