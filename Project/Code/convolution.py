from jax import vmap
from funcs import derivate, correlate4d
import numpy as np


from typing import Callable
from copy import copy

from layer import Layer

class Convolution(Layer):
    """
    Convolution layer. Uses convolution with one or more kernels to detect
    patterns in the input data.

    ## Attributes:
        - input_size (tuple): Shape of input array containing four values, one
        for each dimension of input. The four tuple values are
            0: Number of rows.
            1: Number of columns.
            2: Input depth
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
    def __init__(self, input_size: tuple, kernel_size: tuple, act_func: Callable[[np.ndarray],np.ndarray], scheduler, stride = 1, seed: int = 100):
        """
        Constructor

        ## Parameters:
        - input_size (tuple): Shape of input array containing four values, one
        for each dimension of input. The four tuple values are
            0: Number of rows.
            1: Number of columns.
            2: Input depth
        - kernel_size (tuple): Shape of kernels array containing four values, one
        for each dimension of the kernel. The four tuple values are
            0: Number of kernels.
            1: Number of kernel rows.
            2: Number of kernel columns.
            3: Kernel depth (same as input depth).
        - seed (int): Seed for generating random initial weights and biases in
          the layer.
        """
        super().__init__(seed)
        self.input_size = input_size
        self.kernel_size = kernel_size
        
        self.input_height, self.input_width, self.input_depth = self.input_size
        self.num_kernels = self.kernel_size[0]
        self.kernel_height = self.kernel_size[1]
        self.kernel_width = self.kernel_size[2]

        self.act_func = act_func
        self.scheduler_kernel = copy(scheduler)
        self.scheduler_bias = copy(scheduler)

        self.stride = stride
        
        self.z = None
        ## Compute bias_size. This is equal to the output size.
        self.bias_size = (self.input_height - self.kernel_height + 1, self.input_width - self.kernel_width + 1, self.num_kernels) #valid padding
        ## Initialize kernels and biases.
        self.reset_weights()

    def reset_weights(self):
        seed = np.random.seed(self.seed)
        self.kernels = np.random.normal(size=self.kernel_size)
        self.bias = np.random.normal(size=self.bias_size) * 0.01
    
    def reset_schedulers(self):
        return 0
    
    def find_output_shape(self):
        return self.bias_size
    
    def feed_forward(self, input: np.ndarray):
        self.input = input
        self.num_inputs = np.shape(input)[0]

        C = correlate4d(input, self.kernels, mode="XK") # Dim = [num_inputs, output_depth, height, width, depth]
        z_sum = np.sum(C, axis=-1) # Sum over input_depth
        self.z = z_sum + self.bias
        output = self.act_func(self.z)

        return output
    
    def backpropagate(self, dC_doutput: np.ndarray, lmbd: float = 0.01):
        input = self.input
        input_shape = np.shape(input)

        grad_act = vmap(vmap(vmap(vmap(derivate(self.act_func)))))(self.z)
        delta_matrix = dC_doutput * grad_act

        ## Find gradient wrt biases
        grad_bias = np.sum(delta_matrix, axis=0)/input_shape[0] # bias gradient calculation

        ## Find gradient wrt kernels
        grad_kernels = correlate4d(input, delta_matrix, mode="Xd")
        grad_kernels = np.sum(grad_kernels, axis=0)/input_shape[0] # Normalized sum over inputs

        ## Find gradient wrt input
        # Pad delta_matrix and rotate kernel to do full convolution
        K_h = self.kernels.shape[1]
        K_w = self.kernels.shape[2]
        pad_h = K_h-1
        pad_w = K_w-1
        delta_matrix_fc = np.pad(delta_matrix, ((0,0), (pad_h, pad_h), (pad_w, pad_w), (0,0)))
        kernels_rot = np.rot90(self.kernels, k=2, axes=(1,2))

        # Perform the full convolution
        grad_input = correlate4d(delta_matrix_fc, kernels_rot, mode="dK")
        grad_input = np.sum(grad_input, axis=-1) # Sum over output depth

        # Update kernels and biases
        grad_kernels = grad_kernels + self.kernels * lmbd
        grad_bias = grad_bias + self.bias * lmbd
        self.kernels -= self.scheduler_kernel.update_change(grad_kernels)
        self.bias -= self.scheduler_bias.update_change(grad_bias)

        return grad_input