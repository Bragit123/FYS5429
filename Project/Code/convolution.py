from scipy.signal import correlate2d, convolve2d
import matplotlib.pyplot as plt
from jax import vmap
from funcs import derivate, RELU, padding
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
        self.bias_size = (int(np.floor((self.input_height - self.kernel_height)/stride)) + 1, int(np.floor((self.input_width - self.kernel_width)/stride)) + 1, self.num_kernels)
        ## Initialize kernels and biases.

    def reset_weights(self):
        seed = np.random.seed(self.seed)
        self.kernels = np.random.normal(size=self.kernel_size)
        self.bias = np.random.normal(size=self.bias_size) * 0.01
    
    def reset_schedulers(self):
        return 0
    
    def find_output_shape(self):
        return self.bias_size
    
    def feed_forward(self, input: np.ndarray):
        """
        Feeds input forward through the neural network.

        ## Parameters:
            - input (ndarray): Four-dimensional input array to be fed forward through
            the neural network. The four axes are:
                0: Different inputs.
                1: Rows.
                2: Columns.
                3: Input depth
        
        ## Returns:
            ndarray: Four-dimensional array containing the pooled output. This
            array has the same dimensions as the input, but the number of rows
            and columns should have decreased.
        """
        self.input = input
        num_inputs = np.shape(input)[0]
        output_size = (num_inputs,) + self.bias_size

        ## Initialize output array.
        z = np.zeros(output_size)

        for i in range(0, self.input_height - self.kernel_height +1, self.stride): #can change 1 with stride possibly
            for j in range(0, self.input_width - self.kernel_width + 1, self.stride):
                for d in range(self.num_kernels):
                    z[:, int(i/self.stride), int(j/self.stride), d] = np.sum(input[:, i : i + self.kernel_height, j : j + self.kernel_width, :] * self.kernels[d, :, :, :], axis=(1,2))[:,0]

        #for n in range(num_inputs):
            #for i in range(self.num_kernels):
                #for d in range(self.input_depth):
                    ## Correlate input with the kernels.
                    #corr = correlate2d(input[n,:,:,d], self.kernels[i,:,:,d], "valid") + self.bias[:,:,i]
                    #z[n,:,:,i] = np.sum(corr, axis=1)

        ## Compute output using activation function.
        self.z = z
        output = self.act_func(z)

        return output
    
    def backpropagate(self, dC_doutput: np.ndarray, lmbd: float = 0.01):
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
                1: Rows.
                2: Columns.
                3: Output depth.
            - lmbd (float): WILL BE CHANGED TO SCHEDULER.
        
        ## Returns
            ndarray: Partial derivatives of the cost function with respect to
            every input value to this layer. This array has the same dimensions
            as dC_doutput.
        """
        input = self.input
        input_shape = np.shape(input)
        
        ## Initialize gradients.
        grad_kernel = np.zeros(self.kernel_size)
        grad_biases = np.zeros(self.bias_size)
        grad_input = np.zeros(input_shape)

        grad_act = vmap(vmap(vmap(vmap(derivate(self.act_func)))))(self.z)
        # grad_act = vmap(vmap(derivate(self.act_func)))(self.z)
        delta_matrix = dC_doutput * grad_act

        # for i in range(0, self.input_height - self.kernel_height, self.stride): #can change 1 with stride possibly
        #     for j in range(0, self.input_width - self.kernel_width, self.stride):
        #         for d in range(self.input_depth):
        #             for c in range(self.num_kernels):
        #                 grad_kernel[c, :, :, d] = np.sum(input[:, i : i + self.kernel_height, j : j + self.kernel_width, d] * delta_matrix[:,i : i + self.kernel_height, j : j + self.kernel_width, c], axis=(0,1,2))
        #                 padded_delta = padding(delta_matrix, p=self.kernel_width-1)
        #                 delta = np.rot90(padded_delta, 2, axes=(1, 2)) #rotate 180 degrees
        #                 print(delta_matrix.shape)
        #                 grad_input[:,i,j,d] = np.sum(delta_matrix[:, i : i + self.kernel_height, j : j + self.kernel_width, c] * self.kernels[c, :, :, d], axis=(1,2))




        for n in range(input_shape[0]):
            for i in range(self.num_kernels):
                for d in range(self.input_depth):
                    # Compute gradients with respect to kernels and input.
                    grad_kernel[i,:,:,d] += correlate2d(input[n,:,:,d], delta_matrix[n,:,:,i], "valid")/input_shape[0]

    
        ## Compute the gradient with respect to biases.
        grad_biases = np.sum(delta_matrix, axis=0)/input_shape[0]


        delta_matrix = padding(delta_matrix, self.kernel_height - 1)
        delta_height = delta_matrix.shape[1]
        delta_width = delta_matrix.shape[2]

        for i in range(0, self.input_height - self.kernel_height + 1, self.stride): #can change 1 with stride possibly
            for j in range(0, self.input_width - self.kernel_width + 1, self.stride):
                for d in range(self.input_depth):
                    for k in range(self.num_kernels):

                        #z[:, i, j, d] = np.sum(input[:, i : i + self.kernel_height, j : j + self.kernel_width, :] * self.kernels[d, :, :, :], axis=(1,2))[:,0]

                        grad_input[:,i,j,d] += np.sum(delta_matrix[:,int(i/self.stride) : int(i/self.stride) + self.kernel_height, int(j/self.stride) : int(j/self.stride) + self.kernel_width,d]*self.kernels[k,:,:,d], axis=(1,2)) ##PS: add stride in this one

        #for i in range(0, self.input_height - self.kernel_height + 1, self.stride): #can change 1 with stride possibly
         #   for j in range(0, self.input_width - self.kernel_width + 1, self.stride):
          #      grad_input[:,i+1,j,:] = grad_input[:,i,j,:]
           #     grad_input[:,i,j+1,:] = grad_input[:,i,j,:]
            #    grad_input[:,i+1,j+1,:] = grad_input[:,i,j,:]

        ## Update the kernels and biases using gradient descent.
        self.kernels -= self.scheduler_kernel.update_change(grad_kernel)*lmbd
        self.bias -= self.scheduler_bias.update_change(grad_biases)*lmbd 

        return grad_input
