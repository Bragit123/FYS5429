from scipy.signal import correlate2d, convolve2d
import matplotlib.pyplot as plt
from jax import vmap
from funcs import derivate, RELU, padding, convolve_forward, correlate4d
import numpy as np
import jax.numpy as jnp
from numba import jit
import time
import sys
import warnings
warnings.filterwarnings("ignore")


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
        self.bias_size = (self.input_height, self.input_width, self.num_kernels) #same padding
        self.bias_size = (self.input_height - self.kernel_height + 1, self.input_width - self.kernel_width + 1, self.num_kernels) #valid padding
        ## Initialize kernels and biases.

    def reset_weights(self):
        seed = np.random.seed(self.seed)
        self.kernels = np.random.normal(size=self.kernel_size)
        self.bias = np.random.normal(size=self.bias_size) * 0.01
        # self.bias = np.zeros(self.bias_size)
    
    def reset_schedulers(self):
        return 0
    
    def find_output_shape(self):
        return self.bias_size
    
    # # def feed_forward(self, input: np.ndarray):
    #     """
    #     Feeds input forward through the neural network.

    #     ## Parameters:
    #         - input (ndarray): Four-dimensional input array to be fed forward through
    #         the neural network. The four axes are:
    #             0: Different inputs.
    #             1: Rows.
    #             2: Columns.
    #             3: Input depth
        
    #     ## Returns:
    #         ndarray: Four-dimensional array containing the pooled output. This
    #         array has the same dimensions as the input, but the number of rows
    #         and columns should have decreased.
    #     """
    #     self.input = input
    #     self.num_inputs = np.shape(input)[0]
    #     output_size = (self.num_inputs,) + self.bias_size




    #     i0 = np.repeat(np.arange(self.kernel_height), self.kernel_height)
    #     i1 = np.repeat(np.arange(self.input_height), self.input_height)
    #     j0 = np.tile(np.arange(self.kernel_width), self.kernel_height)
    #     j1 = np.tile(np.arange(self.input_height), self.input_width)
    #     self.i = i0.reshape(-1,1) + i1.reshape(1,-1)
    #     self.j = j0.reshape(-1,1) + j1.reshape(1,-1)
    #     self.k = np.repeat(np.arange(self.input_depth), self.kernel_height*self.kernel_width).reshape(-1,1)


    #     self.pad = 1 #int(np.floor((self.kernel_height - 1)/2))
    #     #pad_width = int((self.kernel_width - 1)/2)
    #     if (self.kernel_height%2 != 0):
    #         pad_img = np.pad(input,((0,0),(self.pad,self.pad),(self.pad,self.pad),(0,0)), mode="constant")
    #     else:
    #         pad_img = np.pad(input,((0,0),(self.pad + 1,self.pad),(self.pad + 1, self.pad),(0,0)), mode="constant")

    #     self.select_img = pad_img[:,self.i,self.j,:].squeeze()
    #     weights = self.kernels.reshape(self.kernel_height*self.kernel_width,-1)
    #     convolve = weights.transpose()@self.select_img
    #     z = convolve.reshape(output_size)

    #     ## Initialize output array.
    #     # z = np.zeros(output_size)
    #     # start = time.time()
    #     # for i in range(0, self.input_height - self.kernel_height + 1): #can change 1 with stride possibly
    #     #     for j in range(0, self.input_width - self.kernel_width + 1):
    #     #         for d in range(self.num_kernels):
    #     #             z[:, i, j, d] = np.sum(input[:, i : i + self.kernel_height, j : j + self.kernel_width, :] * self.kernels[d, :, :, :], axis=(1,2))[:,0]
    #     #             z[:, i, j, d] += self.bias[i,j,d]
    #     # end= time.time()
    #     # print(end-start)
    #     # for n in range(num_inputs):
    #     #     for i in range(self.num_kernels):
    #     #         for d in range(self.input_depth):
    #     #             # Correlate input with the kernels.
    #     #             corr = correlate2d(input[n,:,:,d], self.kernels[i,:,:,d], "valid") + self.bias[:,:,i]
    #     #             z[n,:,:,i] = np.sum(corr, axis=1)
    #     # input = jnp.array(input)
    #     # kernels = jnp.array(self.kernels)
    #     # bias = jnp.array(self.bias)
    #     # self.z = convolve_forward(input, kernels, bias)
    #     #self.z = convolve_forward(input, self.kernels, self.bias)

    #     ## Compute output using activation function.
    #     self.z = z
    #     output = self.act_func(self.z)

    #     return output
    
    # # def backpropagate(self, dC_doutput: np.ndarray, lmbd: float = 0.01):
    #     """
    #     Backpropagates through the layer to find the partial derivatives of the
    #     cost function with respect to each weight (kernel element), bias and
    #     input value. The derivatives with respect to weights and biases are used
    #     to update the weights and biases using gradient descent, while the
    #     derivatives with respect to the input is returned for use in
    #     backpropagation through previous layers.

    #     ## Parameters
    #         - dC_doutput (ndarray): Four-dimensional array containing the
    #           partial derivatives of the cost function with respect to every
    #           output value from this layer. The four axes are:
    #             0: Different inputs.
    #             1: Rows.
    #             2: Columns.
    #             3: Output depth.
    #         - lmbd (float): WILL BE CHANGED TO SCHEDULER.
        
    #     ## Returns
    #         ndarray: Partial derivatives of the cost function with respect to
    #         every input value to this layer. This array has the same dimensions
    #         as dC_doutput.
    #     """
    #     input = self.input
    #     input_shape = np.shape(input)
        
    #     ## Initialize gradients.
    #    # grad_kernel = np.zeros(self.kernel_size)
    #     #grad_biases = np.zeros(self.bias_size)
    #     #grad_input = np.zeros(input_shape)

    #     output_height = self.input_height + 2 * self.pad            #output height with padding
    #     output_width = self.input_width + 2 * self.pad            #output width with padding


    #     grad_act = vmap(vmap(vmap(vmap(derivate(self.act_func)))))(self.z)
    #     delta_matrix = dC_doutput * grad_act

    #     #print(grad_act)

    #     grad_bias = np.sum(delta_matrix, axis=0)/input_shape[0]             #bias gradient calculation
    #     #grad_bias = grad_bias.reshape(self.num_kernels, -1)

    #     delta_reshape = delta_matrix.squeeze().reshape(delta_matrix.shape[0], -1, self.num_kernels)  
    #     #print(delta_reshape)
    #     grad_weights = self.select_img @ delta_reshape   
    #     grad_weights = np.sum(grad_weights, axis = 0)/grad_weights.shape[0]   #Average over batches    
    #     grad_weights = grad_weights.reshape(self.kernels.shape)    #weight gradient caculation

    #     kernel_reshape=self.kernels.reshape(self.num_kernels, -1)
    #     #print(kernel_reshape.shape)
    #     X = (delta_reshape @ kernel_reshape).transpose(0,2,1)                 # gradient calculation w.r.t input image    
    #     #print(X.shape)

                             
        
    #     padded=np.zeros((self.num_inputs, output_height, output_width,  self.input_depth), dtype=X.dtype)  #empty padded array
    #     X_reshaped=X.reshape(self.num_inputs,self.kernel_height*self.kernel_width, self.input_height*self.input_width, self.input_depth)
    #     #print(X_reshaped.shape)
    #     #X_reshaped=X_reshaped.transpose(2,0,1)      
    #     #print(padded.shape)
    #     #print(X_reshaped.shape)             
    #     #print(self.i.shape, self.j.shape, self.k.shape)
    #     X_reshaped = np.sum(X_reshaped, axis = 1)
    #     #print(X_reshaped.shape)
    #     grad_input = X_reshaped.reshape(input_shape)
    #     #print(grad_input.shape)
    #     #np.add.at(padded, (slice(None), self.i, self.j, self.k), X_reshaped)  #gradient are stored in the corresponding locations
    #     #grad_input = padded[:,self.pad:-self.pad, self.pad:-self.pad,:]        #input image gradient

    #     #grad_act = vmap(vmap(vmap(vmap(derivate(self.act_func)))))(self.z)
    #     # grad_act = vmap(vmap(derivate(self.act_func)))(self.z)
    #     #delta_matrix = dC_doutput * grad_act

    #     # for i in range(0, self.input_height - self.kernel_height, self.stride): #can change 1 with stride possibly
    #     #     for j in range(0, self.input_width - self.kernel_width, self.stride):
    #     #         for d in range(self.input_depth):
    #     #             for c in range(self.num_kernels):
    #     #                 grad_kernel[c, :, :, d] = np.sum(input[:, i : i + self.kernel_height, j : j + self.kernel_width, d] * delta_matrix[:,i : i + self.kernel_height, j : j + self.kernel_width, c], axis=(0,1,2))
    #     #                 padded_delta = padding(delta_matrix, p=self.kernel_width-1)
    #     #                 delta = np.rot90(padded_delta, 2, axes=(1, 2)) #rotate 180 degrees
    #     #                 print(delta_matrix.shape)
    #     #                 grad_input[:,i,j,d] = np.sum(delta_matrix[:, i : i + self.kernel_height, j : j + self.kernel_width, c] * self.kernels[c, :, :, d], axis=(1,2))




    #     #for n in range(input_shape[0]):
    #     #    for i in range(self.num_kernels):
    #      #       for d in range(self.input_depth):
    #                 ## Compute gradients with respect to kernels and input.
    #       #          grad_kernel[i,:,:,d] += correlate2d(input[n,:,:,d], delta_matrix[n,:,:,i], "valid")/input_shape[0]
    #        #         grad_input[n,:,:,d] += convolve2d(delta_matrix[n,:,:,i], self.kernels[i,:,:,d], "full")


    
    #     ## Compute the gradient with respect to biases.
    #     #grad_biases = np.sum(delta_matrix, axis=0)/input_shape[0]


    #     #delta_matrix = padding(delta_matrix, self.kernel_height - 1)
    #     #delta_height = delta_matrix.shape[1]
    #     #delta_width = delta_matrix.shape[2]

    #     #for i in range(0, self.input_height - self.kernel_height + 1, self.stride): #can change 1 with stride possibly
    #      #   for j in range(0, self.input_width - self.kernel_width + 1, self.stride):
    #       #      for d in range(self.input_depth):
    #        #         for k in range(self.num_kernels):

    #                     #z[:, i, j, d] = np.sum(input[:, i : i + self.kernel_height, j : j + self.kernel_width, :] * self.kernels[d, :, :, :], axis=(1,2))[:,0]

    #         #            grad_input[:,i,j,d] += np.sum(delta_matrix[:,int(i/self.stride) : int(i/self.stride) + self.kernel_height, int(j/self.stride) : int(j/self.stride) + self.kernel_width,d]*self.kernels[k,:,:,d], axis=(1,2)) ##PS: add stride in this one

    #     #for i in range(0, self.input_height - self.kernel_height + 1, self.stride): #can change 1 with stride possibly
    #      #   for j in range(0, self.input_width - self.kernel_width + 1, self.stride):
    #       #      grad_input[:,i+1,j,:] = grad_input[:,i,j,:]
    #        #     grad_input[:,i,j+1,:] = grad_input[:,i,j,:]
    #         #    grad_input[:,i+1,j+1,:] = grad_input[:,i,j,:]

    #     ## Update the kernels and biases using gradient descent.
    #     if np.any(np.isnan(grad_weights)):
    #         print(grad_weights)

    #         sys.exit()
    #     self.kernels -= self.scheduler_kernel.update_change(grad_weights)*lmbd

    #     self.bias -= self.scheduler_bias.update_change(grad_bias)*lmbd 
        

    #     return grad_input
    
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
        grad_bias = np.sum(delta_matrix, axis=0)/input_shape[0] #bias gradient calculation

        ## Find gradient wrt kernels
        grad_kernels = correlate4d(input, delta_matrix, mode="Xd") ####### OBS OBS! DETTE GÅR KANSKJE IKKE!!!!
        grad_kernels = np.sum(grad_kernels, axis=0)/input_shape[0] # Normalized sum over inputs

        ## Find gradient wrt input
        # Pad delta_matrix and rotate kernel to do full convolution
        K_h = self.kernels.shape[1]
        K_w = self.kernels.shape[2]
<<<<<<< HEAD
        pad_top = int(np.ceil(K_h-1))
        pad_bot = int(np.floor(K_h-1))
        pad_left = int(np.ceil(K_w-1))
        pad_right = int(np.floor(K_w-1))
        delta_matrix_fc = np.pad(delta_matrix, ((0,0), (pad_top, pad_bot), (pad_left, pad_right), (0,0)))
=======
        pad_h = K_h-1
        pad_w = K_w-1
        # pad_top = int(np.ceil((K_h-1)/2))
        # pad_bot = int(np.floor((K_h-1)/2))
        # pad_left = int(np.ceil((K_w-1)/2))
        # pad_right = int(np.floor((K_w-1)/2))
        # delta_matrix_fc = np.pad(delta_matrix, ((0,0), (pad_top, pad_bot), (pad_left, pad_right), (0,0)))
        delta_matrix_fc = np.pad(delta_matrix, ((0,0), (pad_h, pad_h), (pad_w, pad_w), (0,0)))
>>>>>>> f533605cccf09f580ad628df47c7fa19c2cd4f8b
        kernels_rot = np.rot90(self.kernels, k=2, axes=(1,2))

        # Perform the full convolution
        grad_input = correlate4d(delta_matrix_fc, kernels_rot, mode="dK") ####### OBS OBS! DETTE GÅR KANSKJE IKKE!!!!
        grad_input = np.sum(grad_input, axis=-1) # Sum over output depth ####### OBS OBS! DETTE GÅR KANSKJE IKKE!!!!

        # Update kernels and biases
        self.kernels -= self.scheduler_kernel.update_change(grad_kernels)*lmbd
        self.bias -= self.scheduler_bias.update_change(grad_bias)*lmbd 

        return grad_input