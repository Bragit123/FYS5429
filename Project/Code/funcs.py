"""
This code is copied from the lecture notes by Morten Hjort-Jensen at the
following link:
https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/exercisesweek43.html#cost-functions
The only change made to these functions are that we use jax instead of autograd
for automatic differentiation.
"""

import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from scipy.signal import correlate2d, convolve2d
import time
# from numba import jit
# import warnings
# warnings.filterwarnings("ignore")

def CostOLS(target):

    def func(X):
        return (1.0 / target.shape[0]) * jnp.sum((target - X) ** 2)

    return func


#CostCrossEntropy(?)
def CostLogReg(target):

    def func(X):

        return -(1.0 / target.shape[0]) * jnp.sum(
            (target * jnp.log(X + 10e-10)) + ((1 - target) * jnp.log(1 - X + 10e-10))
        )

    return func


def CostCrossEntropy(target):

    def func(X):
        return -(1.0 / target.size) * jnp.sum(target * jnp.log(X + 10e-10))

    return func

# def CategoricalCrossEntropy(target):

#     def func(X):
#         return -jnp.sum(target * jnp.log(X + 10e-10))

#     return func

def identity(X):
    return X


def sigmoid(X):
    try:
        return 1.0 / (1 + jnp.exp(-X))
    except FloatingPointError:
        return jnp.where(X > jnp.zeros(X.shape), jnp.ones(X.shape), jnp.zeros(X.shape))


def softmax(X):
    X = X - np.max(X, axis=-1, keepdims=True)
    delta = 10e-10
    return np.exp(X) / (np.sum(np.exp(X), axis=-1, keepdims=True) + delta)
# def softmax(X, X_sum):
#     X = X - np.max(X, axis=-1, keepdims=True)
#     delta = 10e-10
#     return np.exp(X) / X_sum

def grad_softmax(X):
    f = softmax(X)
    #f @ f
    #f[i]*(i==j) - f[i]*f[j]
    return f - f**2
    


def RELU(X):
    return jnp.where(X > jnp.zeros(X.shape), X, jnp.zeros(X.shape))


def LRELU(X):
    delta = 10e-4
    return jnp.where(X > jnp.zeros(X.shape), X, delta * X)


def derivate(func):
    if func.__name__ == "RELU":

        def func(X):
            return jnp.where(X > 0, 1, 0)

        return func

    elif func.__name__ == "LRELU":

        def func(X):
            delta = 10e-4
            return jnp.where(X > 0, 1, delta)

        return func

    else:
        return grad(func)


def padding(X, p = 1):
    num_inputs, height, width, depth = X.shape
    padded_image = np.zeros((num_inputs, height + 2*p, width + 2*p, depth)) #X has 4 dimensions
    padded_image[:,p:-1-p+1,p:-1-p+1,:] = X
    return padded_image


# def correlate4d(X, K, mode: str = "XK"):
#     """
#     Correlates two 4D arrays, used for feed_forward and backpropagation in the
#     Convolution layer. This function will be used for three different "types" of
#     input, specified by the mode. They are mostly the same, but with slight
#     changes to how the data is structured, and therefore have to be dealt with
#     slightly differently.

#     Note: variable-naming is based on the XK case, as this was made first. Might
#     want to generalize to avoid confusion and increase readability for the two
#     other modes!

#     ## Input:
#         - X [num_inputs, height, width, depth]
#         - K [output_depth, height, width, depth]
#         - mode should be one of the following three:
#             - "XK" for input * kernel
#             - "Xd" for input * delta_matrix
#             - "dK" for full convolution delta_matrix * kernel
#     ## Output:
#         - C [num_inputs, output_depth, height, width, depth] where height/width
#           are for output/kernel/input for mode XK, Xd and dK respectively.
#     """
#     X_h = X.shape[1]
#     X_w = X.shape[2]
#     K_h = K.shape[1]
#     K_w = K.shape[2]
#     z_h = X_h - K_h + 1
#     z_w = X_w - K_w + 1
#     num_inputs = X.shape[0]
#     input_depth = X.shape[3]
#     num_kernels = K.shape[0]
#     kernel_depth = K.shape[3]

#     # Compute output dimensions (assuming "valid" convolution)
#     out_h = X_h - K_h + 1
#     out_w = X_w - K_w + 1

#     i0 = np.repeat(np.arange(K_h), K_w).reshape((-1, 1))
#     i1 = np.repeat(np.arange(z_h), z_w).reshape((1, -1))
#     j0 = np.tile(np.arange(K_w), K_h).reshape((-1, 1))
#     j1 = np.tile(np.arange(z_w), z_h).reshape((1, -1))

#     i = i0 + i1
#     j = j0 + j1

#     X = X[:,i,j,:] # Find indices for matrix multiplication X

#     if mode == "XK":
#         X = X.transpose((0,3,1,2)) # Get rows and cols as two last axes for matmul to work
#         X = X[:,np.newaxis,:,:,:] # Add axis to keep both num_elements from X and output_depth from K

#         K = K.reshape((num_kernels, K_h*K_w, kernel_depth)) # Get all row-col elements at one axis
#         K = K.transpose((0,2,1)) # Want row-col elements at rightmost for matrix multiplication
#         K = K[np.newaxis,:,:,np.newaxis,:] # First axis for same reason as in X, second axis for matmul to work

#         C = K @ X # Perform matrix multiplication. Dimension now is [num_inputs, output_depth, depth, output_elements]
#         C = C.reshape((num_inputs, num_kernels, kernel_depth, out_h, out_w)) # Un-flatten rows and cols of output
#         C = C.transpose((0,1,3,4,2)) # Move depth to last axis

#     elif mode == "Xd":
#         X = X.transpose((0,3,1,2)) # Get rows and cols as two last axes for matmul to work

#         K = K.reshape((num_kernels, K_h*K_w, kernel_depth)) # Get all row-col elements at one axis
#         K = K.transpose((0,2,1)) # Want row-col elements at rightmost for matrix multiplication
#         K = K[:,np.newaxis,:,:] # Add axis for matrix multiplication to work

#         C = K @ X # Perform matrix multiplication. Dimension now is [num_inputs, depth, output_depth, kernel_elements]
#         C = C.reshape((num_inputs, input_depth, kernel_depth, out_h, out_w)) # Un-flatten rows and cols of output
#         C = C.transpose((0,2,3,4,1)) # Move depth to last axis

#     elif mode == "dK":
#         X = X.transpose((0,3,1,2)) # Get rows and cols as two last axes for matmul to work

#         K = K.reshape((num_kernels, K_h*K_w, kernel_depth)) # Get all row-col elements at one axis
#         K = K.transpose((0,2,1)) # Want row-col elements at rightmost for matrix multiplication
#         K = K[np.newaxis,:,:,:] # Add axis for matrix multiplication to work

#         C = K @ X # Perform matrix multiplication. Dimension now is [num_inputs, output_depth, depth, input_elements]
#         C = C.reshape((num_inputs, num_kernels, kernel_depth, out_h, out_w)) # Un-flatten rows and cols of output
#         C = C.transpose((0,1,3,4,2)) # Move depth to last axis

#     else:
#         print(f"Error in correlate4d: mode = {mode} not recognized. Should be 'XK', 'Xd' or 'dK'.")
    

#     # Return convolved matrix. Axes (0,1,4) are (num_inputs, output_depth, :, :,
#     # depth). Axes (2,3) are height and width of output/kernel/input for mode
#     # XK, Xd and dK respectively.
#     return C



def correlate4d(A, B, mode: str = "XK"):
    """
    Correlates two 4D arrays, used for feed_forward and backpropagation in the
    Convolution layer. This function will be used for three different "types" of
    input, specified by the mode. They are mostly the same, but with slight
    changes to how the data is structured, and therefore have to be dealt with
    slightly differently.

    Note: variable-naming is based on the XK case, as this was made first. Might
    want to generalize to avoid confusion and increase readability for the two
    other modes!

    ## Input:
        - X [num_inputs, height, width, depth]
        - K [output_depth, height, width, depth]
        - mode should be one of the following three:
            - "XK" for input * kernel
            - "Xd" for input * delta_matrix
            - "dK" for full convolution delta_matrix * kernel
    ## Output:
        - C [num_inputs, output_depth, height, width, depth] where height/width
          are for output/kernel/input for mode XK, Xd and dK respectively.
    """
    # Find height and width
    A_h = A.shape[1]
    A_w = A.shape[2]
    B_h = B.shape[1]
    B_w = B.shape[2]
    
    # Compute output dimensions (assuming "valid" convolution)
    C_h = A_h - B_h + 1
    C_w = A_w - B_w + 1
    
    # num_inputs = X.shape[0]
    # input_depth = X.shape[3]
    # num_kernels = K.shape[0]
    # kernel_depth = K.shape[3]

    # out_h = X_h - K_h + 1
    # out_w = X_w - K_w + 1

    i0 = np.repeat(np.arange(B_h), B_w).reshape((-1, 1))
    i1 = np.repeat(np.arange(C_h), C_w).reshape((1, -1))
    j0 = np.tile(np.arange(B_w), B_h).reshape((-1, 1))
    j1 = np.tile(np.arange(C_w), C_h).reshape((1, -1))

    i = i0 + i1
    j = j0 + j1

    Aij = A[:,i,j,:] # Extended A-matrix for matrix multiplication

    if mode == "XK":
        Xij = Aij
        K = B
        Xij = Xij.transpose((0,3,1,2)) # Get rows and cols as two last axes for matmul to work
        Xij = Xij[:,np.newaxis,:,:,:] # Add axis to keep both num_elements from X and output_depth from K

        output_depth = K.shape[0]
        num_elements_K = B_h*B_w
        kernel_depth = K.shape[3]
        K_flat = K.reshape((output_depth, num_elements_K, kernel_depth)) # Get all row-col elements at one axis
        K_flat = K_flat.transpose((0,2,1)) # Want row-col elements at rightmost for matrix multiplication
        K_flat = K_flat[np.newaxis,:,:,np.newaxis,:] # First axis for same reason as in X, second axis for matmul to work

        num_inputs = A.shape[0]
        C = K_flat @ Xij # Perform matrix multiplication. Dimension now is [num_inputs, output_depth, depth, output_elements]
        C = C.reshape((num_inputs, output_depth, kernel_depth, C_h, C_w)) # Un-flatten rows and cols of output
        C = C.transpose((0,3,4,1,2)) # Move kernel_depth to last axis (this will be summed over in feed_forward)

    elif mode == "Xd":
        Xij = Aij
        delta = B
        Xij = Xij.transpose((0,3,1,2)) # Get rows and cols as two last axes for matmul to work

        num_inputs = A.shape[0]
        num_elements_delta = B_h*B_w
        output_depth = delta.shape[3]
        delta_flat = delta.reshape((num_inputs, num_elements_delta, output_depth)) # Get all row-col elements at one axis
        delta_flat = delta_flat.transpose((0,2,1)) # Want row-col elements at rightmost for matrix multiplication
        delta_flat = delta_flat[:,np.newaxis,:,:] # Add axis for matrix multiplication to work

        input_depth = A.shape[3]
        C = delta_flat @ Xij # Perform matrix multiplication. Dimension now is [num_inputs, depth, output_depth, kernel_elements]
        C = C.reshape((num_inputs, input_depth, output_depth, C_h, C_w)) # Un-flatten rows and cols of output
        C = C.transpose((0,2,3,4,1)) # Move depth to last axis

    elif mode == "dK":
        deltaij = Aij
        K = B
        deltaij = deltaij.transpose((0,3,1,2)) # Get rows and cols as two last axes for matmul to work

        output_depth = K.shape[0]
        num_elements_K = B_h*B_w
        kernel_depth = K.shape[3]
        K_flat = K.reshape((output_depth, num_elements_K, kernel_depth)) # Get all row-col elements at one axis
        K_flat = K_flat.transpose((0,2,1)) # Want row-col elements at rightmost for matrix multiplication
        K_flat = K_flat[np.newaxis,:,:,:] # Add axis for matrix multiplication to work

        num_inputs = A.shape[0]
        C = K_flat @ deltaij # Perform matrix multiplication. Dimension now is [num_inputs, output_depth, depth, input_elements]
        C = C.reshape((num_inputs, output_depth, kernel_depth, C_h, C_w)) # Un-flatten rows and cols of output
        C = C.transpose((0,3,4,2,1)) # Move depth to last axis

    else:
        print(f"Error in correlate4d: mode = {mode} not recognized. Should be 'XK', 'Xd' or 'dK'.")
    

    # Return convolved matrix. Axes (0,1,4) are (num_inputs, output_depth, :, :,
    # depth). Axes (2,3) are height and width of output/kernel/input for mode
    # XK, Xd and dK respectively.
    return C






# @jit
def convolve_forward(input, kernels, bias):
    #print("CONVOLVING")
    num_inputs = input.shape[0]
    num_kernels = kernels.shape[0]
    # input_depth = input.shape[3]
    bias_size = bias.shape
    output_size = (num_inputs,) + bias_size
    z = np.zeros(output_size)
    # for n in range(num_inputs):
    #     for i in range(num_kernels):
    #         for d in range(input_depth):
    #             # Correlate input with the kernels.
    #             corr = correlate2d(input[n,:,:,d], kernels[i,:,:,d], "valid") + bias[:,:,i]
    #             z[n,:,:,i] = np.sum(corr, axis=1)
    
    input_height = input.shape[1]
    input_width = input.shape[2]
    kernel_height = kernels.shape[1]
    kernel_width = kernels.shape[2]
    #print(input.shape)
    #print(kernels.shape)
    #print(bias.shape)
    start = time.time()
    for i in range(0, input_height - kernel_height + 1): #can change 1 with stride possibly
        for j in range(0, input_width - kernel_width + 1):
            for d in range(num_kernels):
                z[:, i, j, d] = np.sum(input[:, i : i + kernel_height, j : j + kernel_width, :] * kernels[d, :, :, :], axis=(1,2))[:,0]
                z[:, i, j, d] += bias[i,j,d]
    end = time.time()
    print(end-start)
    return z

# @jit
# def convolve_forward(input, kernels, bias):
#     print("CONVOLVING!")
#     num_inputs = input.shape[0]
#     num_kernels = kernels.shape[0]
#     # input_depth = input.shape[3]
#     bias_size = bias.shape
#     output_size = (num_inputs,) + bias_size
#     z = jnp.zeros(output_size)
#     # for n in range(num_inputs):
#     #     for i in range(num_kernels):
#     #         for d in range(input_depth):
#     #             # Correlate input with the kernels.
#     #             corr = correlate2d(input[n,:,:,d], kernels[i,:,:,d], "valid") + bias[:,:,i]
#     #             z[n,:,:,i] = np.sum(corr, axis=1)
    
#     input_height = input.shape[1]
#     input_width = input.shape[2]
#     kernel_height = kernels.shape[1]
#     kernel_width = kernels.shape[2]
#     for i in range(0, input_height - kernel_height + 1): #can change 1 with stride possibly
#         for j in range(0, input_width - kernel_width + 1):
#             for d in range(num_kernels):
#                 z.at[:, i, j, d].set(jnp.sum(input[:, i : i + kernel_height, j : j + kernel_width, :] * kernels[d, :, :, :], axis=(1,2))[:,0] + bias[i,j,d])

#     print("DONE CONVOLVING!")
#     return z





# # Backpropagate
# for n in range(input_shape[0]):
#     for i in range(self.num_kernels):
#         for d in range(self.input_depth):
#             # Compute gradients with respect to kernels and input.
#             grad_kernel[i,:,:,d] += correlate2d(input[n,:,:,d], delta_matrix[n,:,:,i], "valid")/input_shape[0]
#             grad_input[n,:,:,d] += convolve2d(delta_matrix[n,:,:,i], self.kernels[i,:,:,d], "full") ##PS: add stride in this one