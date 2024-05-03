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
# from numba import jit
# import warnings
# warnings.filterwarnings("ignore")

def CostOLS(target):

    def func(X):
        return (1.0 / target.shape[0]) * jnp.sum((target - X) ** 2)

    return func



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





# @jit
def convolve_forward(input, kernels, bias):
    print("CONVOLVING")
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
    print(input.shape)
    print(kernels.shape)
    print(bias.shape)
    for i in range(0, input_height - kernel_height + 1): #can change 1 with stride possibly
        for j in range(0, input_width - kernel_width + 1):
            for d in range(num_kernels):
                z[:, i, j, d] = np.sum(input[:, i : i + kernel_height, j : j + kernel_width, :] * kernels[d, :, :, :], axis=(1,2))[:,0]
                z[:, i, j, d] += bias[i,j,d]

    print("DONE CONVOLVING")
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