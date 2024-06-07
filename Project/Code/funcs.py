"""
The correlate4d function is developed by us, and so is the softmax and grad_softmax functions.
Except for that, the code is copied from the lecture notes by Morten Hjort-Jensen at the following link:
https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/exercisesweek43.html#cost-functions
The only change made to these functions are that we use jax instead of autograd for automatic differentiation.
"""

import jax.numpy as jnp
import numpy as np
from jax import grad

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


def grad_softmax(X):
    f = softmax(X)
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
