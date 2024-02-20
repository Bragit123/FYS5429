import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp

class Flattened_Layer:
    """
    Flattened layer. Used for transforming the previous layer into one
    dimension, in order to run through a regular neural network.

    ## Attributes:
        - input_shape (ndarray): Shape of the input to the layer.
    """
    def __init__(self):
        """ Constructor """
        self.input_shape = None
    
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
            ndarray: Two-dimensional array containing the flattened output. The
            first axis is the same as the input, while the second output contains
            the flattened array of the three last axes of the input.
        """
        self.input_shape = jnp.shape(input) # Save input shape for use in backpropagate().
        length_flattened = self.input_shape[1]*self.input_shape[2]*self.input_shape[3]

        # Flattens the last three axes while keeping the first.
        flattened_output = jnp.reshape(input, (self.input_shape[0], length_flattened))

        return flattened_output
    
    def reset_weights(self, seed):
        return 0
    
    def backpropagate(self, dC_doutput: jnp.ndarray, lmbd: float = 0.01):
        """
        Backpropagates through the layer. Since this layer only reshapes the
        input, the corresponding backpropagation is only to reshape the
        flattened output back into the input shape.

        ## Parameters
            - dC_doutput (ndarray): Two-dimensional array containing the
              partial derivatives of the cost function with respect to every
              output value from this layer. The first axis is the different
              inputs, and the second axis corresponds to every partial
              derivative.
            - lmbd (float): WILL BE CHANGED TO SCHEDULER.
        
        ## Returns
            ndarray: Partial derivatives of the cost function with respect to
            every input value to this layer.
        """
        input_shape = jnp.shape(self.input)
        return jnp.reshape(dC_doutput, input_shape)