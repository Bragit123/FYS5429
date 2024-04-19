import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp

from layer import Layer

class FlattenedLayer(Layer):
    """
    Flattened layer. Used for transforming the previous layer into one
    dimension, in order to run through a regular neural network.

    ## Attributes:
        - input_shape (ndarray): Shape of the input to the layer (depth, height,
          width).
        - input_size (ndarray): Shape of the input to the layer, including how
          many inputs (inputs, depth, height, width).
    """
    def __init__(self, input_shape: tuple, seed: int = 100):
        """ Constructor """
        super().__init__(seed)
        self.input_shape = input_shape
        self.input_size = None
        self.num_inputs = None
    
    def feed_forward(self, input: jnp.ndarray):
        """
        Feeds input forward through the neural network.

        ## Parameters:
            - input (ndarray): Four-dimensional input array to be fed forward through
            the neural network. The four axes are:
                0: Different inputs.
                1: Rows.
                2: Columns.
                3: Input depth.
        
        ## Returns:
            ndarray: Two-dimensional array containing the flattened output. The
            first axis is the same as the input, while the second output contains
            the flattened array of the three last axes of the input.
        """
        self.num_inputs = input.shape[0]
        self.input_size = jnp.shape(input) # Save input shape for use in backpropagate().
        length_flattened = self.input_shape[0]*self.input_shape[1]*self.input_shape[2]

        # Flattens the last three axes while keeping the first.
        flattened_output = jnp.reshape(input, (self.num_inputs, length_flattened))

        return flattened_output
    
    def reset_weights(self):
        return 0
    
    def reset_schedulers(self):
        return 0

    def find_output_shape(self):
        output_length = self.input_shape[0]*self.input_shape[1]*self.input_shape[2]
        return output_length
    
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
        input_size = (self.num_inputs, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        
        dC_dinput = jnp.reshape(dC_doutput, input_size)
        return dC_dinput