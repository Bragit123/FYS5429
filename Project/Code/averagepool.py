import jax.numpy as jnp
import numpy as np
from layer import Layer

class AveragePool(Layer):
    """
    Averagepool layer. Reduces the size of the input in order to increase the
    efficiency of further computation. Reduces size by taking sections of the
    input and using only the average value.

    ## Attributes:
        - input_size (tuple): Shape of input array containing four values, one
        for each dimension of input. The four tuple values are
            0: Number of inputs.
            1: Number of rows.
            2: Number of columns.
            3: Input depth.
        - num_inputs (ndarray): Number of inputs.
        - input_depth (int): Depth of input.
        - input_height (int): Number of input rows.
        - input_width (int): Number of input columns.
        - scale_factor (int): Number of rows and columns of the pooling window.
          This value is an integer, as we only consider square pooling windows
          (equal number of rows and columns).
        - stride (int): How far the pooling window jumps at each iteration.
        - output_height (int): Number of output rows.
        - output_width (int): Number of output columns.
        - output_size (tuple): Shape of output array containing four values, one
          for each dimension of the output. The four tuple values are
            0: Number of inputs.
            1: Number of rows.
            2: Number of columns.
            3: Input depth.
    """
    def __init__(self, input_size: tuple, scale_factor: int, stride: int, seed: int = 100):
        """
        Constructor

        ## Parameters:
        - input_size (tuple): Shape of input array containing four values, one
        for each dimension of input. The four tuple values are
            0: Number of inputs.
            1: Number of rows.
            2: Number of columns.
            3: Input depth.
        - scale_factor (int): Number of rows and columns of the pooling window.
          This value is an integer, as we only consider square pooling windows
          (equal number of rows and columns).
        - stride (int): How far the pooling window jumps at each iteration.
        - seed (int): Seed for generating random initial weights and biases in
          the layer.
        """
        super().__init__(seed)
        self.input_size = input_size
        self.input_height, self.input_width, self.input_depth = self.input_size
        self.scale_factor = scale_factor
        self.stride = stride

        # Computing output size.

        self.output_height = int(jnp.floor((self.input_height - self.scale_factor) / self.stride) + 1)
        self.output_width = int(jnp.floor((self.input_width - self.scale_factor) / self.stride) + 1)

        self.output_size = None


    
    def reset_weights(self):
        return 0
    
    def reset_schedulers(self):
        return 0
    
    def find_output_shape(self):
        return (self.output_height, self.output_width, self.input_depth)

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
            ndarray: Four-dimensional array containing the pooled output. This
            array has the same dimensions as the input, but the number of rows
            and columns should have decreased.
        """
        self.input = input
        ## Initialize output
        num_inputs = self.input.shape[0]
        self.output_size = (num_inputs, self.output_height, self.output_width,  self.input_depth)
        output = np.zeros(self.output_size)

        for i in range(self.output_height):
            ## Define what indices to pool (placement of the pooling window) for this iteration.
            h_start = i*self.stride
            h_end = h_start + self.scale_factor
            for j in range(self.output_width):
                w_start = j*self.stride
                w_end = w_start + self.scale_factor
                input_hw = input[:, h_start:h_end, w_start:w_end, :]

                ## Find average within pooling window, and update output array.
                output_hw = np.average(input_hw, axis=(1,2))
                output[:,i,j,:] = output_hw 
        return output

    def backpropagate(self, dC_doutput: jnp.ndarray, lmbd: float = 0.01):
        """
        Backpropagates through the layer to find the partial derivatives of the
        cost function with respect to the input. Does this by checking which
        adding the contribution from each input element, to the gradient at the corresponding
        location. 

        ## Parameters
            - dC_doutput (ndarray): Four-dimensional array containing the
              partial derivatives of the cost function with respect to every
              output value from this layer. The four axes are:
                0: Different inputs.
                1: Rows.
                2: Columns.
                3: Input depth.
            - lmbd (float): WILL BE CHANGED TO SCHEDULER.
        
        ## Returns
            ndarray: Partial derivatives of the cost function with respect to
            every input value to this layer. This array has the same dimensions
            as dC_doutput, but will have more rows and columns.
        """
        ## Initialize input gradient.
        dC_output_scaled = np.asarray(dC_doutput)/(self.scale_factor*self.scale_factor)
        grad_input = np.zeros(self.input.shape)

        for i in range(self.output_height):
            ## Define what indices to look at (placement of the pooling window) for this iteration.
            h_start = i*self.stride
            h_end = h_start + self.scale_factor
            for j in range(self.output_width):
                w_start = j*self.stride
                w_end = w_start + self.scale_factor
                ## Find the gradient of the output corresponding to this pooling window and scale them.
                dC_ij = dC_output_scaled[:,i,j,:]
                ## Update the new gradient
                new_dC = dC_ij[:,np.newaxis,np.newaxis,:]
                grad_input[:,w_start:w_end, h_start:h_end,:] += new_dC # Plus, to account for multiple contributions
        
        return grad_input

