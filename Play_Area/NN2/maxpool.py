import jax.numpy as jnp

class MaxPool:
    """
    Maxpool layer. Reduces the size of the input in order to increase the
    efficiency of further computation. Reduces size by taking sections of the
    input and using only the maximum value.

    ## Attributes:
        - input_size (tuple): Shape of input array containing four values, one
        for each dimension of input. The four tuple values are
            0: Number of inputs.
            1: Input depth.
            2: Number of rows.
            3: Number of columns.
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
            1: Input depth.
            2: Number of rows.
            3: Number of columns.
        - max_ind (ndarray): Array of same shape as input. Used to keep track of
          where maximum values are retrieved from when pooling, in order to
          backpropagate.
    """
    def __init__(self, input_size: tuple, scale_factor: int, stride: int, seed: int = 100):
        """
        Constructor

        ## Parameters:
        - input_size (tuple): Shape of input array containing four values, one
        for each dimension of input. The four tuple values are
            0: Number of inputs.
            1: Input depth.
            2: Number of rows.
            3: Number of columns.
        - scale_factor (int): Number of rows and columns of the pooling window.
          This value is an integer, as we only consider square pooling windows
          (equal number of rows and columns).
        - stride (int): How far the pooling window jumps at each iteration.
        - seed (int): Seed for generating random initial weights and biases in
          the layer.
        """
        self.input_size = input_size
        self.num_inputs, self.input_depth, self.input_height, self.input_width = self.input_size
        self.scale_factor = scale_factor
        self.stride = stride

        # Computing output size.
        self.output_height = int(jnp.floor((self.input_height - self.scale_factor) / self.stride) + 1)
        self.output_width = int(jnp.floor((self.input_width - self.scale_factor) / self.stride) + 1)
        self.output_size = (self.num_inputs, self.input_depth, self.output_height, self.output_width)

        self.max_ind = jnp.zeros(self.input_size)
    
    def reset_weights(self, seed):
        return 0

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

        ## Initialize max_ind and output
        max_ind = jnp.zeros(self.input_size)
        output = jnp.zeros(self.output_size)

        for i in range(self.output_height):
            ## Define what indices to pool (placement of the pooling window) for this iteration.
            h_start = i*self.stride
            h_end = h_start + self.scale_factor
            for j in range(self.output_width):
                w_start = j*self.stride
                w_end = w_start + self.scale_factor
                input_hw = input[:,:, h_start:h_end, w_start:w_end]

                ## Find maximum within pooling window, and update output array.
                output_hw = jnp.max(input_hw, axis=(2,3))
                output = output.at[:,:,i,j].set(output_hw)

                ## Trace back where the maximum value(s) found place, and update max_ind accordingly.
                compare = output_hw[:,:,jnp.newaxis,jnp.newaxis]
                max_ind_hw = jnp.where(compare == input_hw, 1, 0)

                zero_matrix = jnp.zeros(self.input_size)
                max_ind += zero_matrix.at[:,:, h_start:h_end, w_start:w_end].set(max_ind_hw) # Plus, to avoid overwriting previous ones.

        # The pooling window might find the same maximum more than one time,
        # depending on the choice of stride and scale_factor. Change every
        # non-zero value in max_ind to 1 to avoid double counting.
        self.max_ind = jnp.where(max_ind > 0, 1, 0)

        return output
    
    def backpropagate(self, dC_doutput: jnp.ndarray, lmbd: float = 0.01):
        """
        Backpropagates through the layer to find the partial derivatives of the
        cost function with respect to the input. Does this by checking which
        input values are significant in the output, by checking what values are
        non-zero in max_ind.

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
            as dC_doutput, but will have more rows and columns.
        """
        ## Initialize input gradient.
        grad_input = jnp.zeros(jnp.shape(self.input))

        for i in range(self.output_height):
            ## Define what indices to look at (placement of the pooling window) for this iteration.
            h_start = i*self.stride
            h_end = h_start + self.scale_factor
            for j in range(self.output_width):
                w_start = j*self.stride
                w_end = w_start + self.scale_factor

                ## Find the gradient of the output corresponding to this pooling window.
                dC_ij = dC_doutput[:,:,i,j]

                ## Relate the output gradient value to the input values corresponding to the maximum in the pooling.
                new_dC = dC_ij[:,:,jnp.newaxis,jnp.newaxis]
                dC_hw = jnp.where(self.max_ind[:,:, h_start:h_end, w_start:w_end] == 1, new_dC, 0)
                
                ## Add the gradient value to gradient_input at the correct positions (where the input had a maximum). 
                zero_matrix = jnp.zeros(self.input_size)
                grad_input += zero_matrix.at[:,:, h_start:h_end, w_start:w_end].set(dC_hw) # Plus, to account for same max positions.
        
        return grad_input