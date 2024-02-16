import jax.numpy as jnp

class MaxPool:
    def __init__(self, input_size, scale_factor, stride, seed=100):
        self.input_size = input_size
        self.num_inputs, self.input_depth, self.input_height, self.input_width = self.input_size
        self.scale_height, self.scale_width = scale_factor
        self.stride = stride

        self.output_height = int(jnp.floor((self.input_height - self.scale_height) / self.stride) + 1)
        self.output_width = int(jnp.floor((self.input_width - self.scale_width) / self.stride) + 1)
        self.output_size = (self.num_inputs, self.input_depth, self.output_height, self.output_width)
        self.output = jnp.zeros(self.output_size)

        self.max_ind = jnp.zeros(self.input_size)
    
    def feed_forward(self, input):
        self.input = input
        max_ind = jnp.zeros(self.input_size)
        output = jnp.zeros(self.output_size)
        argmax = jnp.zeros(self.input_size)

        for i in range(self.output_height):
            h_start = i*self.stride
            h_end = h_start + self.scale_height
            for j in range(self.output_width):
                w_start = j*self.stride
                w_end = w_start + self.scale_width
                input_hw = input[:,:, h_start:h_end, w_start:w_end]
                output_hw = jnp.max(input_hw, axis=(2,3))
                output = output.at[:,:,i,j].set(output_hw)

                compare = output_hw[:,:,jnp.newaxis,jnp.newaxis]
                max_ind_hw = jnp.where(compare == input_hw, 1, 0)

                zero_matrix = jnp.zeros(self.input_size)
                max_ind += zero_matrix.at[:,:, h_start:h_end, w_start:w_end].set(max_ind_hw) # Plus, to avoid overwriting previous ones.

        self.max_ind = jnp.where(max_ind > 0, 1, 0)

        return output
    
    def back_propagate(self, dC_doutput, lmbd):
        grad_input = jnp.zeros(jnp.shape(self.input))

        for i in range(self.output_height):
            h_start = i*self.stride
            h_end = h_start + self.scale_height
            for j in range(self.output_width):
                w_start = j*self.stride
                w_end = w_start + self.scale_width
                
                h_start + i
                
                grad_input[:,:, h_start:h_end, w_start:w_end] = dC_doutput[:,:,i,j] * self.max_ind[:,:,:,:]

                dC_ij = dC_doutput[:,:,i,j]
                new_dC = dC_ij[:,:,jnp.newaxis,jnp.newaxis]
                dC_hw = jnp.where(self.max_ind[:,:, h_start:h_end, w_start:w_end] == 1, new_dC, 0)
                zero_matrix = jnp.zeros(self.input_size)
                grad_input += zero_matrix.at[:,:, h_start:h_end, w_start:w_end].set(dC_hw) # Plus, to account for same max positions.
        
        return grad_input