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
    
    def feed_forward(self, input):
        output = jnp.zeros(self.output_size)

        for i in range(self.output_height):
            for j in range(self.output_width):
                h_start = i*self.stride
                h_end = h_start + self.scale_height
                w_start = j*self.stride
                w_end = w_start + self.scale_width
                input_hw = input[:,:, h_start:h_end, w_start:w_end]
                output = output.at[:,:,i,j].set(jnp.max(input_hw, axis=(2,3)))

        return output