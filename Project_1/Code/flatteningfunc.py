import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp

class Flattened_Layer:
    def __init__(self):
        return None
    
    def feed_forward(self, input):
        self.input = input
        length_flattened = self.input_shape[1]*self.input_shape[2]*self.input_shape[3]

        flattened_output = jnp.reshape(input, (self.input_shape[0], length_flattened))

        return flattened_output
    
    def back_propagate(self, dC_doutput, lmbd=0):
        input_shape = jnp.shape(self.input)
        return jnp.reshape(dC_doutput, input_shape)