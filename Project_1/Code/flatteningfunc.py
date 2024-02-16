import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp

class Flattened_Layer:
    def __init__(self):
        return None
    
    def feed_forward(self, input):
        self.input = input
        input_shape = jnp.shape(self.input)
        length_flattened = input_shape[1]*input_shape[2]*input_shape[3]

        flattened_output = jnp.reshape(input, (input_shape[0], length_flattened))

        return flattened_output
    
    def backpropagate(self, dC_doutput, lmbd=0):
        input_shape = jnp.shape(self.input)
        return jnp.reshape(dC_doutput, input_shape)